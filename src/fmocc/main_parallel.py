"""
Parallelized Coupled-Cluster (CC) driver in FMO framework.

This module implements the CCParallel class, which performs parallel CC calculations-
including CCSD, ICCSD, and ICCSD-PT - withing the FMO framework. It leverages Python's
multiprocessing to distribute diagram evaluations across multiple CPU cores, monitors
convergence, and updates CC amplitudes.

Key Responsibilities
--------------------
    - Manage parallel execution of CCSD, iCCSD, and iCCSD-PT energy evaluations.
    - Construct and combine diagrammatic contributions via DiagramBuilder.
    - Update single (t₁), double (t₂), and scattering amplitudes (Sₒ, Sᵥ).
    - Monitor convergence criteria for energy and amplitude updates.
    - Provide energy computation routines for CCSD, iCCSD, and iCCSD-PT variants.

Dependencies
-------------
    - Python standard libraries: copy, multiprocessing (Pool, cpu_count)
    - External library: numpy
    - Local modules: diagrams (DiagramBuilder), utils (Symmetrizer, AmplitudeUpdater, FMOCC_LOGGER)
"""
import os
import copy as cp
import numpy as np
import multiprocessing
from multiprocessing import Pool, cpu_count
from .diagrams import DiagramBuilder
from .utils import Symmetrizer, AmplitudeUpdater, FMOCC_LOGGER

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

try:
    multiprocessing.set_start_method('fork', force=True)
except RuntimeError:
    pass

def _execute_tasks(func, args):
    """Helper function to execute a function with given arguments."""
    return func(*args)

class CCParallel:
    """Parallel coupled-cluster (CC) calculator for FMO calculations.

    Manages the computation of CCSD and ICCSD energies using parallel processing
    for diagram evaluations.

    Attributes
    ----------
    logger : logging.Logger
        Logger instance for FMO calculations.
    diagram_builder : DiagramBuilder
        Object for computing CC diagram contributions.
    amplitude_updater : AmplitudeUpdater
        Object for updating CC amplitudes.
    symmetrizer : Symmetrizer
        Object for symmetrizing residuals.
    """
    def __init__(self, nproc: int = None):
        self.logger = FMOCC_LOGGER
        self.diagram_builder = DiagramBuilder()
        self.amplitude_updater = AmplitudeUpdater()
        self.symmetrizer = Symmetrizer()
        self.nproc = nproc if nproc and nproc > 0 else cpu_count()

    def energy_ccd(self, occ, nao, t2, twoelecint_mo):
        """Compute CCD correlation energy.

        Parameters
        ----------
        occ : int
            Number of occupied orbitals.
        nao : int
            Number of atomic orbitals.
        t2 : np.ndarray
            Double excitation amplitudes.
        twoelecint_mo : np.ndarray
            Two-electron integrals in MO basis.

        Returns
        -------
        float
            CCD correlation energy.
        """
        E_ccd = 2 * np.einsum('ijab,ijab', t2, twoelecint_mo[:occ, :occ, occ:nao, occ:nao]) - np.einsum('ijab,ijba', t2, twoelecint_mo[:occ, :occ, occ:nao, occ:nao])
        return E_ccd

    def energy_ccsd(self, occ, nao, t1, t2, twoelecint_mo):
        """Compute CCSD correlation energy, including single and double excitations.

        Parameters
        ----------
        occ : int
            Number of occupied orbitals.
        nao : int
            Number of atomic orbitals.
        t1 : np.ndarray
            Single excitation amplitudes.
        t2 : np.ndarray
            Double excitation amplitudes.
        twoelecint_mo : np.ndarray
            Two-electron integrals in MO basis.

        Returns
        -------
        float
            CCSD, iCCSD and iCCSD-PT correlation energy.
        """
        E_ccd = self.energy_ccd(occ, nao, t2, twoelecint_mo)
        E_ccd += 2 * np.einsum('ijab,ia,jb', twoelecint_mo[:occ, :occ, occ:nao, occ:nao], t1, t1) - np.einsum('ijab,ib,ja', twoelecint_mo[:occ, :occ, occ:nao, occ:nao], t1, t1)
        return E_ccd

    def convergence_I(self, E_ccd, E_old, eps_t, eps_So, eps_Sv, conv, x):
        """Check convergence for ICCSD calculations.

        Parameters
        ----------
        E_ccd : float
            Current correlation energy.
        E_old : float
            Previous correlation energy.
        eps_t : float
            Norm of t1 and t2 amplitude updates.
        eps_So : float
            Norm of So amplitude updates.
        eps_Sv : float
            Norm of Sv amplitude updates.
        conv : float
            Convergence threshold.
        x : int
            Current iteration number.

        Returns
        -------
        tuple[bool, float]
            A tuple containing a boolean indicating convergence and the current
            correlation energy.
        """
        del_E = E_ccd - E_old
        if all(abs(v) <= conv for v in [eps_t, eps_So, eps_Sv, del_E]):
            self.logger.info(f"ICCSD converged at cycle {x+1}, correlation energy: {E_ccd}")
            return True, E_ccd
        self.logger.info(f"Cycle {x+1}: t1+t2={eps_t}, So={eps_So}, Sv={eps_Sv}, ΔE={del_E}, E_corr={E_ccd}")
        return False, E_ccd

    def convergence(self, E_ccd, E_old, eps, conv, x):
        """Check convergence for CCSD calculations.

        Parameters
        ----------
        E_ccd : float
            Current correlation energy.
        E_old : float
            Previous correlation energy.
        eps : float
            Norm of t1 and t2 amplitude updates.
        conv : float
            Convergence threshold.
        x : int
            Current iteration number.

        Returns
        -------
        tuple[bool, float]
            A tuple containing a boolean indicating convergence and the current
            correlation energy.
        """
        del_E = E_ccd - E_old
        if abs(eps) <= conv and abs(del_E) <= conv:
            self.logger.info(f"[CONVERGENCE INFO] CCSD converged at cycle {x+1}, correlation energy: {E_ccd}")
            return True, E_ccd
        self.logger.info(f"Cycle {x+1}: t1+t2={eps}, ΔE={del_E}, E_corr={E_ccd}")
        return False, E_ccd

    def cc_calc(self, occ, virt, o_act, v_act, nao, t1, t2, So, Sv, D1, D2, Do, Dv, twoelecint_mo, Fock_mo, calc, n_iter, E_old, conv):
        """Perform coupled-cluster calculations (CCSD or ICCSD or ICCSD-PT).

        Parameters
        ----------
        occ : int
            Number of occupied orbitals.
        virt : int
            Number of virtual orbitals.
        o_act : int
            Number of active occupied orbitals.
        v_act : int
            Number of active virtual orbitals.
        nao : int
            Number of atomic orbitals.
        t1 : np.ndarray
            Single excitation amplitudes.
        t2 : np.ndarray
            Double excitation amplitudes.
        So : np.ndarray
            Occupied orbital correction amplitudes.
        Sv : np.ndarray
            Virtual orbital correction amplitudes.
        D1 : np.ndarray
            Denominator for t1 amplitude updates.
        D2 : np.ndarray
            Denominator for t2 amplitude updates.
        Do : np.ndarray
            Denominator for So amplitude updates.
        Dv : np.ndarray
            Denominator for Sv amplitude updates.
        twoelecint_mo : np.ndarray
            Two-electron integrals in MO basis.
        Fock_mo : np.ndarray
            Fock matrix in MO basis.
        calc : str
            Calculation method ('CCSD' or 'ICCSD-PT').
        n_iter : int
            Maximum number of iterations.
        E_old : float
            Initial correlation energy.
        conv : float
            Convergence threshold.

        Returns
        -------
        tuple[float, int]
            A tuple containing the final correlation energy and the number of
            iterations performed.

        Raises
        ------
        ValueError
            If the calculation method is not 'CCSD', 'ICCSD', or 'ICCSD-PT'.
        """
        self.logger.info(f"[CALC INFO] Starting {calc} calculation with {self.nproc} processors")
        tasks_per_iter = 10
        if calc == 'ICCSD':
            tasks_per_iter = 12

        nproc = min(self.nproc, tasks_per_iter)
        with Pool(processes=nproc, maxtasksperchild=10) as pool:
            for x in range(n_iter):
                tau = np.empty_like(t2)
                np.einsum('ia,jb->ijab', t1, t1, out=tau)
                tau += t2
                if calc == 'CCSD':
                    if x%2 == 0:
                        self.logger.info(f"|| -------------- CCSD --------------- ||")
                    tasks = [
                        (self.diagram_builder.update1, (occ, nao, t1, t2, tau, Fock_mo, twoelecint_mo)),
                        (self.diagram_builder.update2, (occ, nao, t1, tau, twoelecint_mo)),
                        (self.diagram_builder.update10, (occ, nao, t1, t2, twoelecint_mo)),
                        (self.diagram_builder.update3, (occ, nao, tau, t1, t2, twoelecint_mo)),
                        (self.diagram_builder.update4, (occ, nao, t1, t2, twoelecint_mo)),
                        (self.diagram_builder.update5, (occ, virt, nao, t1, t2, twoelecint_mo)),
                        (self.diagram_builder.update6, (occ, virt, nao, t1, t2, twoelecint_mo)),
                        (self.diagram_builder.update7, (occ, nao, t1, t2, twoelecint_mo)),
                        (self.diagram_builder.update8, (occ, nao, t1, t2, twoelecint_mo)),
                        (self.diagram_builder.update9, (occ, nao, tau, twoelecint_mo)),
                    ]
                    results = pool.starmap_async(_execute_tasks, tasks).get()

                    R_ia1, R_ijab1 = results[0]
                    R_ia2, R_ijab2 = results[1]
                    R_ia10, R_ijab10 = results[2]
                    R_ijab3 = results[3]
                    R_ijab4 = results[4]
                    R_ijab5 = results[5]
                    R_ijab6 = results[6]
                    R_ijab7 = results[7]
                    R_ijab8 = results[8]
                    R_ijab9 = results[9]

                    R_ia = (R_ia1 + R_ia2 + R_ia10)
                    R_ijab = (R_ijab1 + R_ijab2 + R_ijab3 + R_ijab4 + R_ijab5 + R_ijab6 + R_ijab7 + R_ijab8 + R_ijab9 + R_ijab10)
                    R_ijab = self.symmetrizer.symmetrize(occ, virt, R_ijab)
                    eps_t, t1, t2 = self.amplitude_updater.update_t1t2(R_ia, R_ijab, t1, t2, D1, D2)
                    E_ccd = self.energy_ccsd(occ, nao, t1, t2, twoelecint_mo)
                    val, E_ccd = self.convergence(E_ccd, E_old, eps_t, conv, x)
                    if val:
                        self.logger.info(f"correlation energy: {E_ccd}")
                        break
                    E_old = E_ccd

                if calc == 'ICCSD':
                    if x%2 == 0:
                        self.logger.info(f"|| -------------- ICCSD --------------- ||")
                    tasks = [
                        (self.diagram_builder.update1, (occ, nao, t1, t2, tau, Fock_mo, twoelecint_mo)),
                        (self.diagram_builder.update2, (occ, nao, t1, tau, twoelecint_mo)),
                        (self.diagram_builder.update10, (occ, nao, t1, t2, twoelecint_mo)),
                        (self.diagram_builder.update3, (occ, nao, tau, t1, t2, twoelecint_mo)),
                        (self.diagram_builder.update4, (occ, nao, t1, t2, twoelecint_mo)),
                        (self.diagram_builder.update5, (occ, virt, nao, t1, t2, twoelecint_mo)),
                        (self.diagram_builder.update6, (occ, virt, nao, t1, t2, twoelecint_mo)),
                        (self.diagram_builder.update7, (occ, nao, t1, t2, twoelecint_mo)),
                        (self.diagram_builder.update8, (occ, nao, t1, t2, twoelecint_mo)),
                        (self.diagram_builder.update9, (occ, nao, tau, twoelecint_mo)),
                        (self.diagram_builder.Sv_diagrams, (occ, v_act, nao, Sv, t2, Fock_mo, twoelecint_mo)),
                        (self.diagram_builder.So_diagrams, (occ, o_act, nao, So, t2, Fock_mo, twoelecint_mo)),
                    ]
                    results = pool.starmap_async(_execute_tasks, tasks).get()

                    R_ia1, R_ijab1 = results[0]
                    R_ia2, R_ijab2 = results[1]
                    R_ia10, R_ijab10 = results[2]
                    R_ijab3 = results[3]
                    R_ijab4 = results[4]
                    R_ijab5 = results[5]
                    R_ijab6 = results[6]
                    R_ijab7 = results[7]
                    R_ijab8 = results[8]
                    R_ijab9 = results[9]
                    R_iuab = results[10]
                    R_ijav = results[11]

                    R_ia = (R_ia1 + R_ia2 + R_ia10)
                    R_ijab = (R_ijab1 + R_ijab2 + R_ijab3 + R_ijab4 + R_ijab5 + R_ijab6 + R_ijab7 + R_ijab8 + R_ijab9 + R_ijab10)
                    R_ijab += self.diagram_builder.So_int_diagrams(occ, o_act, nao, So, t2, twoelecint_mo)[0]
                    R_ijab += self.diagram_builder.Sv_int_diagrams(occ, virt, v_act, nao, Sv, t2, twoelecint_mo)[0]
                    R_ijab = self.symmetrizer.symmetrize(occ, virt, R_ijab)

                    eps_t, t1, t2 = self.amplitude_updater.update_t1t2(R_ia, R_ijab, t1, t2, D1, D2)
                    eps_So, So = self.amplitude_updater.update_So(R_ijav, So, Do, conv)
                    eps_Sv, Sv = self.amplitude_updater.update_Sv(R_iuab, Sv, Dv, conv)

                    E_ccd = self.energy_ccsd(occ, nao, t1, t2, twoelecint_mo)
                    val, E_ccd = self.convergence_I(E_ccd, E_old, eps_t, eps_So, eps_Sv, conv, x)
                    if val:
                        break
                    else:
                        E_old = E_ccd

                if calc == 'ICCSD-PT':
                    if x%2 == 0:
                        self.logger.info(f"|| -------------- ICCSD-PT --------------- ||")
                    tasks = [
                        (self.diagram_builder.update1, (occ, nao, t1, t2, tau, Fock_mo, twoelecint_mo)),
                        (self.diagram_builder.update2, (occ, nao, t1, tau, twoelecint_mo)),
                        (self.diagram_builder.update10, (occ, nao, t1, t2, twoelecint_mo)),
                        (self.diagram_builder.update3, (occ, nao, tau, t1, t2, twoelecint_mo)),
                        (self.diagram_builder.update4, (occ, nao, t1, t2, twoelecint_mo)),
                        (self.diagram_builder.update5, (occ, virt, nao, t1, t2, twoelecint_mo)),
                        (self.diagram_builder.update6, (occ, virt, nao, t1, t2, twoelecint_mo)),
                        (self.diagram_builder.update7, (occ, nao, t1, t2, twoelecint_mo)),
                        (self.diagram_builder.update8, (occ, nao, t1, t2, twoelecint_mo)),
                        (self.diagram_builder.update9, (occ, nao, tau, twoelecint_mo)),
                    ]
                    results = pool.starmap_async(_execute_tasks, tasks).get() 

                    R_ia1, R_ijab1 = results[0]
                    R_ia2, R_ijab2 = results[1]
                    R_ia10, R_ijab10 = results[2]
                    R_ijab3 = results[3]
                    R_ijab4 = results[4]
                    R_ijab5 = results[5]
                    R_ijab6 = results[6]
                    R_ijab7 = results[7]
                    R_ijab8 = results[8]
                    R_ijab9 = results[9]

                    R_ia = (R_ia1 + R_ia2 + R_ia10)
                    R_ijab = (R_ijab1 + R_ijab2 + R_ijab3 + R_ijab4 + R_ijab5 + R_ijab6 + R_ijab7 + R_ijab8 + R_ijab9 + R_ijab10)
                    R_ijab += self.diagram_builder.So_int_diagrams(occ, o_act, nao, So, t2, twoelecint_mo)[0]
                    R_ijab += self.diagram_builder.Sv_int_diagrams(occ, virt, v_act, nao, Sv, t2, twoelecint_mo)[0]
                    R_ijab = self.symmetrizer.symmetrize(occ, virt, R_ijab)

                    eps_t, t1, t2 = self.amplitude_updater.update_t1t2(R_ia, R_ijab, t1, t2, D1, D2)
                    E_ccd = self.energy_ccsd(occ, nao, t1, t2, twoelecint_mo)
                    val, E_ccd = self.convergence(E_ccd, E_old, eps_t, conv, x)
                    if val:
                        break
                    else:
                        E_old = E_ccd
                    
        self.logger.info(f"[CALC INFO] CC Calculation completed after {x+1} iterations")
        return E_ccd, x