import copy as cp
import numpy as np
from multiprocessing import Pool
from .diagrams import DiagramBuilder
from .utils import Symmetrizer, AmplitudeUpdater, FMOCC_LOGGER

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
    def __init__(self):
        self.logger = FMOCC_LOGGER
        self.diagram_builder = DiagramBuilder()
        self.amplitude_updater = AmplitudeUpdater()
        self.symmetrizer = Symmetrizer()

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
            self.logger.info(f"CCSD converged at cycle {x+1}, correlation energy: {E_ccd}")
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
            If the calculation method is not 'CCSD' or 'ICCSD-PT'.
        """
        for x in range(n_iter):
            if calc == 'CCSD':
                self.logger.info(f"|| -------------- CCSD --------------- ||")
                pool = Pool(12)
                tau = cp.deepcopy(t2) 
                tau += np.einsum('ia,jb->ijab', t1, t1)
                result_comb_temp1 = pool.apply_async(self.diagram_builder.update1, args=(occ,nao,t1,t2,tau,Fock_mo,twoelecint_mo,))
                result_comb_temp2 = pool.apply_async(self.diagram_builder.update2, args=(occ,nao,t1,tau,twoelecint_mo,))
                result_comb_temp3 = pool.apply_async(self.diagram_builder.update10, args=(occ,nao,t1,t2,twoelecint_mo,))
                R_ijab3_temp = pool.apply_async(self.diagram_builder.update3,args=(occ,nao,tau,t1,t2,twoelecint_mo,))
                R_ijab4_temp = pool.apply_async(self.diagram_builder.update4,args=(occ,nao,t1,t2,twoelecint_mo,))
                R_ijab5_temp = pool.apply_async(self.diagram_builder.update5,args=(occ,virt,nao,t1,t2,twoelecint_mo,))
                R_ijab6_temp = pool.apply_async(self.diagram_builder.update6,args=(occ,virt,nao,t1,t2,twoelecint_mo,))
                R_ijab7_temp = pool.apply_async(self.diagram_builder.update7,args=(occ,nao,t1,t2,twoelecint_mo,))
                R_ijab8_temp = pool.apply_async(self.diagram_builder.update8,args=(occ,nao,t1,t2,twoelecint_mo,))
                R_ijab9_temp = pool.apply_async(self.diagram_builder.update9,args=(occ,nao,tau,twoelecint_mo,))
                pool.close()
                pool.join()

                R_ia1, R_ijab1 = result_comb_temp1.get()
                R_ia2, R_ijab2 = result_comb_temp2.get()
                R_ia10, R_ijab10 = result_comb_temp3.get()
                R_ijab3 = R_ijab3_temp.get()
                R_ijab4 = R_ijab4_temp.get()
                R_ijab5 = R_ijab5_temp.get()
                R_ijab6 = R_ijab6_temp.get()
                R_ijab7 = R_ijab7_temp.get()
                R_ijab8 = R_ijab8_temp.get()
                R_ijab9 = R_ijab9_temp.get()

                R_ia = (R_ia1+R_ia2+R_ia10)
                R_ijab = (R_ijab1+R_ijab2+R_ijab3+R_ijab4+R_ijab5+R_ijab6+R_ijab7+R_ijab8+R_ijab9+R_ijab10)

                R_ijab = self.symmetrizer.symmetrize(occ,virt,R_ijab)
                eps_t, t1, t2 = self.amplitude_updater.update_t1t2(R_ia,R_ijab,t1,t2,D1,D2)

                E_ccd = self.energy_ccsd(occ, nao, t1, t2, twoelecint_mo)
                val, E_ccd = self.convergence(E_ccd, E_old, eps_t, conv, x)
                if val:
                    self.logger.info(f"correlation energy: {E_ccd}")
                    break
                E_old = E_ccd

            if calc == 'ICCSD':
                self.logger.info(f"|| -------------- ICCSD --------------- ||")
                pool=Pool(12)
                tau = cp.deepcopy(t2)
                tau += np.einsum('ia,jb->ijab',t1,t1)
            
                II_oo = self.diagram_builder.So_int_diagrams(occ,o_act,nao,So,t2,twoelecint_mo)[1]
                II_vv = self.diagram_builder.Sv_int_diagrams(occ,virt,v_act,nao,Sv,t2,twoelecint_mo)[1]
            
                result_comb_temp1 = pool.apply_async(self.diagram_builder.update1,args=(occ,nao,t1,t2,tau,Fock_mo,twoelecint_mo,))
                result_comb_temp2 = pool.apply_async(self.diagram_builder.update2,args=(occ,nao,t1,tau,twoelecint_mo,))
                result_comb_temp3 = pool.apply_async(self.diagram_builder.update10,args=(occ,nao,t1,t2,twoelecint_mo,))
                R_ijab3_temp = pool.apply_async(self.diagram_builder.update3,args=(occ,nao,tau,t1,t2,twoelecint_mo,))
                R_ijab4_temp = pool.apply_async(self.diagram_builder.update4,args=(occ,nao,t1,t2,twoelecint_mo,))
                R_ijab5_temp = pool.apply_async(self.diagram_builder.update5,args=(occ,virt,nao,t1,t2,twoelecint_mo,))
                R_ijab6_temp = pool.apply_async(self.diagram_builder.update6,args=(occ,virt,nao,t1,t2,twoelecint_mo,))
                R_ijab7_temp = pool.apply_async(self.diagram_builder.update7,args=(occ,nao,t1,t2,twoelecint_mo,))
                R_ijab8_temp = pool.apply_async(self.diagram_builder.update8,args=(occ,nao,t1,t2,twoelecint_mo,))
                R_ijab9_temp = pool.apply_async(self.diagram_builder.update9,args=(occ,nao,tau,twoelecint_mo,))
                R_iuab_temp = pool.apply_async(self.diagram_builder.Sv_diagrams,args=(occ,v_act,nao,Sv,t1,t2,Fock_mo,twoelecint_mo,))
                R_ijav_temp = pool.apply_async(self.diagram_builder.So_diagrams,args=(occ,o_act,nao,So,t1,t2,Fock_mo,twoelecint_mo,))

                pool.close()
                pool.join()
            
                R_ia1, R_ijab1 = result_comb_temp1.get() 
                R_ia2, R_ijab2 = result_comb_temp2.get() 
                R_ia10, R_ijab10 = result_comb_temp3.get() 
                R_ijab3 = R_ijab3_temp.get()
                R_ijab4 = R_ijab4_temp.get()
                R_ijab5 = R_ijab5_temp.get()
                R_ijab6 = R_ijab6_temp.get()
                R_ijab7 = R_ijab7_temp.get()
                R_ijab8 = R_ijab8_temp.get()
                R_ijab9 = R_ijab9_temp.get()
            
                R_ia = (R_ia1+R_ia2+R_ia10)
                R_ijab = (R_ijab1+R_ijab2+R_ijab3+R_ijab4+R_ijab5+R_ijab6+R_ijab7+R_ijab8+R_ijab9+R_ijab10)
                R_ijab += self.diagram_builder.So_int_diagrams(occ,o_act,nao,So,t2,twoelecint_mo)[0]
                R_ijab += self.diagram_builder.Sv_int_diagrams(occ,virt,v_act,nao,Sv,t2,twoelecint_mo)[0]
                R_ijab = self.symmetrizer.symmetrize(occ,virt,R_ijab)
            
                R_iuab = R_iuab_temp.get()
                #R_iuab += diagrams.T1_contribution_Sv(occ,nao,v_act,t1,twoelecint_mo)
                #R_iuab += diagrams.coupling_terms_So(So,t2)[0]
                #R_iuab += diagrams.w2_int_2(So,Sv,t2)

                R_ijav = R_ijav_temp.get() 
                #R_ijav += diagrams.T1_contribution_So(occ,nao,o_act,t1,twoelecint_mo)
                #R_ijav += diagrams.coupling_terms_Sv(Sv,t2)[0]
                #R_ijav += diagrams.w2_int_1(So,Sv,t2)
            
            
                eps_t, t1, t2 = self.amplitude_updater.update_t1t2(R_ia,R_ijab,t1,t2,D1,D2)
                eps_So, So = self.amplitude_updater.update_So(R_ijav,So,Do,conv)
                eps_Sv, Sv = self.amplitude_updater.update_Sv(R_iuab,Sv,Dv,conv)

                E_ccd = self.energy_ccsd(occ,nao,t1,t2,twoelecint_mo)
                val, E_ccd = self.convergence_I(E_ccd,E_old,eps_t,eps_So,eps_Sv,conv,x)
                if val:
                    break
                else:
                    E_old = E_ccd
                    
            if calc == 'ICCSD-PT':
                self.logger.info(f"|| -------------- ICCSD-PT --------------- ||")
                pool=Pool(12)
                tau = cp.deepcopy(t2)
                tau += np.einsum('ia,jb->ijab',t1,t1)
            
                result_comb_temp1 = pool.apply_async(self.diagram_builder.update1,args=(occ,nao,t1,t2,tau,Fock_mo,twoelecint_mo,))
                result_comb_temp2 = pool.apply_async(self.diagram_builder.update2,args=(occ,nao,t1,tau,twoelecint_mo,))
                result_comb_temp3 = pool.apply_async(self.diagram_builder.update10,args=(occ,nao,t1,t2,twoelecint_mo,))
                R_ijab3_temp = pool.apply_async(self.diagram_builder.update3,args=(occ,nao,tau,t1,t2,twoelecint_mo,))
                R_ijab4_temp = pool.apply_async(self.diagram_builder.update4,args=(occ,nao,t1,t2,twoelecint_mo,))
                R_ijab5_temp = pool.apply_async(self.diagram_builder.update5,args=(occ,virt,nao,t1,t2,twoelecint_mo,))
                R_ijab6_temp = pool.apply_async(self.diagram_builder.update6,args=(occ,virt,nao,t1,t2,twoelecint_mo,))
                R_ijab7_temp = pool.apply_async(self.diagram_builder.update7,args=(occ,nao,t1,t2,twoelecint_mo,))
                R_ijab8_temp = pool.apply_async(self.diagram_builder.update8,args=(occ,nao,t1,t2,twoelecint_mo,))
                R_ijab9_temp = pool.apply_async(self.diagram_builder.update9,args=(occ,nao,tau,twoelecint_mo,))
            
                pool.close()
                pool.join()
            
                R_ia1, R_ijab1 = result_comb_temp1.get() 
                R_ia2, R_ijab2 = result_comb_temp2.get() 
                R_ia10, R_ijab10 = result_comb_temp3.get() 
                R_ijab3 = R_ijab3_temp.get()
                R_ijab4 = R_ijab4_temp.get()
                R_ijab5 = R_ijab5_temp.get()
                R_ijab6 = R_ijab6_temp.get()
                R_ijab7 = R_ijab7_temp.get()
                R_ijab8 = R_ijab8_temp.get()
                R_ijab9 = R_ijab9_temp.get()
            
                R_ia = (R_ia1+R_ia2+R_ia10)
                R_ijab = (R_ijab1+R_ijab2+R_ijab3+R_ijab4+R_ijab5+R_ijab6+R_ijab7+R_ijab8+R_ijab9+R_ijab10)
                R_ijab += self.diagram_builder.So_int_diagrams(occ,o_act,nao,So,t2,twoelecint_mo)[0]
                R_ijab += self.diagram_builder.Sv_int_diagrams(occ,virt,v_act,nao,Sv,t2,twoelecint_mo)[0]
                R_ijab = self.symmetrizer.symmetrize(occ, virt, R_ijab)

                eps_t, t1, t2 = self.amplitude_updater.update_t1t2(R_ia, R_ijab, t1, t2, D1, D2)
                E_ccd = self.energy_ccsd(occ, nao, t1, t2, twoelecint_mo)
                val, E_ccd = self.convergence(E_ccd, E_old, eps_t, conv, x)
                if val:
                    break
                else:
                    E_old = E_ccd
                    
        self.logger.info(f"CC Calculation completed after {x+1} iterations")
        return E_ccd, x