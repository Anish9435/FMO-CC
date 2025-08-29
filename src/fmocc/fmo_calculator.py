import numpy as np
from .MP2 import MP2Calculator
from .main_parallel import CCParallel
from itertools import combinations
from .utils import FMOCC_LOGGER

class FMOCalculator:
    """Calculator for Fragment Molecular Orbital (FMO) energy computations.

    Handles the computation of RHF, MP2, and CC energies for monomers and dimers,
    including transformations of two-electron integrals.

    Parameters
    ----------
    config : FMOConfig
        Configuration object for FMO calculations.
    extractor : FMOExtractor
        Object for extracting data from GAMESS output files.
    processor : FMOProcessor
        Processor object managing the FMO calculation workflow.

    Attributes
    ----------
    logger : logging.Logger
        Logger instance for FMO calculations.
    config : FMOConfig
        Configuration object for FMO calculations.
    extractor : FMOExtractor
        Object for extracting data from GAMESS output files.
    processor : FMOProcessor
        Processor object managing the FMO calculation workflow.
    mp2_calc : MP2Calculator
        Object for MP2 energy calculations.
    cc_parallel : CCParallel
        Object for coupled-cluster calculations.
    """
    def __init__(self, config, extractor, processor):
        self.logger = FMOCC_LOGGER
        self.config = config
        self.extractor = extractor
        self.processor = processor
        self.mp2_calc = MP2Calculator()
        self.cc_parallel = CCParallel()

    def _transform_2eint(self, coeff, twoeint):
        """Transform two-electron integrals from AO to MO basis.

        Parameters
        ----------
        coeff : np.ndarray
            Coefficient matrix for basis transformation.
        twoeint : np.ndarray
            Two-electron integrals in AO basis.

        Returns
        -------
        np.ndarray
            Two-electron integrals in MO basis.

        Raises
        ------
        RuntimeError
            If an error occurs during the transformation process.
        """
        try:
            twoeint_1 = np.einsum('zs,wxyz->wxys', coeff, twoeint)
            twoeint_2 = np.einsum('yr,wxys->wxrs', coeff, twoeint_1)
            twoeint_3 = np.einsum('xq,wxrs->wqrs', coeff, twoeint_2)
            twoelecint_mo = np.einsum('wp,wqrs->pqrs', coeff, twoeint_3)
            twoelecint_mo = np.swapaxes(twoelecint_mo, 1, 2)
            del twoeint_1, twoeint_2, twoeint_3
            return twoelecint_mo
        except Exception as e:
            raise RuntimeError(f"Error transforming 2e integrals: {e}")

    def compute_monomer(self, i, lnum1, lnum2):
        """Compute energies for a single monomer.

        Parameters
        ----------
        i : int
            Index of the monomer.
        lnum1 : int
            Line number index for GAMESS output file.
        lnum2 : int
            Line number index for GAMESS 2e integral file.

        Returns
        -------
        tuple[float, float, float, int, int]
            A tuple containing RHF energy, MP2 correlation energy, CC correlation
            energy, updated lnum1, and updated lnum2.
        """
        hamiltonian_file = f"hamiltonian{i}.txt"
        twoelecint_file = f"twoelecint{i}.txt"
        coeff_file = f"coeff{i}.txt"
        temp_file = "twoelecint_test.txt"
        twoelecintegral_file = "twoelecintegral.txt"

        ifrag, _, Erhf = self.extractor.bare_hamiltonian(lnum1, 1, hamiltonian_file)
        frag_idx = ifrag -1
        
        nao = self.config.nao_mono[frag_idx]
        nmo = self.config.nmo_mono[frag_idx]
        occ = self.config.occ_mono[frag_idx]
        virt = self.config.virt_mono[frag_idx]
        o_act = self.config.o_act_mono[frag_idx]
        v_act = self.config.v_act_mono[frag_idx]
        nfo = self.config.nfo_mono[frag_idx]
        nfv = self.config.nfv_mono[frag_idx]
        self.logger.info(f"Monomer {ifrag}: RHF energy: {Erhf} with nao: {nao}")

        Fock = self.extractor.get_1e_parameter(nao, hamiltonian_file)
        ifrag, _, Erhf, lnum2 = self.extractor.twoelecint(lnum2, 1, twoelecint_file)
        self.extractor.twoelecint_process(twoelecint_file, temp_file)
        self.extractor.bash_run()
        twoeint = self.extractor.read_2eint(nao, twoelecintegral_file)
        ifrag, _, Erhf, lnum1 = self.extractor.coeff(lnum1, 1, coeff_file)
        coeff = self.extractor.get_coeff(nmo, nao, coeff_file)
        hf_mo_E = self.extractor.get_orb_energy(nao, nmo, coeff_file)
        Fock_mo = np.diag(hf_mo_E)

        twoelecint_mo = self._transform_2eint(coeff, twoeint)

        if nfo > 0:
            occ, nmo, twoelecint_mo, Fock_mo, hf_mo_E = self.mp2_calc.occ_frozen(occ, nmo, nfo, twoelecint_mo, Fock_mo, hf_mo_E)
        if nfv > 0:
            nao, nmo, virt, twoelecint_mo, Fock_mo, hf_mo_E = self.mp2_calc.virt_frozen(virt, nmo, nfo, nfv, nao, twoelecint_mo, Fock_mo, hf_mo_E)

        t2, D2 = self.mp2_calc.guess_t2(occ, virt, nmo, hf_mo_E, twoelecint_mo)
        t1, D1 = self.mp2_calc.guess_t1(occ, virt, nmo, hf_mo_E, Fock_mo)
        So, Do = self.mp2_calc.guess_so(occ, virt, nmo, o_act, hf_mo_E, twoelecint_mo)
        Sv, Dv = self.mp2_calc.guess_sv(occ, virt, nmo, v_act, hf_mo_E, twoelecint_mo)
        E_mp2, E_mp2_tot = self.mp2_calc.MP2_energy(occ, nao, t2, twoelecint_mo, Erhf)
        self.logger.info(f"occ: {occ}, virt: {virt}, o_act: {o_act}, v_act: {v_act}")
        self.logger.info(f"t1 shape: {t1.shape}, t2 shape: {t2.shape}, So shape: {So.shape}, Sv shape: {Sv.shape}")
        self.logger.info(f"Monomer {ifrag}: MP2 correlation energy: {E_mp2}, Total MP2 energy: {E_mp2_tot}")

        E_ccd, _ = self.cc_parallel.cc_calc(occ, virt, o_act, v_act, nmo, t1, t2, So, Sv, D1, D2, Do, Dv, twoelecint_mo, Fock_mo, self.config.method, self.config.niter, E_mp2_tot, self.config.conv)
        self.logger.info(f"Monomer {ifrag}: CC correlation energy: {E_ccd}")
        self.extractor.cleanup()
        return Erhf, E_mp2, E_ccd, lnum1, lnum2

    def compute_dimer(self, comb_idx, i, j, lnum1, lnum2):
        """Compute energies for a dimer pair.

        Parameters
        ----------
        comb_idx : int
            Index of the dimer combination.
        i : int
            Index of the first monomer in the dimer.
        j : int
            Index of the second monomer in the dimer.
        lnum1 : int
            Line number index for GAMESS output file.
        lnum2 : int
            Line number index for GAMESS 2e integral file.

        Returns
        -------
        tuple[float, float, float, int, int]
            A tuple containing RHF energy, MP2 correlation energy, CC correlation
            energy, updated lnum1, and updated lnum2.
        """
        nao = self.config.nao_dimer[comb_idx]
        self.logger.info(f"nao: {nao}")
        nmo = self.config.nmo_dimer[comb_idx]
        occ = self.config.occ_dimer[comb_idx]
        virt = self.config.virt_dimer[comb_idx]
        o_act = self.config.o_act_dimer[comb_idx]
        v_act = self.config.v_act_dimer[comb_idx]
        nfo = self.config.nfo_dimer[comb_idx]
        nfv = self.config.nfv_dimer[comb_idx]

        hamiltonian_file = f"hamiltonian{i}.txt"
        twoelecint_file = f"twoelecint{i}.txt"
        coeff_file = f"coeff{i}.txt"
        temp_file = "twoelecint_test.txt"
        twoelecintegral_file = "twoelecintegral.txt"

        ifrag, jfrag, Erhf = self.extractor.bare_hamiltonian(lnum1, 2, hamiltonian_file)
        self.logger.info(f"Dimer ({ifrag},{jfrag}): RHF energy: {Erhf}")
        Fock = self.extractor.get_1e_parameter(nao, hamiltonian_file)
        ifrag, jfrag, Erhf, lnum2 = self.extractor.twoelecint(lnum2, 2, twoelecint_file)
        self.extractor.twoelecint_process(twoelecint_file, temp_file)
        self.extractor.bash_run()
        twoeint = self.extractor.read_2eint(nao, twoelecintegral_file)
        ifrag, jfrag, Erhf, lnum1 = self.extractor.coeff(lnum1, 2, coeff_file)
        coeff = self.extractor.get_coeff(nmo, nao, coeff_file)
        hf_mo_E = self.extractor.get_orb_energy(nao, nmo, coeff_file)
        Fock_mo = np.diag(hf_mo_E)

        twoelecint_mo = self._transform_2eint(coeff, twoeint)

        if nfo > 0:
            occ, nmo, twoelecint_mo, Fock_mo, hf_mo_E = self.mp2_calc.occ_frozen(occ, nmo, nfo, twoelecint_mo, Fock_mo, hf_mo_E)
        if nfv > 0:
            nao, nmo, virt, twoelecint_mo, Fock_mo, hf_mo_E = self.mp2_calc.virt_frozen(virt, nmo, nfo, nfv, nao, twoelecint_mo, Fock_mo, hf_mo_E)

        t2, D2 = self.mp2_calc.guess_t2(occ, virt, nmo, hf_mo_E, twoelecint_mo)
        t1, D1 = self.mp2_calc.guess_t1(occ, virt, nmo, hf_mo_E, Fock_mo)
        So, Do = self.mp2_calc.guess_so(occ, virt, nmo, o_act, hf_mo_E, twoelecint_mo)
        Sv, Dv = self.mp2_calc.guess_sv(occ, virt, nmo, v_act, hf_mo_E, twoelecint_mo)
        E_mp2, E_mp2_tot = self.mp2_calc.MP2_energy(occ, nao, t2, twoelecint_mo, Erhf)
        self.logger.info(f"t1 shape: {t1.shape}, t2 shape: {t2.shape}, So shape: {So.shape}, Sv shape: {Sv.shape}")
        self.logger.info(f"Dimer {ifrag, jfrag}: MP2 correlation energy: {E_mp2}, Total MP2 energy: {E_mp2_tot}")

        E_ccd, _ = self.cc_parallel.cc_calc(occ, virt, o_act, v_act, nmo, t1, t2, So, Sv, D1, D2, Do, Dv, twoelecint_mo, Fock_mo, self.config.method, self.config.niter, E_mp2_tot, self.config.conv)
        self.logger.info(f"Dimer {ifrag, jfrag}: CC correlation energy: {E_ccd}")
        self.extractor.cleanup()
        return Erhf, E_mp2, E_ccd, lnum1, lnum2