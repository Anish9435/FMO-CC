import numpy as np
import subprocess
from itertools import combinations
from .MP2 import MP2Calculator
from .main_parallel import CCParallel
from .utils import FMOCC_LOGGER

class FMOCalculator:
    def __init__(self, config, extractor, processor):
        self.logger = FMOCC_LOGGER
        self.config = config
        self.extractor = extractor
        self.processor = processor
        self.mp2_calc = MP2Calculator()
        self.cc_parallel = CCParallel()

    def _transform_2eint(self, coeff, twoeint):
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
        nao = self.config.nao_mono[i]
        nmo = self.config.nmo_mono[i]
        occ = self.config.occ_mono[i]
        virt = self.config.virt_mono[i]
        o_act = self.config.o_act_mono[i]
        v_act = self.config.v_act_mono[i]
        nfo = self.config.nfo_mono[i]
        nfv = self.config.nfv_mono[i]

        hamiltonian_file = f"hamiltonian{i}.txt"
        twoelecint_file = f"twoelecint{i}.txt"
        coeff_file = f"coeff{i}.txt"
        temp_file = "twoelecint_test.txt"
        twoelecintegral_file = "twoelecintegral.txt"

        ifrag, _, Erhf = self.extractor.bare_hamiltonian(lnum1, 1, hamiltonian_file)
        self.logger.info(f"Monomer {ifrag}: RHF energy: {Erhf}")
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
        So, Do = self.mp2_calc.guess_so(occ, virt, o_act, hf_mo_E, twoelecint_mo)
        Sv, Dv = self.mp2_calc.guess_sv(occ, virt, v_act, hf_mo_E, twoelecint_mo)
        E_mp2, E_mp2_tot = self.mp2_calc.MP2_energy(occ, nao, t2, twoelecint_mo, Erhf)
        self.logger.info(f"Monomer {ifrag}: MP2 correlation energy: {E_mp2}, Total MP2 energy: {E_mp2_tot}")

        E_ccd, _ = self.cc_parallel.cc_calc(occ, virt, o_act, v_act, nmo, t1, t2, So, Sv, D1, D2, Do, Dv, twoelecint_mo, Fock_mo, self.config.method, self.config.niter, E_mp2_tot, self.config.conv)
        self.logger.info(f"Monomer {ifrag}: CC correlation energy: {E_ccd}")
        self.extractor.cleanup()
        return Erhf, E_mp2, E_ccd, lnum1, lnum2

    def compute_dimer(self, comb_idx, i, j, lnum1, lnum2):
        nao = self.config.nao_dimer[i]
        nmo = self.config.nmo_dimer[i]
        occ = self.config.occ_dimer[i]
        virt = self.config.virt_dimer[i]
        o_act = self.config.o_act_dimer[i]
        v_act = self.config.v_act_dimer[i]
        nfo = self.config.nfo_dimer[i]
        nfv = self.config.nfv_dimer[i]

        hamiltonian_file = f"hamiltonian{comb_idx}.txt"
        twoelecint_file = f"twoelecint{comb_idx}.txt"
        coeff_file = f"coeff{comb_idx}.txt"
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
        So, Do = self.mp2_calc.guess_so(occ, virt, o_act, hf_mo_E, twoelecint_mo)
        Sv, Dv = self.mp2_calc.guess_sv(occ, virt, v_act, hf_mo_E, twoelecint_mo)
        E_mp2, E_mp2_tot = self.mp2_calc.MP2_energy(occ, nao, t2, twoelecint_mo, Erhf)
        self.logger.info(f"Dimer {ifrag, jfrag}: MP2 correlation energy: {E_mp2}, Total MP2 energy: {E_mp2_tot}")

        E_ccd, _ = self.cc_parallel.cc_calc(occ, virt, o_act, v_act, nmo, t1, t2, So, Sv, D1, D2, Do, Dv, twoelecint_mo, Fock_mo, self.config.method, self.config.niter, E_mp2_tot, self.config.conv)
        self.logger.info(f"Dimer {comb_idx}: CC correlation energy: {E_ccd}")
        self.extractor.cleanup()
        return Erhf, E_mp2, E_ccd, lnum1, lnum2