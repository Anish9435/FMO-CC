import numpy as np
import subprocess
from itertools import combinations
from .MP2 import MP2Calculator
from .main_parallel import CCParallel
from .logger import get_logger

class FMOCalculator:
    def __init__(self, config, extractor):
        self.logger = get_logger(__name__)
        self.config = config
        self.extractor = extractor
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

    def compute_monomer(self, i):
        nao = self.config.nao_mono[i]
        nmo = self.config.nmo_mono[i]
        occ = self.config.occ_mono[i]
        virt = self.config.virt_mono[i]
        o_act = self.config.o_act_mono[i]
        v_act = self.config.v_act_mono[i]
        nfo = self.config.nfo_mono[i]
        nfv = self.config.nfv_mono[i]

        ifrag, _, Erhf = self.extractor.bare_hamiltonian(1, f'hamiltonian{i}.txt')
        Fock = self.extractor.get_1e_parameter(nao, f'hamiltonian{i}.txt')
        _, _, _, _ = self.extractor.twoelecint(1, f'twoelecint{i}.txt')
        self.extractor.twoelecint_process()
        self.extractor.bash_run()
        twoeint = self.extractor.read_2eint(nao)
        _, _, _, _ = self.extractor.coeff(1, f'coeff{i}.txt')
        coeff = self.extractor.get_coeff(nmo, nao)
        hf_mo_E = self.extractor.get_orb_energy(nao, nmo, f'coeff{i}.txt')
        Fock_mo = np.diag(hf_mo_E)

        oneelecint_mo = np.einsum('ab,ac,cd->bd', coeff, Fock, coeff)
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

        E_ccd, _ = self.cc_parallel.cc_calc(occ, virt, o_act, v_act, nao, t1, t2, So, Sv, D1, D2, Do, Dv, twoelecint_mo, Fock_mo, self.config.method, self.config.niter, E_mp2_tot, self.config.conv)
        self.logger.info(f"Monomer {ifrag}: CC correlation energy: {E_ccd}")
        self.extractor.cleanup()
        return Erhf, E_mp2, E_ccd

    def compute_dimer(self, comb_idx, i, j):
        nao = self.config.nao_dimer[comb_idx]
        nmo = self.config.nmo_dimer[comb_idx]
        occ = self.config.occ_dimer[comb_idx]
        virt = self.config.virt_dimer[comb_idx]
        o_act = self.config.o_act_dimer[comb_idx]
        v_act = self.config.v_act_dimer[comb_idx]
        nfo = self.config.nfo_dimer[comb_idx]
        nfv = self.config.nfv_dimer[comb_idx]

        ifrag, jfrag, Erhf = self.extractor.bare_hamiltonian(2, f'hamiltonian{comb_idx}.txt')
        Fock = self.extractor.get_1e_parameter(nao, f'hamiltonian{comb_idx}.txt')
        _, _, _, _ = self.extractor.twoelecint(2, f'twoelecint{comb_idx}.txt')
        self.extractor.twoelecint_process()
        self.extractor.bash_run()
        twoeint = self.extractor.read_2eint(nao)
        _, _, _, _ = self.extractor.coeff(2, f'coeff{comb_idx}.txt')
        coeff = self.extractor.get_coeff(nmo, nao)
        hf_mo_E = self.extractor.get_orb_energy(nao, nmo, f'coeff{comb_idx}.txt')
        Fock_mo = np.diag(hf_mo_E)

        oneelecint_mo = np.einsum('ab,ac,cd->bd', coeff, Fock, coeff)
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
        self.logger.info(f"Dimer correlation Energy for fragment ({ifrag},{jfrag}): {E_mp2}")
        self.logger.info(f"Total MP2 Energy for fragment ({ifrag},{jfrag}): {E_mp2_tot}")

        E_ccd, _ = self.cc_parallel.cc_calc(occ, virt, o_act, v_act, nao, t1, t2, So, Sv, D1, D2, Do, Dv, twoelecint_mo, Fock_mo, self.config.method, self.config.niter, E_mp2_tot, self.config.conv)
        self.logger.info(f"Dimer {comb_idx}: CC correlation energy: {E_ccd}")
        self.extractor.cleanup()
        return Erhf, E_mp2, E_ccd