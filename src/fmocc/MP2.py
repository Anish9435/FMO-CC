"""
MP2 energy and amplitude utilities for FMO-CC calculations.

This module defines the MP2Calculator class, which handles initialization
of second-order Møller–Plesset perturbation theory (MP2) amplitudes and
the computation of MP2 correlation energies. It includes helper functions
for orbital freezing, amplitude guessing, and energy evaluation.

Key Responsibilities
--------------------
    - Perform orbital freezing (occupied and virtual).
    - Generate initial guesses for T₁, T₂, Sₒ, and Sᵥ amplitudes.
    - Compute MP2 correlation and total energies.

Dependencies
-------------
    - Python standard libraries: gc, copy
    - External library: numpy
    - Local module: utils (FMOCC_LOGGER)
"""
import gc
import copy as cp
import numpy as np
from .utils import FMOCC_LOGGER

class MP2Calculator:
    """Class for computing MP2 amplitudes and correlation energies within FMO-CC"""
    def __init__(self):
        self.logger = FMOCC_LOGGER

    def occ_frozen(self, occ, nmo, nfo, twoelecint_mo, Fock_mo, hf_mo_E):
        occ -= nfo
        nmo -= nfo
        twoelecint_mo = cp.deepcopy(twoelecint_mo[nfo:, nfo:, nfo:, nfo:])
        hf_mo_E = cp.deepcopy(hf_mo_E[nfo:])
        Fock_mo = cp.deepcopy(Fock_mo[nfo:, nfo:])
        gc.collect()
        self.logger.debug(f"Froze {nfo} occupied orbitals")
        return occ, nmo, twoelecint_mo, Fock_mo, hf_mo_E

    def virt_frozen(self, virt, nmo, nfo, nfv, nao, twoelecint_mo, Fock_mo, hf_mo_E):
        twoelecint_mo = cp.deepcopy(twoelecint_mo[:-nfv, :-nfv, :-nfv, :-nfv])
        Fock_mo = cp.deepcopy(Fock_mo[:-nfv, :-nfv])
        hf_mo_E = hf_mo_E[:-nfv]
        nao = nao - nfv - nfo
        nmo = nmo - nfv
        virt = virt - nfv
        gc.collect()
        self.logger.debug(f"Froze {nfv} virtual orbitals")
        return nao, nmo, virt, twoelecint_mo, Fock_mo, hf_mo_E

    def guess_t1(self, occ, virt, nmo, hf_mo_E, Fock_mo):
        t1 = np.zeros((occ, virt))
        D1 = np.zeros((occ, virt))
        for i in range(occ):
            for a in range(occ, nmo):
                D1[i, a - occ] = hf_mo_E[i] - hf_mo_E[a]
                t1[i, a - occ] = Fock_mo[i, a] / D1[i, a - occ]
        self.logger.debug("Computed initial t1 guess")
        return t1, D1

    def guess_t2(self, occ, virt, nmo, hf_mo_E, twoelecint_mo):
        D2 = np.zeros((occ, occ, virt, virt))
        t2 = np.zeros((occ, occ, virt, virt))
        for i in range(occ):
            for j in range(occ):
                for a in range(occ, nmo):
                    for b in range(occ, nmo):
                        D2[i, j, a - occ, b - occ] = hf_mo_E[i] + hf_mo_E[j] - hf_mo_E[a] - hf_mo_E[b]
                        t2[i, j, a - occ, b - occ] = twoelecint_mo[i, j, a, b] / D2[i, j, a - occ, b - occ]
        self.logger.debug("Computed initial t2 guess")
        return t2, D2

    def guess_so(self, occ, virt, nmo, o_act, hf_mo_E, twoelecint_mo):
        Do = np.zeros((occ, occ, virt, o_act))
        So = np.zeros((occ, occ, virt, o_act))
        for i in range(occ):
            for j in range(occ):
                for a in range(virt):
                    for k in range(occ - o_act, occ):
                        Do[i, j, a, k - occ + o_act] = hf_mo_E[i] + hf_mo_E[j] - hf_mo_E[a + occ] + hf_mo_E[k]
                        So[i, j, a, k - occ + o_act] = twoelecint_mo[i, j, a + occ, k] / Do[i, j, a, k - occ + o_act]
        self.logger.debug("Computed initial So guess")
        return So, Do

    def guess_sv(self, occ, virt, nmo, v_act, hf_mo_E, twoelecint_mo):
        Dv = np.zeros((occ, v_act, virt, virt))
        Sv = np.zeros((occ, v_act, virt, virt))
        for i in range(occ):
            for c in range(v_act):
                for a in range(virt):
                    for b in range(virt):
                        Dv[i, c, a, b] = hf_mo_E[i] - hf_mo_E[c + occ] - hf_mo_E[a + occ] - hf_mo_E[b + occ]
                        Sv[i, c, a, b] = twoelecint_mo[i, c + occ, a + occ, b + occ] / Dv[i, c, a, b]
        self.logger.debug("Computed initial Sv guess")
        return Sv, Dv

    def MP2_energy(self, occ, nao, t2, twoelecint_mo, E_hf):
        E_mp2 = 2 * np.einsum('ijab,ijab', t2, twoelecint_mo[:occ, :occ, occ:nao, occ:nao]) - np.einsum('ijab,ijba', t2, twoelecint_mo[:occ, :occ, occ:nao, occ:nao])
        E_mp2_tot = E_hf + E_mp2
        return E_mp2, E_mp2_tot