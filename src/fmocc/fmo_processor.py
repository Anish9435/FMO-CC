from math import comb
import os
import time
import tempfile
from .fmo_config import FMOConfig
from .fmo_extractor import FMOExtractor
from .fmo_calculator import FMOCalculator
from itertools import combinations
from .utils import FMOCC_LOGGER

class FMOProcessor:
    """Fragment Molecular Orbital (FMO) processor for Coupled cluster (CC) calculations.

    This class orchestrates the FMO calculation process, handling initialization,
    fragment extraction, and energy computations for monomers and dimers.

    Parameters
    ----------
    input_file : str
        Path to the input configuration file.
    base_dir : str, optional
        Base directory for input/output files. If None, defaults to the directory
        of the input file.

    Attributes
    ----------
    logger : logging.Logger
        Logger instance for FMO calculations.
    config : FMOConfig
        Configuration object for FMO calculations.
    base_dir : str
        Directory for file operations.
    extractor : FMOExtractor
        Object for extracting data from GAMESS output files.
    lnum1 : int
        Number of lines in the GAMESS output file.
    lnum2 : int
        Number of lines in the GAMESS 2e integral file.
    calculator : FMOCalculator
        Object for performing FMO energy calculations.
    """
    def __init__(self, input_file, base_dir=None):
        self.logger = FMOCC_LOGGER
        self.config = FMOConfig(input_file)
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(input_file))
        self.base_dir = base_dir
        gamess_out = os.path.join(self.base_dir, f"{self.config.filename}.dat")
        gamess_2eint = os.path.join(self.base_dir, f"{self.config.filename}_2eint.dat")
        coeff_file = getattr(self.config, 'coeff_file', 'coeff.txt')
        hamiltonian_file = getattr(self.config, 'hamiltonian_file', 'hamiltonian.txt')
        twoelecint_file = getattr(self.config, 'twoelecint_file', 'twoelecint.txt')
        temp_file = tempfile.NamedTemporaryFile(suffix='.txt', delete=False).name
        twoelecintegral_file = tempfile.NamedTemporaryFile(suffix='.txt', delete=False).name
        self.extractor = FMOExtractor(gamess_out, gamess_2eint, coeff_file, hamiltonian_file, 
                                      twoelecint_file, temp_file, twoelecintegral_file)
        with open(gamess_out, 'r') as f:
            self.lnum1 = len(f.readlines())
        with open(gamess_2eint, 'r') as f:
            self.lnum2 = len(f.readlines())
        nfrag = self.extractor.get_nfrags()
        nao_mono, _, occ_mono = self.extractor.get_frag_naos_atoms(self.lnum1, self.config.complex_type)
        if not occ_mono and self.config.complex_type == "non-covalent":
            occ_mono = [0]*nfrag
        if self.config.complex_type == "covalent":
            nmo_mono = self.extractor.get_frag_nmos(self.lnum1, nfrag)
            self.config.nmo_mono = nmo_mono
            self.config.update_from_gamess(nfrag, nao_mono, occ_mono)
        else:
            self.config.update_from_gamess(nfrag, nao_mono, occ_mono)
        self.calculator = FMOCalculator(self.config, self.extractor, self)
        self.logger.info(f"Initialized FMOProcessor with {nfrag} fragments and nao_mono: {nao_mono}")

    def run(self):
        """Execute the FMO calculation for monomers and dimers.

        Computes RHF, MP2, and CC correlation energies for monomers and dimers,
        and aggregates results to obtain total energies.

        Returns
        -------
        tuple[float, float]
            A tuple containing the total CC correlation energy and the total CC energy.

        Notes
        -----
        - Iterates over all dimer pairs and monomers to compute energies.
        - Logs the progress and results using the configured logger.
        - Measures and logs the total execution time.
        """
        start = time.time()
        mono_rhf, mono_mp2_corr = [], []
        dimer_rhf, dimer_mp2_corr = [], []
        mono_cc_corr = {}
        dimer_pairs = {}
        lnum1, lnum2 = self.lnum1, self.lnum2
        
        if self.config.fmo_type == "FMO2":
            seq = [i for i in range(self.config.nfrag)]
            comb = list(combinations(seq, 2))
            for idx, (i, j) in enumerate(comb):
                Erhf, E_mp2, E_ccd, lnum1, lnum2, fi, fj = self.calculator.compute_dimer(idx, i, j, lnum1, lnum2)
                dimer_rhf.append(Erhf)
                dimer_mp2_corr.append(E_mp2)
                dimer_pairs[(fi, fj)] = E_ccd
                self.logger.info("Completed dimer calculation")

        for i in range(self.config.nfrag):
            Erhf, E_mp2, E_ccd, lnum1, lnum2, frag_id = self.calculator.compute_monomer(i, lnum1, lnum2)
            mono_rhf.append(Erhf)
            mono_mp2_corr.append(E_mp2)
            mono_cc_corr[frag_id] = E_ccd
            self.logger.info("Completed monomer calculation")

        FMO_RHF = self.extractor.get_tot_rhf(self.config.fmo_type)
        if self.config.fmo_type == "FMO2":
            fmo_mp2_corr = sum(dimer_mp2_corr) - (self.config.nfrag - 2) * sum(mono_mp2_corr)
            fmo_cc_corr = sum(dimer_pairs.values()) - (self.config.nfrag - 2) * sum(mono_cc_corr.values())
        else:
            fmo_mp2_corr = sum(mono_mp2_corr)
            fmo_cc_corr = sum(mono_cc_corr.values())
        
        Tot_MP2 = FMO_RHF + fmo_mp2_corr
        Tot_CC = FMO_RHF + fmo_cc_corr

        self.logger.info(f"Total RHF energy: {FMO_RHF}")
        self.logger.info(f"MP2 correlation energy: {fmo_mp2_corr}, Total MP2 energy: {Tot_MP2}")
        self.logger.info(f"{self.config.method} correlation energy: {fmo_cc_corr}, Total {self.config.method} energy: {Tot_CC}")

        self.logger.info(f"Monomer {self.config.method} correlation energies (Ha):")
        for frag in sorted(mono_cc_corr):
            self.logger.info(f"Fragment {frag}: {mono_cc_corr[frag]}")
        
        if self.config.fmo_type == "FMO2":
            conv = 627.5095
            self.logger.info(f"Correlation level IFIEs (Ha / Kcal/mol):")
            for (fi, fj), Eij_cc in dimer_pairs.items():
                Ei_cc = mono_cc_corr[fi]
                Ej_cc = mono_cc_corr[fj]
                IFIE_cc = Eij_cc - Ei_cc - Ej_cc
                self.logger.info(f"Fragments ({fi}-{fj}): {IFIE_cc} / {IFIE_cc * conv}")

        end = time.time()
        self.logger.info(f"Parallel overall time: {end - start} seconds")

        return fmo_cc_corr, Tot_CC