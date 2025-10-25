"""
Orchestration of FMO-CC calculations.

This module implements the FMOProcessor class, responsible for managing and orchestrating
the full FMO-CC calculation workflow. It handles configuration management, data extraction
from GAMESS output, energy computations for monomers and dimers, and aggregation of results.
Also, a limitation on the size of input files based on the number of atoms is enforced to ensure
efficient processing.

Key Responsibilities
--------------------
    - Initialize and validate FMO calculation configurations via FMOConfig.
    - Extract fragment-specific data from GAMESS outputs using FMOExtractor.
    - Perform RHF, MP2, and CC energy computations for monomers and dimers through FMOCalculator.
    - Aggregate fragment and fragment-pair energies to obtain total electronic energies.
    - Manage workflow execution, temporary resources, and logging for reproducibility.

Dependencies
-------------
    - Python standard libraries: math (comb), os, time, tempfile, itertools (combinations)
    - Local modules: fmo_config (FMOConfig), fmo_extractor (FMOExtractor), fmo_calculator (FMOCalculator), utils (FMOCC_LOGGER)
"""
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
        nao_mono, natoms, occ_mono = self.extractor.get_frag_naos_atoms(self.lnum1, self.config.complex_type)
        total_atoms = sum(natoms)
        for file_path, file_desc in [(gamess_out, "GAMESS output"), (gamess_2eint, "GAMESS 2e integral")]:
            file_size = os.path.getsize(file_path) / (1024 * 1024 * 1024)  # Size in GB
            if total_atoms <= 40:
                max_size = 10  # 10GB for <= 40 atoms
            elif total_atoms <= 60:
                max_size = 20  # 20GB for <= 60 atoms
            else:
                max_size = 50  # 50GB for > 60 atoms
            
            basis = getattr(self.config, "basis_set", "").lower()
            if "6-31g" in basis:
                factor = 1.0   # baseline
            elif any(x in basis for x in ["6-311g", "def2-svp", "def2sv(p)", "cc-pvdz"]):
                factor = 1.5   # moderately larger basis
            elif any(x in basis for x in ["def2-tzvp", "cc-pvtz", "tzvp"]):
                factor = 2.0   # triple-zeta
            elif any(x in basis for x in ["def2-qzvp", "cc-pvqz", "qzvp"]):
                factor = 3.0   # quadruple-zeta or larger
            else:
                factor = 1.2   # unknown basis → mild increase

            max_size *= factor
            if file_size > max_size:
                self.logger.error(f"[FILE ERROR] {file_desc} file {file_path} size ({file_size:.2f} GB) exceeds limit ({max_size} GB) for {total_atoms} atoms")
                raise ValueError(f"{file_desc} file size ({file_size:.2f} GB) exceeds limit ({max_size} GB) for {total_atoms} atoms")
            self.logger.info(f"[FILE INFO] {file_desc} file {file_path} size ({file_size:.2f} GB) is within limit ({max_size} GB) for {total_atoms} atoms")

        if not occ_mono and self.config.complex_type == "non-covalent":
            occ_mono = [0]*nfrag
        if self.config.complex_type == "covalent":
            nmo_mono = self.extractor.get_frag_nmos(self.lnum1, nfrag)
            self.config.nmo_mono = nmo_mono
            self.config.update_from_gamess(nfrag, nao_mono, occ_mono)
        else:
            self.config.update_from_gamess(nfrag, nao_mono, occ_mono)
        self.calculator = FMOCalculator(self.config, self.extractor, self)
        self.logger.info(f"[INFO] Initialized FMOProcessor with {nfrag} fragments and nao_mono: {nao_mono}")

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
        mono_rhf, mono_mp2_corr = [], {}
        dimer_rhf, dimer_mp2_corr = [], []
        mono_cc_corr = {}
        dimer_pairs = {}
        lnum1, lnum2 = self.lnum1, self.lnum2
        
        if self.config.fmo_type == "FMO2":
            ndimers_actual = self.extractor.get_available_dimers()
            seq = [i for i in range(self.config.nfrag)]
            comb_expected = list(combinations(seq, 2))
            if ndimers_actual != len(comb_expected):
                self.logger.info(
                    f"[DIMER INFO] Dimer count adjusted: expected {len(comb_expected)}, "
                    f"found {ndimers_actual} — regenerating dimer list"
                )
                comb = [(0, 0)] * ndimers_actual
            else:
                comb = comb_expected
            for idx, (i, j) in enumerate(comb):
                Erhf, E_mp2, E_ccd, lnum1, lnum2, fi, fj = self.calculator.compute_dimer(idx, i, j, lnum1, lnum2)
                dimer_rhf.append(Erhf)
                dimer_mp2_corr.append(E_mp2)
                dimer_pairs[(fi, fj)] = E_ccd
                self.logger.info(f"[INFO] Completed dimer calculation")

        for i in range(self.config.nfrag):
            Erhf, E_mp2, E_ccd, lnum1, lnum2, frag_id = self.calculator.compute_monomer(i, lnum1, lnum2)
            mono_rhf.append(Erhf)
            mono_mp2_corr[frag_id] = E_mp2
            mono_cc_corr[frag_id] = E_ccd
            self.logger.info(f"[INFO] Completed monomer calculation")

        FMO_RHF = self.extractor.get_tot_rhf(self.config.fmo_type)
        if self.config.fmo_type == "FMO2":
            mono_mp2_sum = sum(mono_mp2_corr.values())
            mono_cc_sum = sum(mono_cc_corr.values())
            dimer_mp2_corr_dict = {
                pair: corr for pair, corr in zip(dimer_pairs.keys(), dimer_mp2_corr)
            }
            fmo_mp2_corr_pair = sum(Eij_mp2 - mono_mp2_corr[fi] - mono_mp2_corr[fj] 
                               for (fi, fj), Eij_mp2 in dimer_mp2_corr_dict.items())
            fmo_cc_corr_pair = sum(Eij_cc - mono_cc_corr[fi] - mono_cc_corr[fj] 
                               for (fi, fj), Eij_cc in dimer_pairs.items())
            fmo_mp2_corr = mono_mp2_sum + fmo_mp2_corr_pair
            fmo_cc_corr = mono_cc_sum + fmo_cc_corr_pair
        else:
            fmo_mp2_corr = sum(mono_mp2_corr.values())
            fmo_cc_corr = sum(mono_cc_corr.values())

        self.logger.info(f"[ENERGY INFO] Total RHF energy: {FMO_RHF}")
        if self.config.method == "MP2":
            corr_energy = fmo_mp2_corr
            Tot_MP2 = FMO_RHF + fmo_mp2_corr
            self.logger.info(f"[ENERGY INFO] MP2 correlation energy: {corr_energy}, Total MP2 energy: {Tot_MP2}")
        else:
            corr_energy = fmo_cc_corr
            Tot_CC = FMO_RHF + fmo_cc_corr
            self.logger.info(f"[ENERGY INFO] {self.config.method} correlation energy: {corr_energy}, Total {self.config.method} energy: {Tot_CC}")

        self.logger.info(f"[ENERGY INFO] Monomer {self.config.method} correlation energies (Ha):")
        for frag in sorted(mono_cc_corr):
            self.logger.info(f"[ENERGY INFO] Fragment {frag}: {mono_cc_corr[frag]}")
        
        if self.config.fmo_type == "FMO2":
            conv = 627.5095
            if self.config.method == "MP2":
                self.logger.info(f"[ENERGY INFO] MP2 level IFIEs (Ha | Kcal/mol):")
                for (fi, fj), Eij_mp2 in zip(dimer_pairs.keys(), dimer_mp2_corr):
                    Ei_mp2 = mono_mp2_corr[fi]
                    Ej_mp2 = mono_mp2_corr[fj]
                    IFIE_mp2 = Eij_mp2 - Ei_mp2 - Ej_mp2
                    self.logger.info(f"[ENERGY INFO] Fragments ({fi}-{fj}): {IFIE_mp2} | {IFIE_mp2 * conv}")
            else:
                self.logger.info(f"[ENERGY INFO] Correlation level IFIEs (Ha | Kcal/mol):")
                for (fi, fj), Eij_cc in dimer_pairs.items():
                    Ei_cc = mono_cc_corr[fi]
                    Ej_cc = mono_cc_corr[fj]
                    IFIE_cc = Eij_cc - Ei_cc - Ej_cc
                    self.logger.info(f"[ENERGY INFO] Fragments ({fi}-{fj}): {IFIE_cc} | {IFIE_cc * conv}")

        end = time.time()
        self.logger.info(f"[TIMING INFO] Parallel overall time: {end - start} seconds")

        return fmo_cc_corr, Tot_CC