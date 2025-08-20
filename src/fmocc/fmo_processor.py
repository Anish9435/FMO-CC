import time
import tempfile
from .fmo_config import FMOConfig
from .fmo_extractor import FMOExtractor
from .fmo_calculator import FMOCalculator
from itertools import combinations
from .utils import get_logger

class FMOProcessor:
    def __init__(self, input_file):
        self.logger = get_logger(__name__)
        self.config = FMOConfig(input_file)
        gamess_out = f"{self.config.filename}.dat"
        gamess_2eint = f"{self.config.filename}_2eint.dat"
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
        nao_mono, _ = self.extractor.get_frag_naos_atoms(self.lnum1)
        self.config.update_from_gamess(nfrag, nao_mono)
        self.calculator = FMOCalculator(self.config, self.extractor, self)
        self.logger.info(f"Initialized FMOProcessor with {nfrag} fragments and nao_mono: {nao_mono}")

    def run(self):
        start = time.time()
        mono_rhf, mono_mp2_corr, mono_cc_corr = [], [], []
        dimer_rhf, dimer_mp2_corr, dimer_cc_corr = [], [], []
        lnum1, lnum2 = self.lnum1, self.lnum2
        
        seq = [i for i in range(self.config.nfrag)]
        comb = list(combinations(seq, 2))
        for idx, (i, j) in enumerate(comb):
            Erhf, E_mp2, E_ccd, lnum1, lnum2 = self.calculator.compute_dimer(idx, i, j, lnum1, lnum2)
            dimer_rhf.append(Erhf)
            dimer_mp2_corr.append(E_mp2)
            dimer_cc_corr.append(E_ccd)
            self.logger.info("Completed dimer calculation")

        for i in range(self.config.nfrag):
            Erhf, E_mp2, E_ccd, lnum1, lnum2 = self.calculator.compute_monomer(i, lnum1, lnum2)
            mono_rhf.append(Erhf)
            mono_mp2_corr.append(E_mp2)
            mono_cc_corr.append(E_ccd)
            self.logger.info("Completed monomer calculation")

        FMO_RHF = self.extractor.get_tot_rhf()
        fmo_mp2_corr = sum(dimer_mp2_corr) - (self.config.nfrag - 2) * sum(mono_mp2_corr)
        Tot_MP2 = FMO_RHF + fmo_mp2_corr
        fmo_cc_corr = sum(dimer_cc_corr) - (self.config.nfrag - 2) * sum(mono_cc_corr)
        Tot_CC = FMO_RHF + fmo_cc_corr

        self.logger.info(f"Total RHF energy: {FMO_RHF}")
        self.logger.info(f"MP2 correlation energy: {fmo_mp2_corr}, Total MP2 energy: {Tot_MP2}")
        self.logger.info(f"CC correlation energy: {fmo_cc_corr}, Total CC energy: {Tot_CC}")

        end = time.time()
        self.logger.info(f"Parallel overall time: {end - start} seconds")

        return fmo_cc_corr, Tot_CC