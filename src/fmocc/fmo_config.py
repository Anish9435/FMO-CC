from ast import pattern
import json
from itertools import combinations
from typing import List, Dict, Any
from .utils import FMOCC_LOGGER

class FMOConfig:
    def __init__(self, input_file: str):
        self.logger = FMOCC_LOGGER
        try:
            with open(input_file) as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.error(f"Error reading {input_file}: {e}")
            raise ValueError(f"Error reading {input_file}: {e}")
        self.data = data
        required_keys = ["elements", "method", "conv", "basis_set", "niter", "filename", "icharge"]
        for key in required_keys:
            if key not in data:
                self.logger.error(f"Missing required key in JSON: {key}")
                raise ValueError(f"Missing required key in JSON: {key}")

        self.elements = data["elements"]
        if not all(isinstance(e, str) for e in self.elements):
            self.logger.error("All elements must be strings")
            raise ValueError("All elements must be strings")
        
        self.method = data["method"]
        if self.method not in ["CCSD", "ICCSD", "ICCSD-PT"]:
            self.logger.error(f"Invalid method: {self.method}. Must be 'CCSD', 'ICCSD' or 'ICCSD-PT'")
            raise ValueError(f"Invalid method: {self.method}")
        
        try:
            self.conv = float(data["conv"])
            if self.conv <= 0:
                raise ValueError
        except (ValueError, TypeError):
            self.logger.error(f"Invalid convergence threshold: {data['conv']}")
            raise ValueError(f"Invalid convergence threshold: {data['conv']}")
        
        self.basis_set = data["basis_set"]
        self.niter = int(data["niter"])
        if self.niter <= 0:
            self.logger.error(f"Invalid niter: {self.niter}")
            raise ValueError(f"Invalid niter: {self.niter}")
        
        self.filename = data["filename"]
        self.icharge = data["icharge"]
        if not isinstance(self.icharge, list) or len(self.icharge) == 0:
            self.logger.error("icharge must be a non-empty list")
            raise ValueError("icharge must be a non-empty list")
        
        self.o_act = data.get("occ_act", 1)
        self.v_act = data.get("virt_act", 1)
        self.nfo = data.get("nfo", 0)
        self.nfv = data.get("nfv", 0)
        self.frag_atom = data.get("frag_atom", 3)
        self.fragment_index = data.get("fragment_index", [])
        self.coeff_file = data.get("coeff_file", "coeff.txt")
        self.hamiltonian_file = data.get("hamiltonian_file", "hamiltonian.txt")
        self.twoelecint_file = data.get("twoelecint_file", "twoelecint.txt")

        # CHANGE: Initialize placeholders; updated by FMOExtractor
        self.nfrag: int = 0
        self.nao_mono: List[int] = []
        self.nao_dimer: List[int] = []
        self.nmo_mono: List[int] = []
        self.nmo_dimer: List[int] = []
        self.frag_elec: List[int] = []
        self.occ_mono: List[int] = []
        self.virt_mono: List[int] = []
        self.occ_dimer: List[int] = []
        self.virt_dimer: List[int] = []
        self.o_act_mono: List[int] = []
        self.v_act_mono: List[int] = []
        self.o_act_dimer: List[int] = []
        self.v_act_dimer: List[int] = []
        self.nfo_mono: List[int] = []
        self.nfv_mono: List[int] = []
        self.nfo_dimer: List[int] = []
        self.nfv_dimer: List[int] = []

    def update_from_gamess(self, nfrag: int, nao_mono: List[int]) -> None:
        if nfrag <= 0:
            self.logger.error(f"Invalid nfrag: {nfrag}")
            raise ValueError(f"Invalid nfrag: {nfrag}")
        if not nao_mono or any(n <= 0 for n in nao_mono):
            self.logger.error(f"Invalid nao_mono: {nao_mono}")
            raise ValueError(f"Invalid nao_mono: {nao_mono}")
        
        self.nfrag = nfrag
        self.nao_mono = nao_mono
        self.nao_dimer = [sum(combo) for combo in combinations(self.nao_mono, 2)]
        self.nmo_mono = self.nao_mono[:]
        self.nmo_dimer = self.nao_dimer[:]
        self.frag_elec = self._compute_frag_elec()
        self.occ_mono = [int(e // 2) for e in self.frag_elec]
        self.virt_mono = [nmo - occ for nmo, occ in zip(self.nmo_mono, self.occ_mono)]
        self.occ_dimer = [sum(combo) for combo in combinations(self.occ_mono, 2)]
        self.virt_dimer = [nmo - occ for nmo, occ in zip(self.nmo_dimer, self.occ_dimer)]
        nmer: List[int] = []
        if self.fragment_index:
            if len(self.fragment_index) != self.nfrag:
                self.logger.error(f"Mismatch between nfrag ({self.nfrag}) and len(fragment_index) ({len(self.fragment_index)})")
                raise ValueError(f"Mismatch between nfrag and fragment_index")
            natoms = [max(frag) - min(frag) + 1 for frag in self.fragment_index]
            nmer = [int(natoms_i / self.frag_atom) for natoms_i in natoms]
        else:
            nmer = [1] * self.nfrag
        self.o_act_mono = [self.o_act * nmer_i for nmer_i in nmer]
        self.v_act_mono = [self.v_act * nmer_i for nmer_i in nmer]
        self.nfo_mono = [self.nfo * nmer_i for nmer_i in nmer]
        self.nfv_mono = [self.nfv * nmer_i for nmer_i in nmer]
        self.o_act_dimer = [sum(combo) for combo in combinations(self.o_act_mono, 2)]
        self.v_act_dimer = [sum(combo) for combo in combinations(self.v_act_mono, 2)]
        self.nfo_dimer = [sum(combo) for combo in combinations(self.nfo_mono, 2)]
        self.nfv_dimer = [sum(combo) for combo in combinations(self.nfv_mono, 2)]
        self.logger.info(f"Updated config with nfrag={nfrag}, nao_mono={nao_mono}")  # CHANGE: Logging update
        self.logger.info(f"Number of occupied orbitals for dimer: {self.occ_dimer} and for monomer: {self.occ_mono}")
        self.logger.info(f"Number of virtual orbitals for dimer: {self.virt_dimer} and for monomer: {self.virt_mono}")
        self.logger.info(f"Active occupied orbitals for dimer: {self.o_act_dimer} and for monomer: {self.o_act_mono}")
        self.logger.info(f"Active virtual orbitals for dimer: {self.v_act_dimer} and for monomer: {self.v_act_mono}")
        self.logger.info(f"Number of frozen occupied orbitals for dimer: {self.nfo_dimer} and for monomer: {self.nfo_mono}")
        self.logger.info(f"Number of frozen virtual orbitals for dimer: {self.nfv_dimer} and for monomer: {self.nfv_mono}")
        self.logger.info(f"Total number of electrons in fragments: {self.frag_elec}")

    def _compute_frag_elec(self) -> List[int]:
        elec_map = {"1": 1, "H": 1, "2": 2, "He": 2,
                    "6": 6, "C": 6, "7": 7, "N": 7,
                    "8": 8, "O": 8, "9": 9, "F": 9}

        # You must specify one repeating unit (monomer pattern) in JSON
        # Example: "atom_pattern": ["8","1","1"] for H2O
        pattern = self.data.get("atom_pattern", [])
        if not pattern:
            raise ValueError("Please specify 'atom_pattern' in JSON, e.g. ['8','1','1'] for water")

        result = []
        for frag in self.fragment_index:
            start, end = frag
            length = end - start + 1
            # build the atom list for this fragment by repeating pattern
            atoms = [pattern[i % len(pattern)] for i in range(length)]
            # sum electrons
            frag_elec = sum(elec_map[a] for a in atoms)
            result.append(frag_elec)

        self.logger.info(f"Computed fragment electrons: {result}")
        return result