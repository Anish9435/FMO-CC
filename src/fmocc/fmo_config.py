"""
Configuration management for FMO-CC calculations for fragments and fragment pairs

This module defines the FMOConfig class, responsible for managing and validating
configuration parameters for FMO calculations. It handles JSON-based input, updates
fragment and fragment-pair parameters based on GAMESS output, and supports automatic
active orbital selection and electron count for each fragment.

Key Responsibilities
--------------------
    - Load, parse, and validate JSON configuration files for FMO calculations.
    - Update fragment-specific parameters, including orbital and electron assignments.
    - Support automatic selection of active orbitals and electron counts per fragment.
    - Facilitate configuration management for both fragments and fragment pairs.

Dependencies
-------------
    - Python standard libraries: json, itertools (combinations)
    - External library: typing (List)
    - Local module: utils (FMOCC_LOGGER)
"""
import json
from typing import List
from itertools import combinations
from .utils import FMOCC_LOGGER

class FMOConfig:
    """Configuration manager for Fragment Molecular Orbital (FMO) calculations.

    Loads and validates configuration from a JSON file and updates fragment-related
    parameters based on GAMESS output.

    Parameters
    ----------
    input_file : str
        Path to the JSON configuration file.

    Attributes
    ----------
    logger : logging.Logger
        Logger instance for FMO calculations.
    data : Dict[str, Any]
        Raw configuration data loaded from the JSON file.
    elements : List[str]
        List of chemical elements in the system.
    method : str
        Calculation method (e.g., 'CCSD', 'ICCSD', 'ICCSD-PT').
    conv : float
        Convergence threshold for calculations.
    basis_set : str
        Basis set used for the calculation.
    niter : int
        Maximum number of iterations for convergence.
    filename : str
        Base filename for input/output files.
    icharge : List[int]
        List of charges for fragments.
    o_act : int
        Number of active occupied orbitals.
    v_act : int
        Number of active virtual orbitals.
    nfo : int
        Number of frozen occupied orbitals.
    nfv : int
        Number of frozen virtual orbitals.
    frag_atom : int
        Number of atoms per fragment unit.
    fragment_index : List[List[int]]
        Indices defining fragment boundaries.
    coeff_file : str
        Filename for coefficient output.
    hamiltonian_file : str
        Filename for Hamiltonian output.
    twoelecint_file : str
        Filename for two-electron integral output.
    nfrag : int
        Number of fragments.
    nao_mono : List[int]
        Number of atomic orbitals for each monomer.
    nao_dimer : List[int]
        Number of atomic orbitals for each dimer.
    nmo_mono : List[int]
        Number of molecular orbitals for each monomer.
    nmo_dimer : List[int]
        Number of molecular orbitals for each dimer.
    frag_elec : List[int]
        Number of electrons in each fragment.
    occ_mono : List[int]
        Number of occupied orbitals for each monomer.
    virt_mono : List[int]
        Number of virtual orbitals for each monomer.
    occ_dimer : List[int]
        Number of occupied orbitals for each dimer.
    virt_dimer : List[int]
        Number of virtual orbitals for each dimer.
    o_act_mono : List[int]
        Number of active occupied orbitals for each monomer.
    v_act_mono : List[int]
        Number of active virtual orbitals for each monomer.
    o_act_dimer : List[int]
        Number of active occupied orbitals for each dimer.
    v_act_dimer : List[int]
        Number of active virtual orbitals for each dimer.
    nfo_mono : List[int]
        Number of frozen occupied orbitals for each monomer.
    nfv_mono : List[int]
        Number of frozen virtual orbitals for each monomer.
    nfo_dimer : List[int]
        Number of frozen occupied orbitals for each dimer.
    nfv_dimer : List[int]
        Number of frozen virtual orbitals for each dimer.

    Raises
    ------
    ValueError
        If the input JSON file is invalid, missing required keys, or contains
        invalid values.
    """
    def __init__(self, input_file: str):
        """
        Initialize FMOConfig from an input JSON file.

        Parameters
        ----------
        input_file : str
            Path to the JSON configuration file.
        """
        self.logger = FMOCC_LOGGER
        try:
            with open(input_file) as f:
                data = json.load(f)
            self.logger.info(f"Successfully loaded configuration from {input_file}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.error(f"Error reading {input_file}: {e}")
            raise ValueError(f"Error reading {input_file}: {e}")
        
        if "common" in data:
            common = data.get("common", {})
            ctype = common.get("complex_type", "non-covalent")
            if ctype == "covalent":
                branch = data.get("covalent") or data.get("covalent".replace("-", "_"), {})
            else:
                branch = data.get("non-covalent") or data.get("non-covalent", {})
            merged = {**common, **branch}
            for key, val in data.items():
                if key not in ["common", "covalent", "non-covalent"]:
                    merged[key] = val
            data = merged
            self.data = data
        
        required_keys = ["elements", "method", "conv", "basis_set", "niter", "filename", "icharge", "fmo_type", "complex_type"]
        for key in required_keys:
            if key not in data:
                self.logger.error(f"Missing required key in JSON: {key}")
                raise ValueError(f"Missing required key in JSON: {key}")
        self.elements = data["elements"]
        if not all(isinstance(e, str) for e in self.elements):
            self.logger.error("All elements must be strings")
            raise ValueError("All elements must be strings")
        
        self.method = data["method"]
        if self.method not in ["MP2", "CCSD", "ICCSD", "ICCSD-PT"]:
            self.logger.error(f"Invalid method: {self.method}. Must be 'MP2', 'CCSD', 'ICCSD' or 'ICCSD-PT'")
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
    
        self.fmo_type = data["fmo_type"]
        if self.fmo_type not in ["FMO1", "FMO2"]:
            self.logger.error(f"Invalid fmo_type: {self.fmo_type}. Must be 'FMO1' or 'FMO2'")
            raise ValueError(f"Invalid fmo_type: {self.fmo_type}")
        
        self.complex_type = data["complex_type"]
        if self.complex_type not in ["covalent", "non-covalent"]:
            self.logger.error(f"Invalid complex_type: {self.complex_type}. Must be 'covalent' or 'non-covalent'")
            raise ValueError(f"Invalid complex_type: {self.complex_type}")
        
        if self.complex_type == "non-covalent" and not data.get("atom_pattern"):
            self.logger.warning("Non-covalent system specified but no frag_atom_patterns or atom_pattern provided")

        self.nproc = int(data.get("nproc", 0))
        self.o_act = data.get("occ_act", 1)
        self.v_act = data.get("virt_act", 1)
        self.auto_active = data.get("auto_active", True)
        self.active_threshold = float(data.get("active_threshold", 0.5))
        self.nfo = data.get("nfo", 0)
        self.nfv = data.get("nfv", 0)
        self.nfv_mono = data.get("nfv_mono")
        self.nfv_dimer = data.get("nfv_dimer")
        self.frag_atom = data.get("frag_atom", 3)
        self.fragment_index = data.get("fragment_index", [])
        self.coeff_file = data.get("coeff_file", "coeff.txt")
        self.hamiltonian_file = data.get("hamiltonian_file", "hamiltonian.txt")
        self.twoelecint_file = data.get("twoelecint_file", "twoelecint.txt")

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
        if self.nfv_mono is None:
            self.nfv_mono: List[int] = []
        self.nfo_dimer: List[int] = []
        if self.nfv_dimer is None:
            self.nfv_dimer: List[int] = []

    def update_from_gamess(self, nfrag: int, nao_mono: List[int], occ_mono: List[int]) -> None:
        """Update configuration parameters based on GAMESS output.

        Parameters
        ----------
        nfrag : int
            Number of fragments.
        nao_mono : List[int]
            Number of atomic orbitals for each monomer.

        Raises
        ------
        ValueError
            If nfrag or nao_mono are invalid or inconsistent with fragment_index.
        """
        if nfrag <= 0:
            self.logger.error(f"Invalid nfrag: {nfrag}")
            raise ValueError(f"Invalid nfrag: {nfrag}")
        if not nao_mono or any(n <= 0 for n in nao_mono):
            self.logger.error(f"Invalid nao_mono: {nao_mono}")
            raise ValueError(f"Invalid nao_mono: {nao_mono}")
        if self.complex_type == "covalent" and (not occ_mono or any (n < 0 for n in occ_mono) or len(occ_mono) != nfrag):
            self.logger.error(f"Invalid occ_mono for covalent system: {occ_mono}")
            raise ValueError(f"Invalid occ_mono for covalent system: {occ_mono}") 
        
        self.nfrag = nfrag
        self.nao_mono = nao_mono
        ndimer = len(list(combinations(range(self.nfrag), 2)))
        self.nao_dimer = [0] * ndimer if (self.complex_type == "covalent" and self.fmo_type == "FMO2") else [sum(combo) for combo in combinations(self.nao_mono, 2)]
        if hasattr(self, "nmo_mono") and self.nmo_mono:
            self.nmo_mono = self.nmo_mono
        else:
            self.nmo_mono = self.nao_mono[:]
        self.nmo_dimer = [0] * len(self.nao_dimer) if (self.complex_type == "covalent" and self.fmo_type == "FMO2") else self.nao_dimer[:]
        self.frag_elec = self._compute_frag_elec(occ_mono)
        self.occ_mono = occ_mono if self.complex_type == "covalent" else [int(e // 2) for e in self.frag_elec]
        self.virt_mono = [nmo - occ for nmo, occ in zip(self.nmo_mono, self.occ_mono)]
        if self.complex_type == "covalent" and self.fmo_type == "FMO2":
            self.occ_dimer = [0] * ndimer
            self.virt_dimer = [0] * ndimer
        else:
            self.occ_dimer = [sum(combo) for combo in combinations(self.occ_mono, 2)]
            self.virt_dimer = [nmo - occ for nmo, occ in zip(self.nmo_dimer, self.occ_dimer)]
        nmer: List[int] = []
        if self.fragment_index and self.complex_type == "non-covalent":
            if len(self.fragment_index) != self.nfrag:
                self.logger.error(f"Mismatch between nfrag ({self.nfrag}) and len(fragment_index) ({len(self.fragment_index)})")
                raise ValueError(f"Mismatch between nfrag and fragment_index")
            natoms = [max(frag) - min(frag) + 1 for frag in self.fragment_index]
            nmer = [int(natoms_i / self.frag_atom) for natoms_i in natoms]
        else:
            nmer = [1] * self.nfrag
        if not self.auto_active:
            self.o_act_mono = [self.o_act * nmer_i for nmer_i in nmer]
            self.v_act_mono = [self.v_act * nmer_i for nmer_i in nmer]
            if self.complex_type == "covalent":
                self.o_act_dimer = [self.o_act_mono[i] + self.o_act_mono[j] for i, j in combinations(range(self.nfrag), 2)]
                self.v_act_dimer = [self.v_act_mono[i] + self.v_act_mono[j] for i, j in combinations(range(self.nfrag), 2)]
            else:        
                self.o_act_dimer = [sum(combo) for combo in combinations(self.o_act_mono, 2)]
                self.v_act_dimer = [sum(combo) for combo in combinations(self.v_act_mono, 2)]
        else:
            self.logger.info("Auto active orbitals selection is enabled; manual o_act and v_act settings will be ignored")
            self.o_act_mono = [0] * self.nfrag
            self.v_act_mono = [0] * self.nfrag
            self.o_act_dimer = [0] * ndimer
            self.v_act_dimer = [0] * ndimer
        
        self.nfo_mono = [self.nfo * nmer_i for nmer_i in nmer]
        if not self.nfv_mono:
            self.nfv_mono = [self.nfv * nmer_i for nmer_i in nmer]      
        self.nfo_dimer = [sum(combo) for combo in combinations(self.nfo_mono, 2)]
        if not self.nfv_dimer:
            self.nfv_dimer = [sum(combo) for combo in combinations(self.nfv_mono, 2)]

        self.nao_dimer.sort()
        self.nmo_dimer.sort()
        if self.complex_type == "non-covalent":
            self.occ_dimer.sort()
            self.virt_dimer.sort()
        if self.o_act_dimer:
            self.o_act_dimer.sort()
        if self.v_act_dimer:
            self.v_act_dimer.sort()
        self.nfo_dimer.sort()

        self.logger.info(f"[CALC INFO] system type: {self.complex_type}")
        self.logger.info(f"[CALC INFO] Method: {self.method}, Basis set: {self.basis_set}, Convergence: {self.conv}, Max Iter: {self.niter}")
        self.logger.info(f"[SYSTEM INFO] Updated config with nfrag={nfrag}, nao_mono={nao_mono}, nmo_mono={self.nmo_mono}")
        self.logger.info(f"[SYSTEM INFO] Number of occupied orbitals for dimer: {self.occ_dimer} and for monomer: {self.occ_mono}")
        self.logger.info(f"[SYSTEM INFO] Number of virtual orbitals for dimer: {self.virt_dimer} and for monomer: {self.virt_mono}")
        self.logger.info(f"[SYSTEM INFO] Active occupied orbitals for dimer: {self.o_act_dimer} and for monomer: {self.o_act_mono}")
        self.logger.info(f"[SYSTEM INFO] Active virtual orbitals for dimer: {self.v_act_dimer} and for monomer: {self.v_act_mono}")
        self.logger.info(f"[SYSTEM INFO] Number of frozen occupied orbitals for dimer: {self.nfo_dimer} and for monomer: {self.nfo_mono}")
        self.logger.info(f"[SYSTEM INFO] Number of frozen virtual orbitals for dimer: {self.nfv_dimer} and for monomer: {self.nfv_mono}")
        self.logger.info(f"[SYSTEM INFO] Total number of electrons in fragments: {self.frag_elec}")

    def auto_set_active_orbitals(self, idx: int, hf_mo_E: List[float], occ: int, threshold: float, is_dimer: bool = False) -> None:
        """
        Automatically determine active orbitals for a fragment based on orbital energies and threshold closeness to HOMO/LUMO.

        Parameters
        ----------
        frag_idx : int
            Fragment index
        hf_mo_E : List[float]
            Orbital energies for the fragment
        occ : int
            Number of occupied orbitals
        virt : int
            Number of virtual orbitals
        """
        if not self.auto_active:
            return

        thr = threshold if threshold is not None else self.active_threshold

        if hf_mo_E is None or len(hf_mo_E) == 0:
            self.logger.warning(f"{'Dimer' if is_dimer else 'Fragment'} {idx+1}: empty orbital energy list; skipping auto active selection")
            return

        total_mos = len(hf_mo_E)
        occ_count = min(max(int(occ), 0), total_mos)
        virt_count = max(total_mos - occ_count, 0)

        occ_energies = hf_mo_E[:occ_count] if occ_count > 0 else []
        virt_energies = hf_mo_E[occ_count:] if virt_count > 0 else []

        homo_e = occ_energies[-1] if len(occ_energies) > 0 else None
        lumo_e = virt_energies[0] if len(virt_energies) > 0 else None

        # Active occupied orbitals: within threshold below HOMO
        if homo_e is not None:
            o_act_list = [e for e in occ_energies if (homo_e - e) <= thr]
            o_act = len(o_act_list)
        else:
            o_act_list, o_act = [], 0

        # Active virtual orbitals: within threshold above LUMO
        if lumo_e is not None:
            v_act_list = [e for e in virt_energies if (e - lumo_e) <= thr]
            v_act = len(v_act_list)
        else:
            v_act_list, v_act = [], 0

        if idx < 0 or idx >= (len(self.o_act_dimer) if is_dimer else self.nfrag):
            self.logger.error(f"Fragment index {idx} out of range (nfrag={self.nfrag})")
            return

        if is_dimer:
            self.o_act_dimer[idx] = o_act
            self.v_act_dimer[idx] = v_act
            entity = "Dimer"
        else:
            self.o_act_mono[idx] = o_act
            self.v_act_mono[idx] = v_act
            entity = "Fragment"

        self.logger.info(
            f"[Auto-active] {entity} {idx+1}: "
            f"HOMO={homo_e if homo_e is not None else 'N/A'}, "
            f"LUMO={lumo_e if lumo_e is not None else 'N/A'}, "
            f"thr={thr:.6f}, "
            f"o_act={o_act} (energies {o_act_list}), "
            f"v_act={v_act} (energies {v_act_list})"
        )

    def _compute_frag_elec(self, occ_mono: List[int]) -> List[int]:
        """Compute the number of electrons in each fragment.

        Returns
        -------
        List[int]
            Number of electrons in each fragment.

        Raises
        ------
        ValueError
            If the atom_pattern is not specified in the JSON configuration.
        """
        elec_map = {"1": 1, "H": 1, "2": 2, "He": 2,
                    "6": 6, "C": 6, "7": 7, "N": 7,
                    "8": 8, "O": 8, "9": 9, "F": 9}

        if self.complex_type == "covalent":
            if not occ_mono:
                self.logger.error(f"No occupied orbital has been detected!!")
            result = [2*na for na in occ_mono]
            self.logger.info(f"[SYSTEM INFO] Covalent System: occ_mono: {occ_mono} and fragment electron: {result}")

        else:
            pattern = self.data.get("atom_pattern", [])
            if not pattern:
                raise ValueError("Please specify 'atom_pattern' in JSON, e.g. ['8','1','1'] for water")
            result = []
            for frag in self.fragment_index:
                start, end = frag
                length = end - start + 1
                atoms = [pattern[i % len(pattern)] for i in range(length)]
                frag_elec = sum(elec_map[a] for a in atoms)
                result.append(frag_elec)
        self.logger.info(f"[SYSTEM INFO] Computed fragment electrons: {result}")
        return result