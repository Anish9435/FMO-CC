# 🔬 FMO-CC: Fragment Molecular Orbital with Coupled Cluster

**FMO-CC** is a Python-based implementation of the Fragment Molecular Orbital (FMO) in conjunction with Coupled Cluster (CC) and second-order 
Møller–Plesset perturbation theory (MP2) for high-accuracy quantum chemical calculations. It interfaces with **GAMESS** generated output files 
to compute monomer and dimer energies, providing RHF, MP2, and CC correlation energies for molecular systems. The codebase is **modular**, 
**parallelizable**, and designed with a robust error handling and configurability.

---

## Key Features

### **Quantum Chemistry Methods**
 - FMO-based RHF, MP2, and CC (CCSD, ICCSD, ICCSD-PT) calculations
 - Support for frozen occupied/virtual orbital simulations
 - Supports both one-body FMO (FMO1) and two-body FMO (FMO2) calculations

### **Parallelization**
 - Multiprocessing for parallel CC iterations
 - Scalable design for larger molecular systems

### **Integration with GAMESS**
 - Automatic extraction of integrals, MO coefficients, and orbital data from GAMESS outputs
 - Compatibility with .dat and _2eint.dat files

### **Configurable Input**
 - Easy-to-use JSON input file (input.json) for parameters and workflow control

### **Robust Logging & Reproducibility**
 - Rotating log files for reproducibility, debugging, and traceability

---


## Installation

```bash
git clone https://github.com/Anish9435/FMO-CC.git
cd FMO-CC
chmod +x setup.sh
./setup.sh
```
The `setup.sh` script will configure dependencies, validate the environment and prepare the pipeline for use

Set up the virtual environment via:

```bash
source fmocc_env/bin/activate
```
And then follow the `Quick start` section to run the scripts

## Quick Start

Run the codebase using the launcher script:

```bash
# Default input file (input.json)
python3 path/to/FMO_CC/Scripts/run_fmo_cc.py

#custom input file
python3 path/to/FMO_CC/Scripts/run_fmo_cc.py -c my_config.json
```
or one can simply run as python module:

```bash
# Default input file (input.json)
python -m fmocc

#custom input file
python -m fmocc -c my_config.json
```
or, run via the console script (after the pip install):

```bash
# Default input file (input.json)
run_fmo_cc

#custom input file
run_fmo_cc -c my_config.json
```

## Configuration file input options

```text
|      Flag / Key         |                          Description                                     |
|-------------------------|--------------------------------------------------------------------------|
| `method`                | Correlated method: `ICCSD`, `ICCSD-PT`, `CCSD`                           |
| `conv`                  | Convergence threshold for CC iterations (e.g., `1e-7`)                   |
| `auto_active`           | automatically determine active orbs from HOMO/LUMO (`true` or `false`)   |
| `active_threshold`      | Energy window (in Hartree) around HOMO/LUMO for auto-active selection    |
| `occ_act`               | Number of active occupied orbitals                                       |
| `virt_act`              | Number of active virtual orbitals                                        |
| `nfo`                   | Number of frozen occupied orbitals                                       |
| `nfv`                   | Number of frozen virtual orbitals                                        |
| `basis_set`             | Basis set to use (e.g., `6-21g`, `cc-pVDZ`, etc.)                        |
| `niter`                 | Maximum number of CC iterations                                          |
| `frag_atom`             | Number of atoms per fragment (used in noncovalent fragmentation mode)    |
| `elements`              | List of atomic numbers present in the system                             |
| `atom_pattern`          | Atom sequence pattern defining fragments                                 |
| `filename`              | Base filename for GAMESS outputs (`.dat` and `_2eint.dat`)               |
| `fragment_index`        | Explicit fragment index ranges (for custom/nonstandard fragmentation)    |
| `icharge`               | List of charges for each fragment                                        |
| `integral_transform`    | Mode for integral transformation: `incore` or `disk`                     |
| `fmo_type`              | Fragment Molecular Orbital type: `FMO1`, `FMO2`, etc.                    |
| `complex_type`          | specification of the complex: `covalent`, `non-covalent`, etc.           |
```
**Note:** Use `-c your_config.json` to load all parameters from the custom made JSON file

**Note:** Currently, the codebase accepts GAMESS-generated output files as input. Work is in progress to automate the fragmentation process and GAMESS execution. Using the provided output files, the code extracts the relevant parameters for subsequent runs and computes accurate MP2 and CC energies.

**Note** In addition to the conventional CCSD methodology, this codebase implements two alternative variants of coupled-cluster theory: iCCSDn and iCCSDn-PT. Both methods exhibit computational scaling comparable to CCSD. Further details of the theoretical developments can be found [Here](https://pubs.aip.org/aip/jcp/article/156/24/244117/2841424/A-double-exponential-coupled-cluster-theory-in-the?searchresult=1)

## Project Structure

```text
FMO-CC/
├── src/                             # Source root
│   └── fmocc/                       # Main Python package
│       ├── __init__.py
│       ├── __main__.py              # Enables `python -m fmocc` execution
│       ├── fmo_calculator.py        # Core FMO-CC energy calculations
│       ├── fmo_config.py            # Configuration and parameter management
│       ├── fmo_extractor.py         # Data extraction from GAMESS outputs
│       ├── fmo_processor.py         # High-level FMO-CC workflow orchestration
│       ├── main_parallel.py         # Parallelization logic for CC computations
│       ├── MP2.py                   # MP2-specific methods and corrections
│       ├── diagrams.py              # Diagrammatic CC expansions
│       └── utils.py                 # Logging, helpers, and cache management
│
├── Scripts/                         # CLI scripts and execution helpers
│   ├── run_fmo_cc.py                # Main launcher for FMO-CC calculations
│   ├── run_gamess.sh                # Wrapper script for GAMESS runs
│   └── twoeint_process.sh           # Post-processing for two-electron integrals
│
├── pyproject.toml                   # Packaging config (PEP 621 / pip install)
├── input.json                       # Example input file for FMO-CC
├── setup.sh                         # Script to automate installation
└── README.md                        # Project documentation and usage guide
```