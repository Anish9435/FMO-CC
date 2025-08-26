# 🔬 FMO-CC: Fragment Molecular Orbital with Coupled Cluster

**FMO-CC** is a Python-based implementation of the Fragment Molecular Orbital (FMO) in conjunction with Coupled Cluster (CC) and second-order 
Møller–Plesset perturbation theory (MP2) for high-accuracy quantum chemical calculations. It interfaces with **GAMESS** output files to compute 
monomer and dimer energies, providing RHF, MP2, and CC correlation energies for molecular systems. The codebase is **modular**, **parallelizable**, 
and designed with a robust error handling and configurability.

---

## Key Features

### **Quantum Chemistry Methods**
 - FMO-based RHF, MP2, and CC (CCSD, ICCSD, ICCSD-PT) calculations
 - Support for frozen occupied/virtual orbital simulations

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
./setup.sh
```
The `setup.sh` script will configure dependencies, validate the environment and prepare the pipeline for use

## Quick Start

Run the codebase using the input file:

```bash
python3 path/to/FMO_CC/Scripts/run_fmo_cc.py
```
or one can simply run:

```bash
run_fmo_cc
```

## Project Structure

```text
FMO-CC/
├── src/                             # Source root (used for editable install)
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