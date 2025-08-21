# FMO-CC: Fragment Molecular Orbital with Coupled Cluster

## Overview

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

## Dependencies

### Runtime

 - Python ≥ 3.8  
 - NumPy = 1.26.4  
 - Standard-library modules: `multiprocessing`, `json`, `logging`, `copy`, `gc`, `itertools`, `subprocess`, `glob`, `os`

### External Software

 - **GAMESS** (to generate `.dat` and `_2eint.dat` files)
 - **Bash** (to run `run_gamess.sh`, `twoeint_process.sh`)

---

## Installation

```bash
git clone <repo-url>
cd FMO-CC

# Install Python dependency
pip install numpy
```

## Project Structure

```text
FMO-CC/
├── input.json
├── README.md
├── run_fmo_cc.py
├── Scripts/
│   ├── run_gamess.sh
│   └── twoeint_process.sh
└── src/
    └── fmocc/
        ├── __init__.py
        ├── fmo_calculator.py
        ├── fmo_config.py
        ├── fmo_extractor.py
        ├── fmo_processor.py
        ├── main_parallel.py
        ├── MP2.py
        ├── diagrams.py
        └── utils.py
```