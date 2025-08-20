# FMO-CC: Fragment Molecular Orbital with Coupled Cluster

## Overview

**FMO-CC** is a Python-based implementation of the Fragment Molecular Orbital (FMO) method combined with Coupled Cluster (CC) and second-order 
Møller–Plesset perturbation theory (MP2) for high-accuracy quantum chemical calculations. It interfaces with **GAMESS** output files to compute 
monomer and dimer energies, providing RHF, MP2, and CC correlation energies for molecular systems. The codebase is modular, parallelizable, and 
designed with research-oriented error handling and configurability.

---

## Features

- FMO-based **RHF**, **MP2**, and **CC** (CCSD, ICCSD, ICCSD-PT) energy calculations  
- Support for **frozen occupied/virtual orbital** simulations  
- Multiprocessing for parallel CC iterations  
- Automatic extraction of integrals, MO coefficients, and orbital data from GAMESS outputs  
- Easy configurability via **JSON input file** (`input.json`)  
- Robust, rotating **logging** for reproducibility and debugging

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
├── input.json
├── README.md
├── run_fmo_cc.py
├── Scripts
│   ├── run_gamess.sh
│   └── twoeint_process.sh
└── src
    └── fmocc
        ├── diagrams.py
        ├── fmo_calculator.py
        ├── fmo_config.py
        ├── fmo_extractor.py
        ├── fmo_processor.py
        ├── __init__.py
        ├── main_parallel.py
        ├── MP2.py
        └── utils.py
```