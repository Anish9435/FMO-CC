# 🔬 FMO-CC: Fragment Molecular Orbital with Coupled Cluster

## 🌟 Summary  
**FMO-CC** is a Python package for **fragment-based quantum chemistry calculations**, combining the **Fragment Molecular Orbital (FMO)** method 
with **Coupled Cluster (CC)** and **MP2**.  It is designed to scale quantum chemical calculations to **larger molecular systems** by partitioning 
them into fragments while retaining the quantitative accuracy.  

With **seamless integration with GAMESS**, automated orbital selection, parallel execution, and a **configurable JSON workflow**, FMO-CC provides 
a **research-grade, modular, parallelizable, and reproducible framework** for electronic structure calculations.

---

## 🚀 Key Features

### **Quantum Chemistry Methods**
 - Fragment Molecular Orbital (FMO) framework supporting RHF, MP2, and CC methods (CCSD, ICCSD, ICCSD-PT)
 - Configurable frozen occupied and virtual orbital treatments
 - Flexible support for both FMO1 (one-body) and FMO2 (two-body) calculations

### **Selection of active orbitals**
 - Automated detection of **chemically relevant orbitals** based on HOMO–LUMO proximity
 - User-controlled overrides via input file for fine-grained orbital specification

### **Parallelization**
 - Multiprocessing-enabled CC iterations for high-throughput simulations
 - Scalable architecture designed to handle large and complex molecular systems efficiently

### **Integration with GAMESS**
 - Automated parsing of GAMESS outputs for integrals, MO coefficients, and orbital energies
 - Full compatibility with standard GAMESS output formats (`.dat` for coeffients, orbitals, `_2eint.dat` for 2e integrals)

### **Configurable Input**
 - Intuitive JSON-based configuration (`input.json`) for workflow setup and parameter control

### **Robust Logging & Reproducibility**
 - Structured, rotating log files capturing all computational steps
 - Designed for debugging, reproducibility, and auditability in research and production workflows

---


## ⚙️ Installation

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

## ▶️ Quick Start

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

## 📝 Configuration file input options

```text
|      Flag / Key         |                          Description                                     |
|-------------------------|--------------------------------------------------------------------------|
| `method`                | Correlated method: `ICCSD`, `ICCSD-PT`, `CCSD`                           |
| `conv`                  | Convergence threshold for CC iterations (e.g., `1e-7`)                   |
| `nproc`                 | Number of parallel processes for CC calculations (0 = auto-detect CPUs)  |
| `auto_active`           | automatically determine active orbs from HOMO/LUMO (`true` or `false`)   |
| `active_threshold`      | Energy window (in Hartree) around HOMO/LUMO for auto-active selection    |
| `occ_act`               | Number of active occupied orbitals (used only if `auto_active=false`)    |
| `virt_act`              | Number of active virtual orbitals (used only if `auto_active=false`)     |
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

**Note** In addition to conventional CCSD, two alternative variants are implemented: iCCSDn and iCCSDn-PT, both with comparable scaling to CCSD. Details: 
[A double exponential Coupled Cluster Theory](https://pubs.aip.org/aip/jcp/article/156/24/244117/2841424/A-double-exponential-coupled-cluster-theory-in-the?searchresult=1)

## 📂 Project Structure

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