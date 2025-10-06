"""
Main entry point for executing FMO-CC calculations.

This script serves as a high-level driver for the FMO-CC workflows. It reads a JSON 
configurational file, initializes the computational env, and invokes the FMOProcessor
class to perform the calculations.

Key Responsibilities
--------------------
    - Parse command-line arguments for:
        - Configuration file (`--config` or `-c`)
        - Data directory (`--data_dir` or `-d`)
    - Load and validate the JSON configuration file.
    - Determine the working data directory (explicitly provided or default).
    - Initialize and execute the FMO-CC workflow via the `FMOProcessor`.
    - Log the computed correlation and total energies upon completion.
    - Clean up Python cache directories (`__pycache__`).

Dependencies
-------------
    - fmocc.fmo_processor.FMOProcessor
    - fmocc.utils
    - Python standard libraries: os, sys, json, argparse
"""
import os
import sys
import json
import argparse
from fmocc.fmo_processor import FMOProcessor
from fmocc.utils import HelperFunction, FMOCC_LOGGER

def main():
    parser = argparse.ArgumentParser(description="Run FMO-CC calculation")
    parser.add_argument("-c", "--config", default="input.json", help="Path to input JSON configuration")
    parser.add_argument("-d", "--data_dir", default=None, help="Path to data directory")
    args = parser.parse_args()

    config_file = os.path.abspath(args.config)
    config_dir = os.path.dirname(config_file)
    logger = FMOCC_LOGGER

    with open(config_file) as f:
        config_json = json.load(f)
    
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_data_dir = os.path.join(os.path.dirname(repo_root), "data")
    
    if args.data_dir:
        data_dir = os.path.abspath(args.data_dir)
    else:
        data_dir = os.path.abspath(config_json.get("data_dir", default_data_dir))

    if not os.path.exists(os.path.join(data_dir, f"{config_json.get('common', config_json).get('filename')}.dat")):
        data_dir = os.path.join(os.path.dirname(data_dir), "data")

    try:
        processor = FMOProcessor(config_file, base_dir=data_dir)
        E_cc, E_cc_tot = processor.run()
        logger.info(f"{processor.config.method} Correlation Energy: {E_cc:.9f}")
        logger.info(f"Total {processor.config.method} Energy: {E_cc_tot:.9f}")

        fmocc_dir = os.path.dirname(__file__)
        logger.info(f"Clearing __pycache__ in directory: {fmocc_dir}")
        HelperFunction.clear_pycache(fmocc_dir)
    
    except Exception as e:
        logger.error(f"Error running FMO-CC: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()