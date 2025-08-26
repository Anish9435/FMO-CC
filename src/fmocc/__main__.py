import os
import sys
import argparse
from fmocc.fmo_processor import FMOProcessor
from fmocc.utils import HelperFunction, FMOCC_LOGGER

def main():
    parser = argparse.ArgumentParser(
        description="Run FMO-CC calculations with a specified configuration file."
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="input.json",
        help="Path to the input JSON configuration file (default: input.json)."
    )
    args = parser.parse_args()
    logger = FMOCC_LOGGER
    try:
        processor = FMOProcessor(args.config)
        E_cc, E_cc_tot = processor.run()
        logger.info(f"CC Correlation Energy: {E_cc:.9f}")
        logger.info(f"Total CC Energy: {E_cc_tot:.9f}")

        fmocc_dir = os.path.dirname(__file__)
        logger.info(f"Clearing __pycache__ in directory: {fmocc_dir}")
        HelperFunction.clear_pycache(fmocc_dir)
    
    except Exception as e:
        logger.error(f"Error running FMO-CC: {e}")
        sys.exit(1)