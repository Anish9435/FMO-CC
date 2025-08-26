import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
from fmocc.fmo_processor import FMOProcessor
from fmocc.utils import HelperFunction, FMOCC_LOGGER

def main():
    logger = FMOCC_LOGGER
    try:
        processor = FMOProcessor("input.json")
        E_cc, E_cc_tot = processor.run()
        logger.info(f"CC Correlation Energy: {E_cc:.9f}")
        logger.info(f"Total CC Energy: {E_cc_tot:.9f}")
        fmocc_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src", "fmocc")
        logger.info(f"Clearing __pycache__ in directory: {fmocc_dir}")
        HelperFunction.clear_pycache(fmocc_dir)
    except Exception as e:
        logger.error(f"Error running FMO-CC: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()