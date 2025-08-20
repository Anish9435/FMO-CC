import sys
from fmocc.fmo_processor import FMOProcessor
from fmocc.utils import get_logger

def main():
    logger = get_logger(__name__)
    try:
        processor = FMOProcessor("input.json")
        E_cc, E_cc_tot = processor.run()
        logger.info(f"CC Correlation Energy: {E_cc:.6f}")
        logger.info(f"Total CC Energy: {E_cc_tot:.6f}")
    except Exception as e:
        logger.error(f"Error running FMO-CC: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()