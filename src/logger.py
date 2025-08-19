import logging
from logging.handlers import RotatingFileHandler
from typing import Optional
import os

def get_logger(name: str, log_file: str = "fmo_cc.log") -> logging.Logger:
    """
    Configure and return a logger with console and file handlers.
    
    Args:
        name (str): Logger name, typically module name.
        log_file (str): Path to log file.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    # CHANGE: Added centralized logging system
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger
    
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # File handler with rotation
    try:
        file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except (PermissionError, OSError) as e:
        console_handler.error(f"Failed to set up file handler for logging: {e}")
    
    logger.addHandler(console_handler)
    return logger