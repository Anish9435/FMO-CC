"""
Logging, cache management, and numerical utilities for FMO-CC calculations

This module provides core utilities for the FMO-CC workflows, including standardized logging
configuration, cache and environment management, and numerical tensor operations. It ensures
consistent console and file logging, supports colorized terminal output for debugging, and
implements essential helpers for updating coupled-cluster amplitudes efficiently and correctly.

Key Components
--------------
    - CustomColoredFormatter : Colorizes FMOCC log output with contextual emphasis.
    - FMOCC_LOGGER : Preconfigured global logger instance.
    - HelperFunction : Includes methods for cache cleanup and environment maintenance.
    - Symmetrizer : Ensures proper permutation symmetry for CC residual tensors.
    - AmplitudeUpdater : Handles iterative amplitude updates (T₁, T₂, Sₒ, Sᵥ).

Dependencies
-------------
    - Python standard libraries: os, gc, re, shutil, logging
    - External libraries: numpy, colorama, colorlog
    - typing (Optional)
    - logging.handlers (RotatingFileHandler, QueueHandler)
"""

import os
import gc
import re
import shutil
import logging
import copy as cp
import numpy as np
from queue import Queue
from colorama import init  # type: ignore
from colorlog import ColoredFormatter
from logging.handlers import RotatingFileHandler, QueueHandler

init(autoreset=True)
log_queue = Queue()
class CustomColoredFormatter(ColoredFormatter):
    def format(self, record):
        timestamp = self.formatTime(record, self.datefmt)
        levelname = record.levelname
        green = "\033[32m"
        reset = "\033[0m"
        formatted_prefix = f"{green}{timestamp} [{levelname}]:{reset}"
        yellow = "\033[33m"
        formatted_details = f"{yellow}{record.filename}:{record.funcName}:{record.lineno} -{reset}"
        colored_msg = self._auto_color_message(str(record.msg))
        formatted_message = f"{formatted_prefix} {formatted_details} {colored_msg}"
        return formatted_message
    
    def _auto_color_message(self, message: str) -> str:
        RESET = "\033[0m"
        CYAN = "\033[36m"
        MAGENTA = "\033[35m"

        # Color file paths (e.g., /path/to/file)
        message = re.sub(r'(/[\w\-/\.]+)', rf'{CYAN}\1{RESET}', message)

        # Highlight fmocc-specific keywords
        message = re.sub(r'\b(fragment|monomer|dimer|energy|correlation|converged)\b',
                         rf'{MAGENTA}\1{RESET}', message, flags=re.IGNORECASE)

        return message

formatter = CustomColoredFormatter(
    "%(message)s",  # Message is fully formatted in CustomColoredFormatter
    datefmt="%Y-%m-%d %H:%M:%S",
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'white',  # Message body in white
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
)

# Stream handler for console output
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)

# File handler for logging to file
log_file = "fmocc.log"
file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
plain_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s]: %(filename)s:%(funcName)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(plain_formatter)
file_handler.setLevel(logging.DEBUG)

# QueueHandler for log queue
queue_handler = QueueHandler(log_queue)
queue_handler.setFormatter(plain_formatter)
queue_handler.setLevel(logging.DEBUG)

FMOCC_LOGGER = logging.getLogger("fmocc")
FMOCC_LOGGER.setLevel(logging.DEBUG)
FMOCC_LOGGER.addHandler(console_handler)
FMOCC_LOGGER.addHandler(file_handler)
FMOCC_LOGGER.addHandler(queue_handler)
FMOCC_LOGGER.propagate = False

class HelperFunction:
    """"Provides static utility functions for system maintenance and cleanup"""

    @staticmethod
    def clear_pycache(directory="."):
        """
        Remove all `__pycache__` directories recursively from a given path.

        Parameters
        ----------
        directory : str, optional
            Base directory to search for cache directories (default: current directory).

        Returns
        -------
        int
            Number of `__pycache__` directories successfully deleted.
        """
        deleted_count = 0
        if not os.path.exists(directory):
            FMOCC_LOGGER.error(f"Directory {directory} does not exist.")
            return deleted_count
        try:
            for root, dirs, _ in os.walk(directory, topdown=False):
                for dir_name in dirs:
                    if dir_name == "__pycache__":
                        pycache_path = os.path.join(root, dir_name)
                        FMOCC_LOGGER.info(f"Deleting __pycache__ directory: {pycache_path}")
                        try:
                            shutil.rmtree(pycache_path)
                            deleted_count += 1
                        except Exception as e:
                            FMOCC_LOGGER.warning(f"Failed to delete {pycache_path}: {str(e)}")
            FMOCC_LOGGER.info(f"Total __pycache__ directories deleted: {deleted_count}")
            return deleted_count
        except Exception as e:
            FMOCC_LOGGER.error(f"Error while clearing __pycache__ directories in {directory}: {str(e)}")
            return deleted_count

class Symmetrizer:
    """
    Handles tensor symmetrization operations used in coupled-cluster updates.

    Ensures correct exchange symmetry for the doubles residual tensor (R_ijab),
    where R_ijab = R_jiab + R_ijba.
    """
    def symmetrize(self, occ, virt, R_ijab):
        R_ijab_new = np.zeros((occ, occ, virt, virt))
        for i in range(occ):
            for j in range(occ):
                for a in range(virt):
                    for b in range(virt):
                        R_ijab_new[i, j, a, b] = R_ijab[i, j, a, b] + R_ijab[j, i, b, a]
        R_ijab = cp.deepcopy(R_ijab_new)
        del R_ijab_new
        gc.collect()
        return R_ijab

class AmplitudeUpdater:
    """
    Implements update equations for cluster amplitudes.

    Provides convergence-controlled iterative update routines for CC and
    amplitudes — including single, double, and Sᵥ/Sₒ amplitudes.

    Methods
    -------
    update_t2(R_ijab, t2, D2, conv)
        Update doubles amplitudes t₂.
    update_t1t2(R_ia, R_ijab, t1, t2, D1, D2)
        Joint update of single and double amplitudes.
    update_So(R_ijav, So, Do, conv)
        Update scattering operator Sₒ.
    update_Sv(R_iuab, Sv, Dv, conv)
        Update scattering operator Sᵥ.
    """
    def update_t2(self, R_ijab, t2, D2, conv):
        ntmax = 0
        eps_t = 100
        if eps_t >= conv:
            delt2 = np.divide(R_ijab, D2)
            t2 += delt2
        ntmax = np.size(t2)
        eps_t = float(np.sum(abs(R_ijab)) / ntmax)
        return eps_t, t2, R_ijab

    def update_t1t2(self, R_ia, R_ijab, t1, t2, D1, D2):
        ntmax = 0
        eps = 100
        delt2 = np.divide(R_ijab, D2)
        delt1 = np.divide(R_ia, D1)
        t1 += delt1
        t2 += delt2
        ntmax = np.size(t1) + np.size(t2)
        eps = float(np.sum(abs(R_ia) + np.sum(abs(R_ijab)))/ntmax)
        return eps, t1, t2

    def update_So(self, R_ijav, So, Do, conv):
        ntmax = 0
        eps_So = 100
        if eps_So >= conv:
            delSo = np.divide(R_ijav, Do)
            So += delSo
        ntmax = np.size(So)
        eps_So = float(np.sum(abs(R_ijav)) / ntmax)
        return eps_So, So

    def update_Sv(self, R_iuab, Sv, Dv, conv):
        ntmax = 0
        eps_Sv = 100
        if eps_Sv >= conv:
            delSv = np.divide(R_iuab, Dv)
            Sv += delSv
        ntmax = np.size(Sv)
        eps_Sv = float(np.sum(abs(R_iuab)) / ntmax)
        return eps_Sv, Sv