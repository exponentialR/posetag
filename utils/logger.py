from __future__ import annotations
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_FMT = "%(asctime)s | %(levelname)s | %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

def init_project_logger(log_file: Path, level: str = "INFO", console: bool = False) -> logging.Logger:
    """
    Create (or reuse) a project-scoped logger that writes to a rotating file.
    - log_file: full path to the log file (parent dirs created)
    - level: "DEBUG", "INFO", "WARNING", "ERROR"
    - console: also mirror logs to stderr if True
    """
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("posetag")
    if getattr(logger, "_posetag_inited", False):
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fh = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3)
    fh.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
    logger.addHandler(fh)

    if console:
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
        logger.addHandler(sh)

    logger._posetag_inited = True  # type: ignore[attr-defined]
    logger.debug("Logger initialised at %s (level=%s)", log_file, level.upper())
    return logger
