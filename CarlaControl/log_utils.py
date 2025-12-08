import logging
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent.parent / "scripts.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

_logger = None


def get_logger(name: str = "scripts") -> logging.Logger:
    global _logger
    if _logger:
        return _logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")
    fh.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(fh)
    logger.propagate = False
    _logger = logger
    return logger
