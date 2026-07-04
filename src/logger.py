import logging
from pathlib import Path

def setup_logger(root_dir):

    log_dir = Path(root_dir) / "logs"
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / "pipeline.log"

    logger = logging.getLogger()

    # Prevent duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger