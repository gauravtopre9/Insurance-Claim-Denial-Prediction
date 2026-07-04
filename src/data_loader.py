
import pandas as pd

from pathlib import Path
import pandas as pd
import yaml
from src.logger import *


def root_dir():
    return Path(__file__).resolve().parent.parent

ROOT_DIR = root_dir()
logger = setup_logger(ROOT_DIR)

hist_path  = ROOT_DIR / "data" / "claims_history.csv"
current_path  = ROOT_DIR / "data" / "current_claims.csv"

def load_data(hist_path = hist_path, current_path =current_path ):

    hist = pd.read_csv(hist_path)
    current = pd.read_csv(current_path)

    logger.info("=============== Data Display ===============")
    logger.info("= HISTORY ===========================")
    logger.info(hist.head(3))
    logger.info("= CURRENT ===========================")
    logger.info(current.head(3))

    return hist, current
