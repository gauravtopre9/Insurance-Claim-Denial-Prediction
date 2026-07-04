
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              classification_report, brier_score_loss)
from src.data_loader import root_dir
from src.logger import *
# Set up logging and data directory
ROOT_DIR = root_dir()
logger = setup_logger(ROOT_DIR)


def recall_at_k(y, p, k):
    n = max(1, int(np.ceil(k * len(y))))
    return float(y[np.argsort(p)[::-1][:n]].sum() / (y.sum() + 1e-9))


def precision_at_k(y, p, k):
    n = max(1, int(np.ceil(k * len(y))))
    return float(y[np.argsort(p)[::-1][:n]].mean())


def lift_at_k(y, p, k):
    return precision_at_k(y, p, k) / (y.mean() + 1e-9)


def vol_threshold(p, k):
    return float(np.percentile(p, (1 - k) * 100))


def print_scorecard(label, y, p, k):
    t = vol_threshold(p, k)
    logger.info(f"  {label}")
    logger.info(f"    Recall@{k:.0%}      {recall_at_k(y,p,k):.4f}  ← north-star")
    logger.info(f"    Precision@{k:.0%}   {precision_at_k(y,p,k):.4f}")
    logger.info(f"    Lift@{k:.0%}        {lift_at_k(y,p,k):.2f}×")
    logger.info(f"    ROC-AUC       {roc_auc_score(y,p):.4f}")
    logger.info(f"    PR-AUC        {average_precision_score(y,p):.4f}")
    logger.info(f"    Brier Score   {brier_score_loss(y,p):.4f}")
