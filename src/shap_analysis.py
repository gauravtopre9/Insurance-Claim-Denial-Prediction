
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from src.data_loader import root_dir
from src.logger import *
# Set up logging and data directory
ROOT_DIR = root_dir()
logger = setup_logger(ROOT_DIR)
CONFIG_SHAP_PLOT_PATH  = ROOT_DIR / "reports" / "shap_values_plot.png"


def run_shap_on_validation(model_list, params, X_val, FEAT_COLS):

    explainer = shap.TreeExplainer(model_list[params["best_model_name"]])

    shap_values = explainer.shap_values(X_val)

    logger.info(type(shap_values))

    if isinstance(shap_values, list):
        for i, s in enumerate(shap_values):
            logger.info(f"Class {i}: {s.shape}")
    else:
        logger.info(shap_values.shape)

    explainer = shap.TreeExplainer(model_list[params["best_model_name"]])

    shap_values = explainer.shap_values(X_val)

    if isinstance(shap_values, list):
        sv = shap_values[1]

    elif len(shap_values.shape) == 3:
        sv = shap_values[:, :, 1]

    else:
        sv = shap_values

    shap.summary_plot(
        sv,
        X_val,
        feature_names=FEAT_COLS,
        max_display=20,
        show=False
    )
    if sv.ndim == 3:
        sv = sv[:, :, 1]

    top10_shap = (
        pd.Series(
            np.abs(sv).mean(axis=0),
            index=FEAT_COLS
        )
        .sort_values(ascending=False)
        .head(10)
    )

    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    # plt.savefig(CONFIG_SHAP_PLOT_PATH)
    plt.savefig("shap_values_plot.png")

    return explainer, shap_values, sv, top10_shap


def run_shap_on_current(explainer, X_cur):

    shap_cur = explainer.shap_values(X_cur)

    if isinstance(shap_cur, list):
        sv_cur = shap_cur[1]
    elif shap_cur.ndim == 3:
        sv_cur = shap_cur[:, :, 1]
    else:
        sv_cur = shap_cur

    return shap_cur, sv_cur
