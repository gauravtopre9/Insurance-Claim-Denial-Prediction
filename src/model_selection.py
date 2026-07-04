
from sklearn.metrics import classification_report
import pandas as pd
from src.metrics import vol_threshold, print_scorecard
from src.data_loader import root_dir
from src.logger import *
# Set up logging and data directory
ROOT_DIR = root_dir()
logger = setup_logger(ROOT_DIR)


def build_model_list(rf_tuned, lgb_tuned, xgb_tuned, rf_smote, lgb_smote, xgb_smote):

    params = {
        "best_model_name": None,
        "best_model": None,
        "best_val_p": None,
        "best_test_p": None,
        "best_threshold": None,
        "capture_rate": 0.0
    }

    model_list = {
        "rf_tuned": rf_tuned,
        "lgb_tuned": lgb_tuned,
        "xgb_tuned": xgb_tuned,
        "rf_smote": rf_smote,
        "lgb_smote": lgb_smote,
        "xgb_smote": xgb_smote
    }

    return params, model_list


def run_error_analysis(model_list, params, hist_enc, X_val, X_test, K):

    for model_name, model in model_list.items():

        # Predict probabilities
        val_p = model.predict_proba(X_val)[:, 1]
        test_p = model.predict_proba(X_test)[:, 1]

        # Calibrate threshold on validation set
        threshold = vol_threshold(val_p, K)
        preds_test = (test_p >= threshold).astype(int)

        # Error Analysis
        test_df = (
            hist_enc[hist_enc["split"] == "test"]
            .copy()
            .reset_index(drop=True)
        )

        test_df["prob"] = test_p
        test_df["in_top25"] = preds_test

        denials = test_df[test_df["is_denied"] == 1]
        captured = denials[denials["in_top25"] == 1]
        missed = denials[denials["in_top25"] == 0]

        capture_rate = len(captured) / len(denials)

        logger.info(f"\nError Analysis: {model_name}")
        logger.info(f"Total denials   : {len(denials)}")
        logger.info(f"Captured (@25%) : {len(captured)} ({capture_rate:.1%})")
        logger.info(f"Missed (FN)     : {len(missed)} ({1-capture_rate:.1%})")

        # Keep best model
        if capture_rate > params["capture_rate"]:
            params.update({
                "best_model_name": model_name,
                "best_model": model,
                "best_val_p": val_p,
                "best_test_p": test_p,
                "best_threshold": threshold,
                "capture_rate": capture_rate
            })

    return params, test_df, denials, captured, missed


def print_final_scorecard(params, y_test, K):

    t_thresh = vol_threshold(params["best_val_p"], K)
    preds_ts = (params["best_test_p"] >= t_thresh).astype(int)
    logger.info(f" Threshold (top-25% volume, val-calibrated) : {t_thresh:.4f}")
    print_scorecard(params["best_model_name"], y_test, params["best_test_p"], K)
    logger.info(classification_report(y_test, preds_ts, digits=3))
    return t_thresh, preds_ts


def denial_reason_breakdown(test_df, missed, captured):
    if "denial_reason" in test_df.columns:
        miss_r = missed["denial_reason"].value_counts()
        cap_r  = captured["denial_reason"].value_counts()
        ea     = pd.DataFrame({"Missed": miss_r, "Captured": cap_r}).fillna(0).astype(int)
        ea["Capture_%"] = (ea["Captured"] / (ea["Missed"]+ea["Captured"]) * 100).round(1)
        print(f"\n  Denial reason capture rates:")
        print(ea.sort_values("Capture_%").to_string())

    logger.info(f"\n  Payer type — Missed vs Captured:")
    pt_ea = pd.DataFrame({
        "Missed_%":   missed["payer_type"].value_counts(normalize=True).round(3),
        "Captured_%": captured["payer_type"].value_counts(normalize=True).round(3)
    }).fillna(0)
    logger.info(pt_ea.to_string())

    logger.info(f"\n  Visit type — Missed vs Captured:")
    pt_ea = pd.DataFrame({
        "Missed_%":   missed["visit_type"].value_counts(normalize=True).round(3),
        "Captured_%": captured["visit_type"].value_counts(normalize=True).round(3)
    }).fillna(0)
    logger.info(pt_ea.to_string())
