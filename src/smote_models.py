
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

from src.metrics import recall_at_k, vol_threshold
from sklearn.metrics import roc_auc_score
from src.data_loader import root_dir
from src.logger import *
# Set up logging and data directory
ROOT_DIR = root_dir()
logger = setup_logger(ROOT_DIR)


def apply_smote(X_train, y_train, RS):

    sm = SMOTE(sampling_strategy=0.5, random_state=RS)
    X_sm, y_sm = sm.fit_resample(X_train, y_train)
    logger.info(f"  Before: {len(X_train):,} samples  dr={y_train.mean():.3f}")
    logger.info(f"  After : {len(X_sm):,} samples  dr={y_sm.mean():.3f}")
    return X_sm, y_sm


def train_smote_models(X_sm, y_sm, best_lr_p, best_rf_p, best_lgb_p, best_xgb_p):

    # LR on SMOTE
    best_lr_smote_p = {k: v for k, v in best_lr_p.items() if k != "scale_pos_weight"}
    best_lr_smote_p["scale_pos_weight"] = 1.0
    lr_smote = LogisticRegression(**best_lr_p).fit(X_sm, y_sm)

    # RF on SMOTE
    best_rf_smote_p = {k: v for k, v in best_rf_p.items() if k != "scale_pos_weight"}
    best_rf_smote_p["scale_pos_weight"] = 1.0
    rf_smote = RandomForestClassifier(**best_rf_p).fit(X_sm, y_sm)

    # LGB on SMOTE
    best_lgb_smote_p = {k: v for k, v in best_lgb_p.items() if k != "scale_pos_weight"}
    best_lgb_smote_p["scale_pos_weight"] = 1.0
    lgb_smote = lgb.LGBMClassifier(**best_lgb_smote_p).fit(X_sm, y_sm)

    # XGB on SMOTE
    best_xgb_smote_p = {k: v for k, v in best_xgb_p.items() if k != "scale_pos_weight"}
    best_xgb_smote_p["scale_pos_weight"] = 1.0
    xgb_smote = xgb.XGBClassifier(**best_xgb_smote_p).fit(X_sm, y_sm)

    return lr_smote, rf_smote, lgb_smote, xgb_smote


def evaluate_smote_models(lr_smote, rf_smote, lgb_smote, xgb_smote, X_val, y_val, K):
    for name, mdl in [("LR_smote", lr_smote),("RF_smote", rf_smote),("LGB_smote", lgb_smote), ("XGB_smote", xgb_smote)]:
        p = mdl.predict_proba(X_val)[:,1]
        logger.info(f"  {name:15s}  Recall@25%={recall_at_k(y_val,p,K):.4f}  AUC={roc_auc_score(y_val,p):.4f}")


def select_best_variants(X_val, y_val, K,
                          lr_base, lr_tuned, lr_smote,
                          rf_base, rf_tuned, rf_smote,
                          lgb_base, lgb_tuned, lgb_smote,
                          xgb_base, xgb_tuned, xgb_smote):

    def best_variant(variants):
        return max(
            variants,
            key=lambda x: recall_at_k(y_val, x[1].predict_proba(X_val)[:, 1], K)
        )

    lr_name,  final_lr  = best_variant([("base", lr_base),  ("tuned", lr_tuned),  ("smote", lr_smote)])
    rf_name,  final_rf  = best_variant([("base", rf_base),  ("tuned", rf_tuned),  ("smote", rf_smote)])
    lgb_name, final_lgb = best_variant([("base", lgb_base), ("tuned", lgb_tuned), ("smote", lgb_smote)])
    xgb_name, final_xgb = best_variant([("base", xgb_base), ("tuned", xgb_tuned), ("smote", xgb_smote)])

    logger.info(
        f"\nSelected Models:"
        f"\n  LR  :: {lr_name}"
        f"\n  RF  :: {rf_name}"
        f"\n  LGB :: {lgb_name}"
        f"\n  XGB :: {xgb_name}"
    )

    return lr_name, final_lr, rf_name, final_rf, lgb_name, final_lgb, xgb_name, final_xgb
