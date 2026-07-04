
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from src.metrics import recall_at_k
from src.data_loader import root_dir
from src.logger import *

# Set up logging and data directory
ROOT_DIR = root_dir()
logger = setup_logger(ROOT_DIR)


def compute_scale_pos_weight(X_train, y_train, X_val, X_test, X_cur, FEAT_COLS, y_val):

    spw = (y_train==0).sum() / (y_train==1).sum()   # scale_pos_weight for boosting

    logger.info(f"  train:{X_train.shape}  val:{X_val.shape}  "
          f"test:{X_test.shape}  current:{X_cur.shape}")
    logger.info(f"  Feature count     : {len(FEAT_COLS)}")
    logger.info(f"  scale_pos_weight  : {spw:.2f}  (denial rate = {y_train.mean():.3f})")

    return spw


def train_baseline_models(X_train, y_train, spw, RS):

    lr_base = LogisticRegression(class_weight="balanced").fit(X_train, y_train)

    rf_base  = RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                       max_features="sqrt", class_weight="balanced",
                                       random_state=RS, n_jobs=-1).fit(X_train, y_train)

    lgb_base = lgb.LGBMClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.8,
                                    scale_pos_weight=spw, random_state=RS,
                                    n_jobs=-1, verbose=-1).fit(X_train, y_train)

    xgb_base = xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8,
                                   scale_pos_weight=spw, eval_metric="auc",
                                   use_label_encoder=False, random_state=RS,
                                   n_jobs=-1, verbosity=0).fit(X_train, y_train)

    return lr_base, rf_base, lgb_base, xgb_base


def evaluate_baselines(lr_base, rf_base, lgb_base, xgb_base, X_val, y_val, K):

    logger.info(f"  {'Model':20s}  Recall@25%   AUC")
    for name, mdl in [("LR_baseline",lr_base),("RF_baseline",rf_base),("LGB_baseline",lgb_base),("XGB_baseline",xgb_base)]:
        p = mdl.predict_proba(X_val)[:,1]
        logger.info(f"  {name:20s}  {recall_at_k(y_val,p,K):.4f}       {roc_auc_score(y_val,p):.4f}")
