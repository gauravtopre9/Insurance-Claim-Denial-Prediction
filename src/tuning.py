
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import optuna
from sklearn.metrics import roc_auc_score
from src.data_loader import root_dir
from src.logger import *
# Set up logging and data directory
ROOT_DIR = root_dir()
logger = setup_logger(ROOT_DIR)

optuna.logging.set_verbosity(optuna.logging.WARNING)

from src.metrics import recall_at_k


def make_cv_recall(X_train, y_train, RS, N_FOLDS, K):

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RS)

    def cv_recall(model_cls, params, X=X_train, y=y_train):
        scores = []
        for tr_i, vl_i in skf.split(X, y):
            m = model_cls(**params)
            m.fit(X[tr_i], y[tr_i])
            scores.append(recall_at_k(y[vl_i], m.predict_proba(X[vl_i])[:,1], K))
        return float(np.mean(scores))

    return skf, cv_recall


def tune_logistic_regression(cv_recall, RS, N_TRIALS):
    def lr_obj(trial):
        return cv_recall(
            LogisticRegression,
            dict(
                C=trial.suggest_float("C", 1e-3, 100, log=True),
                penalty="l2",
                solver="liblinear",
                class_weight=trial.suggest_categorical(
                    "class_weight",
                    ["balanced"]
                ),
                max_iter=1000,
                random_state=RS
            )
        )

    lr_study = optuna.create_study(direction="maximize")
    lr_study.optimize(lr_obj, n_trials=N_TRIALS, show_progress_bar=True)
    best_lr_p = lr_study.best_params | {"random_state": RS, "n_jobs": -1}
    logger.info(f"  LR  best CV Recall@25% : {lr_study.best_value:.4f}")
    return lr_study, best_lr_p


def tune_random_forest(cv_recall, RS, N_TRIALS):

    def rf_obj(trial):
        return cv_recall(RandomForestClassifier, dict(
            n_estimators      = trial.suggest_int("n_estimators", 200, 500),
            max_depth         = trial.suggest_int("max_depth", 5, 20),
            min_samples_leaf  = trial.suggest_int("min_samples_leaf", 1, 20),
            max_features      = trial.suggest_categorical("max_features", ["sqrt","log2",0.3,0.5]),
            min_samples_split = trial.suggest_int("min_samples_split", 2, 15),
            class_weight      = trial.suggest_categorical("class_weight",
                                    ["balanced","balanced_subsample"]),
            random_state=RS, n_jobs=-1
        ))

    rf_study = optuna.create_study(direction="maximize")
    rf_study.optimize(rf_obj, n_trials=N_TRIALS, show_progress_bar=True)
    best_rf_p = rf_study.best_params | {"random_state": RS, "n_jobs": -1}
    logger.info(f"  RF  best CV Recall@25% : {rf_study.best_value:.4f}")
    return rf_study, best_rf_p


def tune_lightgbm(cv_recall, RS, N_TRIALS, spw):

    def lgb_obj(trial):
        return cv_recall(lgb.LGBMClassifier, dict(
            num_leaves        = trial.suggest_int("num_leaves", 15, 500),
            max_depth         = trial.suggest_int("max_depth", 3, 20),
            learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            n_estimators      = trial.suggest_int("n_estimators", 100, 700),
            min_child_samples = trial.suggest_int("min_child_samples", 5, 100),
            subsample         = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree  = trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_alpha         = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda        = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            scale_pos_weight  = trial.suggest_float("scale_pos_weight", 1.0, spw * 1.5),
            random_state=RS, n_jobs=-1, verbose=-1
        ))

    lgb_study = optuna.create_study(direction="maximize")
    lgb_study.optimize(lgb_obj, n_trials=N_TRIALS, show_progress_bar=True)
    best_lgb_p = lgb_study.best_params | {"random_state": RS, "n_jobs": -1, "verbose": -1}
    logger.info(f"  LGB best CV Recall@25% : {lgb_study.best_value:.4f}")
    return lgb_study, best_lgb_p


def tune_xgboost(cv_recall, RS, N_TRIALS, spw):

    def xgb_obj(trial):
        return cv_recall(xgb.XGBClassifier, dict(
            n_estimators      = trial.suggest_int("n_estimators", 100, 600),
            max_depth         = trial.suggest_int("max_depth", 2, 20),
            learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample         = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree  = trial.suggest_float("colsample_bytree", 0.4, 1.0),
            min_child_weight  = trial.suggest_int("min_child_weight", 1, 20),
            gamma             = trial.suggest_float("gamma", 1e-8, 5.0, log=True),
            reg_alpha         = trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
            reg_lambda        = trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True),
            scale_pos_weight  = trial.suggest_float("scale_pos_weight", 1.0, spw * 1.5),
            eval_metric="auc", use_label_encoder=False, random_state=RS, n_jobs=-1, verbosity=0
        ))

    xgb_study = optuna.create_study(direction="maximize")
    xgb_study.optimize(xgb_obj, n_trials=N_TRIALS, show_progress_bar=True)
    best_xgb_p = xgb_study.best_params | {
        "eval_metric":"auc","use_label_encoder":False,"random_state":RS,"n_jobs":-1,"verbosity":0
    }
    logger.info(f"  XGB best CV Recall@25% : {xgb_study.best_value:.4f}")
    return xgb_study, best_xgb_p


def fit_tuned_models(best_lr_p, best_rf_p, best_lgb_p, best_xgb_p, X_train, y_train, X_val, y_val, K):

    lr_tuned  = LogisticRegression(**best_lr_p).fit(X_train, y_train)
    rf_tuned  = RandomForestClassifier(**best_rf_p).fit(X_train, y_train)
    lgb_tuned = lgb.LGBMClassifier(**best_lgb_p).fit(X_train, y_train)
    xgb_tuned = xgb.XGBClassifier(**best_xgb_p).fit(X_train, y_train)

    logger.info(f"\n  {'Model':15s}  Recall@25%   AUC   (validation)")
    for name, mdl in [("LR_tuned",lr_tuned),("RF_tuned",rf_tuned),("LGB_tuned",lgb_tuned),("XGB_tuned",xgb_tuned)]:
        p = mdl.predict_proba(X_val)[:,1]
        logger.info(f"  {name:15s}  {recall_at_k(y_val,p,K):.4f}       {roc_auc_score(y_val,p):.4f}")

    return lr_tuned, rf_tuned, lgb_tuned, xgb_tuned


def run_tuning(X_train, y_train, X_val, y_val, RS, N_FOLDS, N_TRIALS, K, spw):

    skf, cv_recall = make_cv_recall(X_train, y_train, RS, N_FOLDS, K)

    lr_study, best_lr_p = tune_logistic_regression(cv_recall, RS, N_TRIALS)
    rf_study, best_rf_p = tune_random_forest(cv_recall, RS, N_TRIALS)
    lgb_study, best_lgb_p = tune_lightgbm(cv_recall, RS, N_TRIALS, spw)
    xgb_study, best_xgb_p = tune_xgboost(cv_recall, RS, N_TRIALS, spw)

    lr_tuned, rf_tuned, lgb_tuned, xgb_tuned = fit_tuned_models(
        best_lr_p, best_rf_p, best_lgb_p, best_xgb_p, X_train, y_train, X_val, y_val, K
    )

    return {
        "best_lr_p": best_lr_p, "best_rf_p": best_rf_p,
        "best_lgb_p": best_lgb_p, "best_xgb_p": best_xgb_p,
        "lr_tuned": lr_tuned, "rf_tuned": rf_tuned,
        "lgb_tuned": lgb_tuned, "xgb_tuned": xgb_tuned,
    }
