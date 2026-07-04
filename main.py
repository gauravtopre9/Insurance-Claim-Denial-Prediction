import pandas as pd
import joblib
from src.config_loader import load_config
from src.data_loader import load_data,root_dir
from src.logger import setup_logger
from src import eda
from src.feature_engineering import run_feature_engineering
from src.encoding import run_encoding
from src.params import load_params
from src.baseline_models import compute_scale_pos_weight, train_baseline_models, evaluate_baselines
from src.tuning import run_tuning
from src.smote_models import apply_smote, train_smote_models, evaluate_smote_models, select_best_variants
from src.model_selection import build_model_list, run_error_analysis, print_final_scorecard, denial_reason_breakdown
from src.shap_analysis import run_shap_on_validation, run_shap_on_current
from src.scoring import score_current_claims, save_predictions
from src.llm_insights import (
    setup_llm, build_top_bottom_frames,
    generate_top10_explanations, generate_bottom10_explanations,
)


ROOT_DIR = root_dir()
logger = setup_logger(ROOT_DIR)


def main():
    logger.info("=========================================================== Pipeline complete ===========================================================")

    # Load Config

    config = load_config("config.yaml")
    hist, current = load_data()

    # Log EDA results
    eda.run_basic_info(hist, current)
    eda.denial_rate_across_splits(hist, current)
    print(eda.denial_reason_unique(hist))
    print(eda.hist_head(hist))

    train_eda, dr_pt, dr_vt, dr_cf = eda.build_train_eda_and_plots(hist)
    pivot_visit_type, pivot_payer_type = eda.build_pivots(train_eda)
    eda.print_eda_key_findings(train_eda, dr_pt, dr_cf)
    print(eda.train_eda_describe(train_eda))

    df_summary = eda.build_df_summary(train_eda)

    # Feature Engineering Pipe;ine
    hist_fe, cur_fe, hist_cust_profile, current_cust_profile = run_feature_engineering(hist, current, train_eda, df_summary)


    #Encoding Pipeline
    enc = run_encoding(hist_fe, cur_fe, config)
    FEAT_COLS = enc["FEAT_COLS"]
    X_train, y_train = enc["X_train"], enc["y_train"]
    X_val, y_val     = enc["X_val"], enc["y_val"]
    X_test, y_test   = enc["X_test"], enc["y_test"]
    X_cur            = enc["X_cur"]
    hist_enc         = enc["hist_enc"]

    RS, K, N_FOLDS, N_TRIALS = load_params(config)

    #Baseline Models
    spw = compute_scale_pos_weight(X_train, y_train, X_val, X_test, X_cur, FEAT_COLS, y_val)
    lr_base, rf_base, lgb_base, xgb_base = train_baseline_models(X_train, y_train, spw, RS)
    evaluate_baselines(lr_base, rf_base, lgb_base, xgb_base, X_val, y_val, K)

    # Optuna Params Tuning
    tuned = run_tuning(X_train, y_train, X_val, y_val, RS, N_FOLDS, N_TRIALS, K, spw)
    best_lr_p, best_rf_p   = tuned["best_lr_p"], tuned["best_rf_p"]
    best_lgb_p, best_xgb_p = tuned["best_lgb_p"], tuned["best_xgb_p"]
    lr_tuned, rf_tuned     = tuned["lr_tuned"], tuned["rf_tuned"]
    lgb_tuned, xgb_tuned   = tuned["lgb_tuned"], tuned["xgb_tuned"]

    # Smote
    X_sm, y_sm = apply_smote(X_train, y_train, RS)
    lr_smote, rf_smote, lgb_smote, xgb_smote = train_smote_models(
        X_sm, y_sm, best_lr_p, best_rf_p, best_lgb_p, best_xgb_p
    )
    evaluate_smote_models(lr_smote, rf_smote, lgb_smote, xgb_smote, X_val, y_val, K)

    (lr_name, final_lr, rf_name, final_rf,
     lgb_name, final_lgb, xgb_name, final_xgb) = select_best_variants(
        X_val, y_val, K,
        lr_base, lr_tuned, lr_smote,
        rf_base, rf_tuned, rf_smote,
        lgb_base, lgb_tuned, lgb_smote,
        xgb_base, xgb_tuned, xgb_smote,
    )


    # ERROR ANALYSIS + BEST MODEL SELECTION
    params, model_list = build_model_list(rf_tuned, lgb_tuned, xgb_tuned, rf_smote, lgb_smote, xgb_smote)
    params, test_df, denials, captured, missed = run_error_analysis(
        model_list, params, hist_enc, X_val, X_test, K
    )
    print_final_scorecard(params, y_test, K)
    denial_reason_breakdown(test_df, missed, captured)

   # SHAP Analysis
    explainer, shap_values, sv, top10_shap = run_shap_on_validation(model_list, params, X_val, FEAT_COLS)
    shap_cur, sv_cur = run_shap_on_current(explainer, X_cur)
    
    ## SAVE BEST MODEL
    MODEL_DIR = ROOT_DIR / "models"
    model_path = MODEL_DIR / "best_model.pkl"
    best_model = params["best_model"]
    joblib.dump(best_model, model_path)

    # Score Current Claims
    model = joblib.load(model_path)
    current_df = score_current_claims(model, X_cur, current, K)
    save_predictions(current_df)

    # LLM Based Insights
    llm, parser, chain = setup_llm()
    top10_df, bottom10_df = build_top_bottom_frames(current_df)

    top10_df = generate_top10_explanations(top10_df, chain)
    bottom10_df = generate_bottom10_explanations(bottom10_df, chain)

    logger.info("\nPipeline complete.")


if __name__ == "__main__":
    main()
