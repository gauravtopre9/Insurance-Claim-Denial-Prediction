
from src.metrics import vol_threshold
from src.data_loader import root_dir
from src.logger import *
from src.llm_insights import assign_risk_tier
# Set up logging and data directory
ROOT_DIR = root_dir()
logger = setup_logger(ROOT_DIR)


current_scored_path  = ROOT_DIR / "outputs" / "Predictions_current_claims.csv"
current_scored_path_for_insights  = ROOT_DIR / "outputs" / "Predictions_current_claims_for_insights.csv"


def score_current_claims(model, X_cur, current, K):

    best_model = model
    best_cur_p = best_model.predict_proba(X_cur)[:, 1]
    threshold = vol_threshold(best_cur_p, K)
    preds_cur = (best_cur_p >= threshold).astype(int)
    current_df = (
            current
            .copy()
            .reset_index(drop=True)
        )
    current_df["denial_probability"] = best_cur_p
    current_df["predicted_denial"] = preds_cur

    current_df = current_df.sort_values(by=['denial_probability'],ascending=False)
    current_df["risk_tier"] = current_df["denial_probability"].apply(assign_risk_tier)

    return current_df


def save_predictions(current_df, path=current_scored_path):
    current_df_insights = current_df.copy(deep=True)
    current_df = current_df[["claim_id","denial_probability",
                "predicted_denial","risk_tier"]]
    current_df = current_df.sort_values(by=['denial_probability'],ascending=False)
    current_df.to_csv(path, index=False)
    current_df_insights.to_csv(current_scored_path_for_insights, index=False)
