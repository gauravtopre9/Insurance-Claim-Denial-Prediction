from src.data_loader import root_dir
from src.logger import *
import matplotlib.pyplot as plt
import pandas as pd
from src.utils import display
# Set up logging and data directory
ROOT_DIR = root_dir()
logger = setup_logger(ROOT_DIR)


def run_basic_info(hist, current):
    display(hist.info())
    display(current.info())


def denial_rate_across_splits(hist, current):

    for s in ["train", "validation", "test"]:
        sub = hist[hist["split"] == s]
        logger.info(f"  {s:12s}  n={len(sub):,}  "
              f"denial_rate={sub['is_denied'].mean():.3f}  "
              f"n_denied={int(sub['is_denied'].sum()):,}")
    logger.info(f"  {'current':12s}  n={len(current):,}  (no labels)")


def denial_reason_unique(hist):
    return hist['denial_reason'].unique()


def hist_head(hist):
    return hist.head()


def build_train_eda_and_plots(hist):
    train_eda = hist[hist["split"] == "train"].copy()
    train_eda["n_cf"] = (
        ((train_eda["prior_auth_required"] == 1) & (train_eda["has_prior_auth"] == 0)).astype(int) +
        ((train_eda["referral_required"] == 1) & (train_eda["referral_present"] == 0)).astype(int) +
        train_eda["missing_documentation_flag"] +
        (train_eda["eligibility_verified"] == 0).astype(int)
        +
        (train_eda["is_in_network"] == 0).astype(int)
    )

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # Denial rate by payer type
    dr_pt = train_eda.groupby("payer_type")["is_denied"].mean().sort_values(ascending=False)
    axes[0, 0].bar(dr_pt.index, dr_pt.values, color="steelblue")
    axes[0, 0].set_title("Denial Rate by Payer Type"); axes[0, 0].set_ylabel("Denial Rate")
    axes[0, 0].tick_params(axis="x", rotation=15)
    for i, v in enumerate(dr_pt.values):
        axes[0, 0].text(i, v + 0.005, f"{v:.2f}", ha="center", fontsize=9)

    # Denial rate by visit type
    dr_vt = train_eda.groupby("visit_type")["is_denied"].mean().sort_values(ascending=False)
    axes[0, 1].bar(dr_vt.index, dr_vt.values, color="darkorange")
    axes[0, 1].set_title("Denial Rate by Visit Type"); axes[0, 1].set_ylabel("Denial Rate")
    for i, v in enumerate(dr_vt.values):
        axes[0, 1].text(i, v + 0.005, f"{v:.2f}", ha="center", fontsize=9)

    # Denial rate vs compliance failure count (key insight)
    dr_cf = train_eda.groupby("n_cf")["is_denied"].mean()
    axes[0, 2].bar(dr_cf.index.astype(str), dr_cf.values, color="seagreen")
    axes[0, 2].set_title("Denial Rate by # Compliance Gaps"); axes[0, 2].set_xlabel("# Gaps")
    for i, v in enumerate(dr_cf.values):
        axes[0, 2].text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)

    # Days to submit — denied vs not
    for outcome, label, color in [(0, "Not denied", "steelblue"), (1, "Denied", "tomato")]:
        axes[1, 0].hist(train_eda[train_eda["is_denied"] == outcome]["days_to_submit"].clip(0, 200),
                         bins=40, alpha=0.6, color=color, label=label, density=True)
    axes[1, 0].set_title("Days to Submit Distribution")
    axes[1, 0].set_xlabel("Days"); axes[1, 0].legend()

    # Billed / expected ratio
    ratio = train_eda["total_billed"] / (train_eda["expected_payment"] + 1)
    for outcome, label, color in [(0, "Not denied", "steelblue"), (1, "Denied", "tomato")]:
        axes[1, 1].hist(ratio[train_eda["is_denied"] == outcome].clip(0, 5),
                         bins=40, alpha=0.6, color=color, label=label, density=True)
    axes[1, 1].set_title("Billed/Expected Ratio")
    axes[1, 1].set_xlabel("Ratio"); axes[1, 1].legend()

    plt.suptitle("EDA — Denial Rate Drivers (Training Data)", fontsize=13, y=1.01)
    plt.tight_layout()

    return train_eda, dr_pt, dr_vt, dr_cf


def build_pivots(train_eda):

    pivot_visit_type = train_eda.pivot_table("is_denied", "payer_id", "visit_type", aggfunc="mean")
    pivot_payer_type = train_eda.pivot_table("is_denied", "payer_id", "payer_type", aggfunc="mean")
    display(pivot_payer_type.fillna(0)), display(pivot_visit_type.fillna(0))
    return pivot_visit_type, pivot_payer_type


def print_eda_key_findings(train_eda, dr_pt, dr_cf):

    logger.info("################## EDA Key Findings ##################")
    logger.info(f"  Total Claims in Training Data : {train_eda['claim_id'].nunique()}")
    logger.info(f"  Total Denied Claims in Training Data : {train_eda['is_denied'].sum()}")
    logger.info("-" * 50)
    logger.info(f"  Reasons for denial:: \n {train_eda.groupby(['denial_reason'])['is_denied'].sum()}")
    logger.info("-" * 50)

    logger.info(f"  Highest denial payer type : {dr_pt.index[0]} ({dr_pt.iloc[0]:.1%})")
    logger.info(f"  Lowest  denial payer type : {dr_pt.index[-1]} ({dr_pt.iloc[-1]:.1%})")
    logger.info(f"  0 compliance gaps denial rate: {dr_cf.get(0, 0):.1%}")
    logger.info(f"  3+ compliance gaps denial rate: {dr_cf.iloc[-2]:.1%}")
    logger.info(f"  Auth gap present   : {train_eda[(train_eda['prior_auth_required']==1)&(train_eda['has_prior_auth']==0)]['is_denied'].mean():.1%}")
    logger.info(f"  Ref gap present   : {train_eda[(train_eda['referral_required']==1)&(train_eda['referral_present']==0)]['is_denied'].mean():.1%}")
    logger.info(f"  Eligibility not verified: {train_eda[train_eda['eligibility_verified']==0]['is_denied'].mean():.1%}")
    logger.info(f"  Out of network     : {train_eda[train_eda['is_in_network']==0]['is_denied'].mean():.1%}")


def train_eda_describe(train_eda):
    return train_eda.describe()


def build_df_summary(train_eda):

    df_summary = (
        train_eda
        .groupby(['payer_id', 'is_denied'])
        .agg({
            'expected_payment': 'mean',
            'num_procedures': ['min', 'max'],
            'days_to_submit': ['min', 'max', 'mean']
        })
    )

    df_summary = df_summary.unstack('is_denied')

    df_summary.columns = [
        f"{col}_{agg}_{'accepted' if denied == 0 else 'rejected'}"
        for col, agg, denied in df_summary.columns
    ]

    df_summary = df_summary.reset_index()

    df_summary.head()

    return df_summary
