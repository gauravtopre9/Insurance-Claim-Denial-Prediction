
import pandas as pd
import numpy as np
from src.data_loader import root_dir
from src.logger import *
import warnings
warnings.filterwarnings("ignore")
# Set up logging and data directory
ROOT_DIR = root_dir()
logger = setup_logger(ROOT_DIR)


def engineer(df, df_summary):
    d, eps = df.copy(), 0.00001
    d = d.merge(df_summary, on='payer_id', how='left')

    # Compliances & Administrative Gaps 
    d["auth_gap"]    = ((d["prior_auth_required"]==1) & (d["has_prior_auth"]==0)).astype(int)
    d["ref_gap"]     = ((d["referral_required"]==1)   & (d["referral_present"]==0)).astype(int)
    d["elig_gap"]    = (d["eligibility_verified"]==0).astype(int)
    d["oon"]         = (1 - d["is_in_network"]).astype(int)

    d["cf_total"]    = d["auth_gap"] + d["ref_gap"] + d["missing_documentation_flag"] + d["elig_gap"]
    d["cf_any"]      = (d["cf_total"] > 0).astype(int)
    d["cf_multi"]    = (d["cf_total"] >= 2).astype(int)
    d["cf_all4"]     = (d["cf_total"] == 4).astype(int)

    d["is_admin_clean"] = ((d["cf_total"] == 0) & (d["oon"] == 0)).astype(int)

    # Financial & Clinical KPIs 
    d["b_e_ratio"]   = d["total_billed"] / (d["expected_payment"] + eps)
    d["e_gap"]       = d["total_billed"]  - d["expected_payment"]
    d["cpp"]         = d["total_billed"]  / (d["num_procedures"] + eps)
    d["cpd"]         = d["total_billed"]  / (d["num_diagnoses"]  + eps)

    d["pd_ratio"]      = d["num_procedures"] / (d["num_diagnoses"] + eps)
    d["pd_ratio_exp"]  = d["pd_ratio"] * d['expected_payment']
    d["pd_ratio_bill"] = d["pd_ratio"] * d['total_billed']
    d["items"]         = d["num_procedures"] + d["num_diagnoses"]

    d["cost_per_item"] = d["total_billed"] / (d["items"] + eps)

    d["hi_cost"]     = (d["b_e_ratio"] > 1.5).astype(int)
    d["vhi_cost"]    = (d["b_e_ratio"] > 2.0).astype(int)

    #  Timely Filing 
    d["days_log"]    = np.log1p(d["days_to_submit"])
    d["late_30"]     = (d["days_to_submit"] > 30).astype(int)
    d["late_45"]     = (d["days_to_submit"] > 45).astype(int)
    d["late_60"]     = (d["days_to_submit"] > 60).astype(int)

    # Temporal & Seasonality (Restored & Enhanced) 
    dt               = pd.to_datetime(d["service_month"], format="%Y-%m")
    d["month_num"]   = dt.dt.month
    d["svc_year"]    = dt.dt.year

    # Compound Interactions 
    d["auth_oon"]            = d["auth_gap"] * d["oon"]
    d["auth_late"]           = d["auth_gap"] * d["late_45"]
    d["auth_extreme_late"]   = d["auth_gap"] * d["late_60"]
    d["oon_elig"]            = d["oon"]      * d["elig_gap"]
    d["miss_late"]           = d["missing_documentation_flag"] * d["late_45"]
    d["miss_extreme_late"]   = d["missing_documentation_flag"] * d["late_60"]
    d["hc_auth"]             = d["hi_cost"]  * d["auth_gap"]
    d["hc_oon"]              = d["hi_cost"]  * d["oon"]
    d["triple_risk"]         = ((d["auth_gap"]==1) & (d["oon"]==1) & (d["elig_gap"]==1)).astype(int)

    d["overutilization_risk"] = ((d["num_procedures"] > 3) & (d["num_diagnoses"] <= 1)).astype(int)

    d["clean_but_expensive"] = d["is_admin_clean"] * d["hi_cost"]

    d["high_yield_low_proc"] = ((d["expected_payment"] > 1000) & (d["num_procedures"] == 1)).astype(int)

    return d


def build_payer_profiles_leak_free(hist_df):

    if 'split' in hist_df.columns:
        train_only = hist_df[hist_df['split'] == 'train']


        payer_profile = train_only.groupby(['payer_id']).agg(
            payer_claim_vol=('claim_id', 'count'),
            payer_avg_billed=('total_billed', 'mean'),
            payer_avg_expected=('expected_payment', 'mean'),

            # Payer specific behaviors
            payer_avg_days_to_submit=('days_to_submit', 'mean'),
            payer_auth_req_rate=('prior_auth_required', 'mean'),
            payer_ref_req_rate=('referral_required', 'mean'),

            # Target Encoding - Strictly from Train!
            payer_hist_denial_rate=('is_denied', 'mean')
        ).reset_index()

        # gloab_denial_rate = payer_profile[['payer_id','payer_type','visit_type','payer_hist_denial_rate']]
    else:

        payer_profile = hist_df.groupby(['payer_id']).agg(
            payer_claim_vol=('claim_id', 'count'),
            payer_avg_billed=('total_billed', 'mean'),
            payer_avg_expected=('expected_payment', 'mean'),

            # Payer specific behaviors
            payer_avg_days_to_submit=('days_to_submit', 'mean'),
            payer_auth_req_rate=('prior_auth_required', 'mean'),
            payer_ref_req_rate=('referral_required', 'mean'),
        ).reset_index()


    payer_profile['payer_payment_ratio'] = payer_profile['payer_avg_billed'] / (payer_profile['payer_avg_expected'] + 0.00001)
    payer_profile['payer_payment_gap'] = payer_profile['payer_avg_billed'] - (payer_profile['payer_avg_expected'] + 0.00001)

    return payer_profile


def additional_payer_features(df):
    network_importance = df[df['split']=='train'].groupby(['payer_id','payer_type','visit_type','oon'])['is_denied'].mean().reset_index().rename(columns={'is_denied':'network_importance'})
    auth_importance = df[df['split']=='train'].groupby(['payer_id','payer_type','visit_type','auth_gap'])['is_denied'].mean().reset_index().rename(columns={'is_denied':'auth_importance'})
    ref_importance = df[df['split']=='train'].groupby(['payer_id','payer_type','visit_type','ref_gap'])['is_denied'].mean().reset_index().rename(columns={'is_denied':'ref_importance'})
    elig_importance = df[df['split']=='train'].groupby(['payer_id','payer_type','visit_type','elig_gap'])['is_denied'].mean().reset_index().rename(columns={'is_denied':'elig_importance'})
    payer_importance = df[df['split']=='train'].groupby(['payer_id','payer_type'])['is_denied'].mean().reset_index().rename(columns={'is_denied':'payer_importance'})
    visit_importance = df[df['split']=='train'].groupby(['payer_id','visit_type'])['is_denied'].mean().reset_index().rename(columns={'is_denied':'visit_importance'})

    return network_importance.merge(auth_importance).merge(ref_importance).merge(elig_importance).merge(payer_importance,how='left',on=['payer_id','payer_type']).merge(
        visit_importance,how='left',on=['payer_id','visit_type']
    )


def drop_leakage_columns(hist_fe, cur_fe):

    hist_fe.drop(['cf_all4','triple_risk','auth_extreme_late','svc_year','late_60',
                  "num_procedures_min_rejected","num_procedures_min_accepted","overutilization_risk","auth_late",'miss_extreme_late'],axis=1,
    inplace=True)

    cur_fe.drop(['cf_all4','triple_risk','auth_extreme_late','svc_year',
                 "num_procedures_min_rejected","num_procedures_min_accepted","overutilization_risk","auth_late",'late_60','miss_extreme_late'],axis=1,
    inplace=True)

    return hist_fe, cur_fe


def run_feature_engineering(hist, current, train_eda, df_summary):

    # logger.info(hist.shape, current.shape)  

    hist_fe = engineer(hist, df_summary)
    cur_fe  = engineer(current, df_summary)

    # logger.info(hist_fe.shape, cur_fe.shape)  
    hist_cust_profile = build_payer_profiles_leak_free(hist_fe)
    current_cust_profile = build_payer_profiles_leak_free(cur_fe)
    current_cust_profile = current_cust_profile.merge(hist_cust_profile[['payer_id','payer_hist_denial_rate']]).fillna(0)

    hist_fe = hist_fe.merge(hist_cust_profile,on=['payer_id'])
    cur_fe = cur_fe.merge(current_cust_profile,on=['payer_id'])

    # logger.info(hist_fe.shape, cur_fe.shape)  
    hist_fe = hist_fe.merge(additional_payer_features(hist_fe),how='left')
    cur_fe = cur_fe.merge(additional_payer_features(hist_fe),how='left')

    # logger.info(hist_fe.shape, cur_fe.shape)  
    cur_fe = cur_fe.fillna(0) 


    logger.info(f"  Columns after engineering: {hist_fe.shape[1]}")
    logger.info(f"  Columns after engineering: {cur_fe.shape[1]}\n")

    for col in hist_fe.columns:
        if col not in cur_fe.columns:
            logger.info(col)

    hist_fe, cur_fe = drop_leakage_columns(hist_fe, cur_fe)  

    return hist_fe, cur_fe, hist_cust_profile, current_cust_profile
