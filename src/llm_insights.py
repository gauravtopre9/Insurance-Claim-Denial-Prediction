
import time
import os
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.data_loader import root_dir
from src.logger import *

# Set up logging and data directory
ROOT_DIR = root_dir()
logger = setup_logger(ROOT_DIR)

from pathlib import Path

env_path = Path(__file__).resolve().parent.parent / ".env"

with open(env_path) as f:
    for line in f:
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        key, value = line.split("=", 1)
        os.environ[key] = value

groq_api_key = os.getenv("GROQ_API_KEY")
groq_llm_model = os.getenv("MODEL_NAME")


bottom_10_path  = ROOT_DIR / "outputs" / "bottom10_claim_explanations.csv"
top_10_path  = ROOT_DIR / "outputs" / "top10_claim_explanations.csv"


# PROMPT TEMPLATE
EXPLANATION_PROMPT = ChatPromptTemplate.from_template("""
You are an expert claims revenue-cycle analyst assistant. A predictive model has
scored the insurance claim below for denial risk. Write a short explanation
that a front-end review analyst can read and act on in seconds.

CLAIM FIELDS:
  Claim ID:              {claim_id}
  Payer:                 {payer_type} ({payer_id})
  Visit type:            {visit_type}
  Total billed:          ${total_billed:,.2f}
  Expected payment:      ${expected_payment:,.2f}
  Procedures / Diagnoses: {num_procedures} / {num_diagnoses}
  Prior auth required / on file: {prior_auth_required} / {has_prior_auth}
  Referral required / present:   {referral_required} / {referral_present}
  In network:            {is_in_network}
  Days to submit:        {days_to_submit}
  Missing documentation flag: {missing_documentation_flag}
  Eligibility verified:  {eligibility_verified}
  Service month:         {service_month}

MODEL OUTPUT:
  Predicted denial probability: {denial_probability:.0%}
  Risk drivers identified: {top_risk_factors}

STRICT RULES — You must write EXACTLY three sentences. Vary your phrasing naturally so the output reads like a human analyst wrote it, rather than a copy-pasted template.

  1. SENTENCE 1 (The Context & Why): State the Claim ID, the predicted risk probability, and summarize the situation. 
     - IF THE CLAIM HAS RISK DRIVERS: State that the claim is at risk and naturally list the drivers provided.
     - IF NO RISK DRIVERS ("no major administrative gaps identified"): State that the claim has a low risk score and appears administratively clean. Do not invent risk factors.
     
  2. SENTENCE 2 (The Action): Provide exactly ONE specific recommended action.
     - IF THE CLAIM HAS RISK DRIVERS: Tell the user exactly what to fix based on the drivers (e.g., "As per policy, please ensure the missing referral is attached before...").
     - IF NO RISK DRIVERS: Advise them to proceed with standard submission workflows.
     
  3. SENTENCE 3 (The Disclaimer): You MUST use this exact text for legal/compliance reasons: "Please note this is a risk estimate based on historical claim patterns, not a guarantee of payment or denial."
  
  4. NO HALLUCINATIONS: Use ONLY the provided field values and risk drivers. Do not invent clinical details.
  5. FORMAT: Output only the three sentences in a single paragraph. No introductory filler, no bullet points.

Write the explanation now:
""")


def setup_llm(model_name: str = groq_llm_model, temperature: float = 0.2, groq_api_key: str = groq_api_key):

    os.environ["GROQ_API_KEY"] = groq_api_key
    MODEL_NAME   = model_name
    TEMPERATURE  = temperature

    llm = ChatGroq(model=MODEL_NAME, temperature=TEMPERATURE)

    parser = StrOutputParser()
    chain  = EXPLANATION_PROMPT | llm | parser

    return llm, parser, chain


def extract_risk_drivers(row: pd.Series) -> list:
    drivers = []

    if row["prior_auth_required"] == 1 and row["has_prior_auth"] == 0:
        drivers.append("prior authorization required but not on file")
    if row["referral_required"] == 1 and row["referral_present"] == 0:
        drivers.append("referral required but not present")
    if row["eligibility_verified"] == 0:
        drivers.append("patient eligibility not verified")
    if row["is_in_network"] == 0:
        drivers.append("provider out of network for this payer")
    if row["missing_documentation_flag"] == 1:
        drivers.append("required documentation appears incomplete or missing")
    if row["days_to_submit"] > 30:
        drivers.append(f"submitted late ({int(row['days_to_submit'])} days after service)")

    ratio = row["total_billed"] / max(row["expected_payment"], 1)
    if ratio > 2.0:
        drivers.append(f"billed amount is {ratio:.1f}x the expected payment")

    if not drivers:
        drivers.append("no major administrative gaps identified")
    return drivers


def assign_risk_tier(prob):
    if prob >= 0.75:
        return "High Risk"
    elif prob >= 0.50:
        return "Moderate Risk"
    else:
        return "Low Risk"


def generate_explanation(row: pd.Series, chain, retries: int = 3) -> str:
    payload = row.to_dict()
    for attempt in range(retries):
        try:
            return chain.invoke(payload).strip()
        except Exception as e:
            logger.info(f"  [retry {attempt+1}] {row['claim_id']}: {e}")
            time.sleep(2)
    return "ERROR: explanation generation failed after retries."


def build_top_bottom_frames(current_df):
    top10_df = current_df.head(10)
    bottom10_df = current_df.tail(10)
    return top10_df, bottom10_df


def generate_top10_explanations(top10_df, chain, out_path=top_10_path):

    top10_df["risk_tier"] = top10_df["denial_probability"].apply(assign_risk_tier)
    top10_df["risk_drivers"] = top10_df.apply(extract_risk_drivers, axis=1)
    top10_df["top_risk_factors"] = top10_df["risk_drivers"].apply(lambda x: "; ".join(x))

    logger.info(" Generating explanations for top 10 highest-risk claims ")
    explanations = []
    for _, row in top10_df.iterrows():
        exp = generate_explanation(row, chain)
        explanations.append(exp)
        logger.info(f"\n{row['claim_id']}  (risk={row['denial_probability']:.0%})")
        logger.info(f"  Drivers: {row['top_risk_factors']}")
        logger.info(f"  → {exp}")

    top10_df["explanation"] = explanations
    top10_df[["claim_id", "denial_probability","risk_tier","predicted_denial", "top_risk_factors", "explanation"]].to_csv(
        out_path, index=False
    )
    logger.info(f"Saved : {out_path}")

    return top10_df


def generate_bottom10_explanations(bottom10_df, chain, out_path=bottom_10_path):

    bottom10_df["risk_drivers"] = bottom10_df.apply(extract_risk_drivers, axis=1)
    bottom10_df["top_risk_factors"] = bottom10_df["risk_drivers"].apply(lambda x: "; ".join(x))
    bottom10_df["risk_tier"] = bottom10_df["denial_probability"].apply(assign_risk_tier)

    logger.info("#Generating explanations for bottom 10 low-risk claims")
    explanations = []
    for _, row in bottom10_df.iterrows():
        exp = generate_explanation(row, chain)
        explanations.append(exp)
        print(f"\n{row['claim_id']}  (risk={row['denial_probability']:.0%})")
        print(f"  Drivers: {row['top_risk_factors']}")
        print(f"  {exp}")

    bottom10_df["explanation"] = explanations
    bottom10_df[["claim_id", "denial_probability","risk_tier","predicted_denial", "top_risk_factors", "explanation"]].to_csv(
        out_path, index=False
    )
    logger.info(f"Saved : {out_path}")

    return bottom10_df
