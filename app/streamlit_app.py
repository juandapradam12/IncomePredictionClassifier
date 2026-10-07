"""Streamlit demo for the Adult Income Prediction Classifier."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from income_classifier.data import FEATURE_COLUMNS, load_adult_official_split, split_features_target
from income_classifier.interpret import positive_proba
from income_classifier.pipeline import make_model_pipeline

MODEL_PATH = ROOT / "artifacts" / "best_model.joblib"

WORKCLASS = [
    "Private",
    "Self-emp-not-inc",
    "Self-emp-inc",
    "Federal-gov",
    "Local-gov",
    "State-gov",
    "Without-pay",
    "Never-worked",
]
MARITAL = [
    "Married-civ-spouse",
    "Divorced",
    "Never-married",
    "Separated",
    "Widowed",
    "Married-spouse-absent",
    "Married-AF-spouse",
]
OCCUPATION = [
    "Tech-support",
    "Craft-repair",
    "Other-service",
    "Sales",
    "Exec-managerial",
    "Prof-specialty",
    "Handlers-cleaners",
    "Machine-op-inspct",
    "Adm-clerical",
    "Farming-fishing",
    "Transport-moving",
    "Priv-house-serv",
    "Protective-serv",
    "Armed-Forces",
]
RELATIONSHIP = [
    "Wife",
    "Own-child",
    "Husband",
    "Not-in-family",
    "Other-relative",
    "Unmarried",
]
RACE = [
    "White",
    "Asian-Pac-Islander",
    "Amer-Indian-Eskimo",
    "Other",
    "Black",
]
SEX = ["Female", "Male"]
COUNTRY = [
    "United-States",
    "Mexico",
    "Philippines",
    "Germany",
    "Canada",
    "India",
    "England",
    "China",
    "Cuba",
    "Other",
]


@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    train_df, _ = load_adult_official_split()
    X, y = split_features_target(train_df)
    pipe = make_model_pipeline("hist_gradient_boosting")
    pipe.fit(X, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    return pipe


st.set_page_config(page_title="Income Prediction Classifier", layout="centered")
st.title("Income Prediction Classifier")
st.caption("UCI Adult Census — probability of earning more than $50K/year")

model = load_model()

with st.form("profile"):
    c1, c2 = st.columns(2)
    age = c1.number_input("Age", 17, 90, 39)
    education_num = c2.slider("Education years (education-num)", 1, 16, 13)
    hours = c1.slider("Hours per week", 1, 99, 40)
    capital_gain = c2.number_input("Capital gain", 0, 99999, 0)
    capital_loss = c1.number_input("Capital loss", 0, 99999, 0)
    workclass = c2.selectbox("Workclass", WORKCLASS)
    marital = c1.selectbox("Marital status", MARITAL, index=2)
    occupation = c2.selectbox("Occupation", OCCUPATION, index=8)
    relationship = c1.selectbox("Relationship", RELATIONSHIP, index=3)
    race = c2.selectbox("Race", RACE)
    sex = c1.selectbox("Sex", SEX, index=1)
    country = c2.selectbox("Native country", COUNTRY)
    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01)
    submitted = st.form_submit_button("Predict")

if submitted:
    row = pd.DataFrame(
        [
            {
                "age": age,
                "education-num": education_num,
                "capital-gain": capital_gain,
                "capital-loss": capital_loss,
                "hours-per-week": hours,
                "workclass": workclass,
                "marital-status": marital,
                "occupation": occupation,
                "relationship": relationship,
                "race": race,
                "sex": sex,
                "native-country": "United-States" if country == "Other" else country,
            }
        ]
    )[FEATURE_COLUMNS]
    proba = float(positive_proba(model, row)[0])
    label = ">50K" if proba >= threshold else "<=50K"
    st.metric("P(>50K)", f"{proba:.3f}")
    st.success(f"Prediction at threshold {threshold:.2f}: **{label}**")
    st.progress(min(max(proba, 0.0), 1.0))

st.markdown(
    "Educational demo only. Adult includes protected attributes; "
    "see `docs/MODEL_CARD.md` and fairness slice results before any real-world use."
)
