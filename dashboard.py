import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Fintech Fraud Detection AI", layout="wide")

st.title("💳 AI Fraud Detection Dashboard (Fintech Level)")

model = joblib.load("model/fraud_xgb.pkl")
scaler = joblib.load("model/scaler.pkl")

st.sidebar.header("Enter Transaction Details")

time = st.sidebar.number_input("Time", min_value=0.0, max_value=200000.0, value=1000.0)
amount = st.sidebar.number_input("Amount",min_value=0.0, max_value=10000.0, value=5000.0)

features = []
for i in range(1,29):
    val = st.sidebar.number_input(
        f"V{i}",
        min_value=-50.0,
        max_value=50.0,
        value=0.0
    )
    features.append(val)

if st.sidebar.button("Detect Fraud"):
    data = [time] + features + [amount]
    data = np.array(data).reshape(1,-1)

# Extract Time and Amount properly
    time_amount = data[:, [0, -1]]
    scaled_time_amount = scaler.transform(time_amount)

    data[:, 0] = scaled_time_amount[:, 0]
    data[:, -1] = scaled_time_amount[:, 1]

    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]

    if pred == 1:
        st.error(f"🚨 FRAUD DETECTED! Risk Score: {prob:.2f}")
    else:
        st.success(f"✅ Genuine Transaction. Risk Score: {prob:.2f}")