import pandas as pd
import joblib
import time

model = joblib.load("model/fraud_xgb.pkl")
scaler = joblib.load("model/scaler.pkl")

df = pd.read_csv("data/creditcard.csv").sample(50)

print("Starting real-time fraud detection...\n")

for i,row in df.iterrows():
    data = row.drop("Class").values.reshape(1,-1)

    data[:,[0,-1]] = scaler.transform(data[:,[0,-1]])

    pred = model.predict(data)[0]

    if pred == 1:
        print("🚨 Fraud Transaction Detected!")
    else:
        print("✅ Genuine transaction")

    time.sleep(1)