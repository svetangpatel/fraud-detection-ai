from flask import Flask,request,jsonify
import joblib
import numpy as np
from email_alert  import send_fraud_alert

app = Flask(__name__)

model = joblib.load("model/fraud_xgb.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route("/")
def home():
    return "Fraud Detection API Running"

@app.route("/predict",methods=["POST"])
def predict():
    data = request.json["features"]
    data = np.array(data).reshape(1,-1)

    data[:,[0,-1]] = scaler.transform(data[:,[0,-1]])

    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]

    if pred==1:
        send_fraud_alert()
        result="Fraud"
    else:
        result="Genuine"

    return jsonify({"prediction":result,"risk_score":float(prob)})

 