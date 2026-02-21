import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Load data
df = pd.read_csv("data/creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

# Scale
scaler = StandardScaler()
X[['Amount','Time']] = scaler.fit_transform(X[['Amount','Time']])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,stratify=y,random_state=42
)

# Handle imbalance
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print("Before SMOTE:", y_train.value_counts())
print("After SMOTE:", y_train_sm.value_counts())

# XGBoost model
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=5,
    eval_metric="logloss",
    use_label_encoder=False
)

model.fit(X_train_sm, y_train_sm)

# Prediction
y_probs = model.predict_proba(X_test)[:,1]
y_pred = model.predict(X_test)

print(classification_report(y_test,y_pred))

# Precision Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_probs)
pr_auc = auc(recall, precision)

plt.plot(recall, precision, label=f"PR AUC={pr_auc:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Fraud Detection)")
plt.legend()
plt.savefig("model/pr_curve.png")
plt.show()

# Save model
joblib.dump(model,"model/fraud_xgb.pkl")
joblib.dump(scaler,"model/scaler.pkl")

print("Model saved successfully!")