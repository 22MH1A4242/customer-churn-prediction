"""
train_model.py
Trains XGBoost churn model and saves to models/churn_model.pkl
Run: python train_model.py
"""
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load ──────────────────────────────────────────────────────────────────
df = pd.read_csv("data/churn_data.csv")
print(f"Shape: {df.shape}  |  Churn rate: {(df['Churn']=='Yes').mean():.1%}")

# ── 2. Preprocess ─────────────────────────────────────────────────────────────
df = df.drop(columns=["customerID"])
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Encode categoricals
cat_cols = df.select_dtypes(include="object").columns.tolist()
cat_cols.remove("Churn")

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

df["Churn"] = (df["Churn"] == "Yes").astype(int)

# ── 3. Feature engineering ────────────────────────────────────────────────────
df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)
df["HighValueCustomer"] = (df["MonthlyCharges"] > df["MonthlyCharges"].median()).astype(int)

feature_cols = [c for c in df.columns if c != "Churn"]
X = df[feature_cols]
y = df["Churn"]

# ── 4. Split + SMOTE ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"After SMOTE: {dict(zip(*np.unique(y_train_res, return_counts=True)))}")

# ── 5. Train XGBoost ──────────────────────────────────────────────────────────
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1,
    eval_metric="logloss",
    random_state=42,
    use_label_encoder=False,
)
model.fit(X_train_res, y_train_res,
          eval_set=[(X_test, y_test)],
          verbose=False)

# ── 6. Evaluate ───────────────────────────────────────────────────────────────
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n── Classification Report ──")
print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
print(f"5-Fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Confusion matrix plot
os.makedirs("models", exist_ok=True)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix — XGBoost Churn Model")
plt.tight_layout()
plt.savefig("models/confusion_matrix.png", dpi=150)
plt.close()

# ── 7. Save ───────────────────────────────────────────────────────────────────
joblib.dump({"model": model, "feature_cols": feature_cols, "encoders": encoders},
            "models/churn_model.pkl")
print("\nModel saved → models/churn_model.pkl")
print("Confusion matrix saved → models/confusion_matrix.png")
