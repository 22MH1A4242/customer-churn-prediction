"""
generate_data.py
Run once to create data/churn_data.csv
"""
import pandas as pd
import numpy as np

np.random.seed(42)
n = 7043  # same size as Telco dataset

df = pd.DataFrame({
    "customerID":        [f"CUST-{i:05d}" for i in range(n)],
    "gender":            np.random.choice(["Male", "Female"], n),
    "SeniorCitizen":     np.random.choice([0, 1], n, p=[0.84, 0.16]),
    "Partner":           np.random.choice(["Yes", "No"], n),
    "Dependents":        np.random.choice(["Yes", "No"], n, p=[0.30, 0.70]),
    "tenure":            np.random.randint(1, 73, n),
    "PhoneService":      np.random.choice(["Yes", "No"], n, p=[0.90, 0.10]),
    "MultipleLines":     np.random.choice(["Yes", "No", "No phone service"], n, p=[0.42, 0.48, 0.10]),
    "InternetService":   np.random.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22]),
    "OnlineSecurity":    np.random.choice(["Yes", "No", "No internet service"], n, p=[0.28, 0.50, 0.22]),
    "OnlineBackup":      np.random.choice(["Yes", "No", "No internet service"], n, p=[0.34, 0.44, 0.22]),
    "DeviceProtection":  np.random.choice(["Yes", "No", "No internet service"], n, p=[0.34, 0.44, 0.22]),
    "TechSupport":       np.random.choice(["Yes", "No", "No internet service"], n, p=[0.29, 0.49, 0.22]),
    "StreamingTV":       np.random.choice(["Yes", "No", "No internet service"], n, p=[0.38, 0.40, 0.22]),
    "StreamingMovies":   np.random.choice(["Yes", "No", "No internet service"], n, p=[0.39, 0.39, 0.22]),
    "Contract":          np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24]),
    "PaperlessBilling":  np.random.choice(["Yes", "No"], n, p=[0.59, 0.41]),
    "PaymentMethod":     np.random.choice(
                             ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                             n, p=[0.34, 0.23, 0.22, 0.21]),
    "MonthlyCharges":    np.round(np.random.uniform(18.25, 118.75, n), 2),
    "TotalCharges":      np.round(np.random.uniform(18.80, 8684.80, n), 2),
})

# Make churn realistic (month-to-month + fiber → higher churn)
churn_prob = (
    0.10
    + 0.20 * (df["Contract"] == "Month-to-month")
    + 0.15 * (df["InternetService"] == "Fiber optic")
    + 0.10 * (df["SeniorCitizen"] == 1)
    - 0.10 * (df["tenure"] > 24)
    + 0.05 * (df["MonthlyCharges"] > 80)
)
churn_prob = churn_prob.clip(0.03, 0.75)
df["Churn"] = np.where(np.random.rand(n) < churn_prob, "Yes", "No")

df.to_csv("data/churn_data.csv", index=False)
print(f"Saved {len(df)} rows | Churn rate: {(df['Churn']=='Yes').mean():.1%}")
