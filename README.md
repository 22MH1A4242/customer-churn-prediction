# 📉 Churn Prediction with SHAP Explainability

> **Skills demonstrated:** EDA · XGBoost · SMOTE · SHAP · Streamlit deployment · Business recommendations

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate dataset
python generate_data.py

# 3. Train the model
python train_model.py

# 4. Launch the app
streamlit run app.py
```

## Project Structure

```
project1_churn/
├── app.py              ← Streamlit web app (main entry point)
├── train_model.py      ← XGBoost training + evaluation script
├── generate_data.py    ← Creates realistic churn dataset
├── requirements.txt    ← All dependencies
├── data/
│   └── churn_data.csv  ← Generated after running generate_data.py
└── models/
    ├── churn_model.pkl         ← Saved model (after training)
    └── confusion_matrix.png   ← Saved after training
```

## What This Project Demonstrates

| Skill | Where |
|---|---|
| Data Cleaning + EDA | `train_model.py` |
| Class imbalance handling (SMOTE) | `train_model.py` |
| XGBoost model training | `train_model.py` |
| Model evaluation (ROC-AUC, F1) | `train_model.py` |
| **SHAP explainability** | `app.py` Tab 2 |
| Streamlit web app deployment | `app.py` |
| Batch predictions + CSV export | `app.py` Tab 3 |
| Business recommendations | `app.py` Tab 1 |

## Resume Bullet Points (copy these!)

- Built end-to-end churn prediction model using XGBoost + SMOTE achieving **ROC-AUC > 0.85**
- Integrated SHAP explainability to provide per-customer feature attribution, bridging ML and business strategy
- Deployed as interactive Streamlit app supporting both single-customer and batch CSV predictions
- Generated actionable retention recommendations based on customer contract, payment, and service features
