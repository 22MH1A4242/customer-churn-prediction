"""
app.py — Churn Prediction with SHAP Explainability
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: #f0f2f6; border-radius: 10px;
        padding: 1rem 1.25rem; text-align: center;
    }
    .risk-high   { color: #d32f2f; font-size: 2rem; font-weight: 700; }
    .risk-medium { color: #f57c00; font-size: 2rem; font-weight: 700; }
    .risk-low    { color: #388e3c; font-size: 2rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("models/churn_model.pkl")

try:
    artifact = load_model()
    model       = artifact["model"]
    feature_cols = artifact["feature_cols"]
    encoders    = artifact["encoders"]
    MODEL_LOADED = True
except FileNotFoundError:
    MODEL_LOADED = False

# ── Sidebar — customer inputs ──────────────────────────────────────────────────
st.sidebar.header("🧑 Customer Profile")
st.sidebar.markdown("Adjust inputs to predict churn risk.")

gender          = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior          = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner         = st.sidebar.selectbox("Has Partner", ["Yes", "No"])
dependents      = st.sidebar.selectbox("Has Dependents", ["Yes", "No"])
tenure          = st.sidebar.slider("Tenure (months)", 1, 72, 12)
phone           = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multi_lines     = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet        = st.sidebar.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
online_sec      = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_bk       = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_prot     = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_supp       = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
stream_tv       = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
stream_mov      = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract        = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless       = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment         = st.sidebar.selectbox("Payment Method",
                    ["Electronic check", "Mailed check",
                     "Bank transfer (automatic)", "Credit card (automatic)"])
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
total_charges   = st.sidebar.slider("Total Charges ($)", 18.0, 9000.0,
                                    float(monthly_charges * tenure), step=10.0)

# ── Build input DataFrame ──────────────────────────────────────────────────────
raw = pd.DataFrame([{
    "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
    "Partner": partner, "Dependents": dependents, "tenure": tenure,
    "PhoneService": phone, "MultipleLines": multi_lines, "InternetService": internet,
    "OnlineSecurity": online_sec, "OnlineBackup": online_bk,
    "DeviceProtection": device_prot, "TechSupport": tech_supp,
    "StreamingTV": stream_tv, "StreamingMovies": stream_mov, "Contract": contract,
    "PaperlessBilling": paperless, "PaymentMethod": payment,
    "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
}])

def preprocess(df_in, encoders):
    df = df_in.copy()
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col])
    df["AvgMonthlySpend"]   = df["TotalCharges"] / (df["tenure"] + 1)
    df["HighValueCustomer"] = (df["MonthlyCharges"] > 65).astype(int)
    return df[feature_cols]

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("📉 Customer Churn Prediction + SHAP Explainability")
st.markdown("Predict whether a customer will churn — and **understand exactly why** using SHAP.")

tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "🔍 SHAP Explainability", "📊 Batch Analysis"])

# ── Tab 1: Prediction ─────────────────────────────────────────────────────────
with tab1:
    if not MODEL_LOADED:
        st.error("⚠️ Model not found. Run `python train_model.py` first.")
        st.stop()

    X_input = preprocess(raw, encoders)
    prob    = model.predict_proba(X_input)[0][1]
    pred    = "Will Churn" if prob > 0.5 else "Will Stay"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size:0.85rem;color:#555;'>Churn Probability</div>
            <div class='{"risk-high" if prob>0.6 else "risk-medium" if prob>0.4 else "risk-low"}'>{prob:.1%}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size:0.85rem;color:#555;'>Prediction</div>
            <div style='font-size:1.5rem;font-weight:700;color:{"#d32f2f" if pred=="Will Churn" else "#388e3c"}'>{pred}</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        risk = "🔴 High" if prob > 0.6 else "🟠 Medium" if prob > 0.4 else "🟢 Low"
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size:0.85rem;color:#555;'>Risk Level</div>
            <div style='font-size:1.5rem;font-weight:600;'>{risk}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    # Gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(prob * 100, 1),
        title={"text": "Churn Risk Score"},
        delta={"reference": 50, "suffix": "%"},
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": "#d32f2f" if prob > 0.6 else "#f57c00" if prob > 0.4 else "#388e3c"},
            "steps": [
                {"range": [0, 40],   "color": "#e8f5e9"},
                {"range": [40, 60],  "color": "#fff3e0"},
                {"range": [60, 100], "color": "#ffebee"},
            ],
            "threshold": {"line": {"color": "black", "width": 3}, "thickness": 0.75, "value": 50},
        }
    ))
    fig_gauge.update_layout(height=300)
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Recommendations
    st.subheader("💡 Recommended Actions")
    recs = []
    if contract == "Month-to-month":
        recs.append("🔄 Offer a discounted **annual contract** to improve retention.")
    if internet == "Fiber optic" and tech_supp == "No":
        recs.append("🛠 Offer **free Tech Support** for 3 months — fiber customers churn when service issues go unresolved.")
    if tenure < 12:
        recs.append("🎁 Enrol in **early loyalty program** — customers in first year are highest risk.")
    if payment == "Electronic check":
        recs.append("💳 Encourage switch to **auto-pay** — reduces churn by ~15%.")
    if not recs:
        recs.append("✅ Customer profile looks healthy. Continue standard engagement.")
    for r in recs:
        st.markdown(f"- {r}")

# ── Tab 2: SHAP ───────────────────────────────────────────────────────────────
with tab2:
    st.subheader("🔍 Why did the model make this prediction?")
    st.markdown("SHAP (SHapley Additive exPlanations) shows how each feature **pushed the prediction** higher or lower.")

    X_input = preprocess(raw, encoders)
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_input)

    # Waterfall-style bar chart
    sv = shap_vals[0]
    feat_names = feature_cols
    sorted_idx = np.argsort(np.abs(sv))[-12:]  # top 12

    colors = ["#d32f2f" if v > 0 else "#388e3c" for v in sv[sorted_idx]]
    fig_shap, ax = plt.subplots(figsize=(8, 5))
    ax.barh([feat_names[i] for i in sorted_idx], sv[sorted_idx], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP Value (impact on churn probability)")
    ax.set_title("Feature Impact — Red = increases churn, Green = reduces churn")
    plt.tight_layout()
    st.pyplot(fig_shap)

    st.markdown("""
    **How to read this:**
    - 🔴 **Red bars** → feature increases churn probability
    - 🟢 **Green bars** → feature decreases churn probability
    - Longer bar = stronger impact
    """)

    # Feature importance (global)
    st.subheader("📊 Global Feature Importance")
    importances = model.feature_importances_
    fi_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances})
    fi_df = fi_df.sort_values("Importance", ascending=False).head(10)
    fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                    color="Importance", color_continuous_scale="Blues",
                    title="Top 10 Most Important Features (XGBoost)")
    fig_fi.update_layout(yaxis={"categoryorder": "total ascending"}, height=400)
    st.plotly_chart(fig_fi, use_container_width=True)

# ── Tab 3: Batch ──────────────────────────────────────────────────────────────
with tab3:
    st.subheader("📂 Upload a CSV for Batch Prediction")
    st.markdown("Upload a CSV with the same columns as the training data (without 'Churn' column).")
    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])

    if uploaded:
        batch_df = pd.read_csv(uploaded)
        if "customerID" in batch_df.columns:
            ids = batch_df["customerID"]
            batch_df = batch_df.drop(columns=["customerID"])
        else:
            ids = pd.Series([f"C{i}" for i in range(len(batch_df))])

        try:
            X_batch = preprocess(batch_df, encoders)
            probs   = model.predict_proba(X_batch)[:, 1]
            preds   = ["Churn" if p > 0.5 else "Stay" for p in probs]
            result  = pd.DataFrame({
                "CustomerID": ids, "Churn Probability": [f"{p:.1%}" for p in probs],
                "Prediction": preds,
                "Risk": ["🔴 High" if p > 0.6 else "🟠 Medium" if p > 0.4 else "🟢 Low" for p in probs]
            })
            st.dataframe(result, use_container_width=True)
            st.download_button("⬇️ Download Predictions", result.to_csv(index=False),
                               "churn_predictions.csv", "text/csv")

            # Summary pie
            counts = pd.Series(preds).value_counts()
            fig_pie = px.pie(values=counts.values, names=counts.index,
                             color=counts.index, color_discrete_map={"Churn": "#d32f2f", "Stay": "#388e3c"},
                             title="Batch Churn Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.info("👆 Upload a CSV to get batch predictions with churn probabilities for each customer.")
        # Show sample CSV format
        sample = raw.drop(columns=[], errors="ignore")
        st.markdown("**Expected CSV format (first row sample):**")
        st.dataframe(raw, use_container_width=True)
