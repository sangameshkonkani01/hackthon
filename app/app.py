# app/app.py
import streamlit as st
import pickle
import numpy as np
import os

# ---------- CONFIG ----------
# Correct __file__ usage
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# Features for Diabetes (as your model is trained for this)
DIABETES_FEATURES = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
                     'Insulin','BMI','DiabetesPedigreeFunction','Age']

# Risk thresholds
def prob_to_risk(prob):
    if prob < 0.4:
        return "Low"
    elif prob < 0.7:
        return "Medium"
    else:
        return "High"

# Health tips for display
HEALTH_TIPS = {
    'diabetes': {
        'Low': "Keep a balanced diet and regular exercise. Monitor glucose periodically.",
        'Medium': "Consult a doctor; improve diet and exercise. Consider blood tests.",
        'High': "High risk — see a healthcare provider immediately for tests and treatment."
    }
}

# ---------- UTILITIES ----------
@st.cache_resource
def load_pickle(filename):
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

def prepare_input_dict(feature_list, values_dict):
    row = []
    for f in feature_list:
        val = values_dict.get(f)
        row.append(val)
    return np.array(row).reshape(1, -1)

def predict_with_model(model, scaler, feature_list, values_dict):
    X = prepare_input_dict(feature_list, values_dict)
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception:
            pass
    proba = None
    try:
        proba = model.predict_proba(X)[0][1]
    except Exception:
        pred = model.predict(X)[0]
        proba = float(pred)
    risk = prob_to_risk(proba)
    return risk, proba

# ---------- LOAD MODEL & SCALER ----------
diabetes_model = load_pickle("diabetes_model.pkl")
diabetes_scaler = load_pickle("diabetes_scaler.pkl")

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Smart Health Risk Predictor", layout="centered")
st.markdown("<h1 style='text-align: center; color: green;'>Smart Health Risk Predictor 🩺</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predicts risk for Diabetes. For demo only — not a medical diagnosis.</p>", unsafe_allow_html=True)
st.markdown("---")

tabs = st.tabs(["Diabetes"])  # Add Heart, Liver later

# --- DIABETES TAB ---
with tabs[0]:
    st.header("Diabetes Risk")
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
        glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=120)
        bp = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
        skin = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
    with col2:
        insulin = st.number_input("Insulin (mu U/ml)", min_value=0.0, max_value=900.0, value=80.0)
        bmi = st.number_input("BMI", min_value=0.0, max_value=80.0, value=25.0, format="%.1f")
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=5.0, value=0.5)
        age = st.number_input("Age", min_value=1, max_value=120, value=30)

    if st.button("Predict Diabetes Risk"):
        if diabetes_model is None:
            st.error("Diabetes model not found. Ask ML engineer to add models/diabetes_model.pkl")
        else:
            vals = dict(zip(DIABETES_FEATURES, [pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]))
            risk, proba = predict_with_model(diabetes_model, diabetes_scaler, DIABETES_FEATURES, vals)
            if risk == "High":
                st.error(f"Risk: {risk} — probability: {proba:.2f}")
            elif risk == "Medium":
                st.warning(f"Risk: {risk} — probability: {proba:.2f}")
            else:
                st.success(f"Risk: {risk} — probability: {proba:.2f}")
            st.info(HEALTH_TIPS['diabetes'][risk])

# Footer
st.markdown("---")
st.caption("This is a demo tool for educational purposes. Not a medical device.")
