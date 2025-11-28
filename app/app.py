import streamlit as st
import pickle
import numpy as np
import os

# ---------- CONFIG ----------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# Features for Diabetes (as your model is trained for this)
DIABETES_FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# Risk thresholds
def prob_to_risk(prob: float) -> str:
    if prob < 0.4:
        return "Low"
    elif prob < 0.7:
        return "Medium"
    else:
        return "High"


# Health tips for display
HEALTH_TIPS = {
    "diabetes": {
        "Low": (
            "Your estimated risk is low. Keep a balanced diet, "
            "stay active, and continue regular health check-ups."
        ),
        "Medium": (
            "Your estimated risk is moderate. Consider improving your diet, "
            "increasing physical activity, and discussing this result with a doctor."
        ),
        "High": (
            "Your estimated risk is high. Please consult a healthcare professional "
            "soon for proper tests, advice, and possible treatment."
        ),
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

    # Scale features if scaler is available
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception:
            pass

    # Get probability if possible, otherwise fall back to prediction
    try:
        proba = float(model.predict_proba(X)[0][1])
    except Exception:
        pred = float(model.predict(X)[0])
        proba = pred

    risk = prob_to_risk(proba)
    return risk, proba


# ---------- LOAD MODEL & SCALER ----------
diabetes_model = load_pickle("diabetes_model.pkl")
diabetes_scaler = load_pickle("diabetes_scaler.pkl")

# ---------- STREAMLIT UI ----------
st.set_page_config(
    page_title="Smart Health Risk Predictor",
    layout="centered",
)

st.markdown(
    "<h1 style='text-align: center; color: green;'>Smart Health Risk Predictor 🩺</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center;'>"
    "This tool estimates your diabetes risk using a machine-learning model. "
    "<br>It is for educational use only and is <b>not</b> a medical diagnosis."
    "</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

tabs = st.tabs(["Diabetes"])

# --- DIABETES TAB ---
with tabs[0]:
    st.header("Diabetes Risk Assessment")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input(
            "Number of pregnancies",
            min_value=0,
            max_value=20,
            value=0,
        )
        glucose = st.number_input(
            "Glucose level (mg/dL)",
            min_value=0,
            max_value=300,
            value=120,
        )
        bp = st.number_input(
            "Blood pressure (mm Hg)",
            min_value=0,
            max_value=250,
            value=70,
        )
        skin = st.number_input(
            "Skin thickness (mm)",
            min_value=0,
            max_value=100,
            value=20,
        )

    with col2:
        insulin = st.number_input(
            "Insulin (mu U/ml)",
            min_value=0.0,
            max_value=900.0,
            value=80.0,
        )
        bmi = st.number_input(
            "Body Mass Index (BMI)",
            min_value=0.0,
            max_value=80.0,
            value=25.0,
            format="%.1f",
        )
        dpf = st.number_input(
            "Family history score (Diabetes Pedigree Function)",
            min_value=0.0,
            max_value=5.0,
            value=0.5,
        )
        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=120,
            value=30,
        )

    if st.button("Predict diabetes risk", type="primary"):
        if diabetes_model is None:
            st.error(
                "Sorry, the diabetes prediction model is not available right now. "
                "Please try again later."
            )
        else:
            vals = dict(
                zip(
                    DIABETES_FEATURES,
                    [pregnancies, glucose, bp, skin, insulin, bmi, dpf, age],
                )
            )
            risk, proba = predict_with_model(
                diabetes_model,
                diabetes_scaler,
                DIABETES_FEATURES,
                vals,
            )

            percent = proba * 100.0
            message = f"Estimated risk level: **{risk}** (about {percent:.0f}% chance)"

            if risk == "High":
                st.error(message)
            elif risk == "Medium":
                st.warning(message)
            else:
                st.success(message)

            st.info(HEALTH_TIPS["diabetes"][risk])

# Footer
st.markdown("---")
st.caption(
    "This is a demo tool for educational purposes and does not replace professional medical advice."
)

# ---------- OPTIONAL: HIDE STREAMLIT DEFAULT MENU & FOOTER ----------
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
