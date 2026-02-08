import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier 
import shap
import matplotlib.pyplot as plt
import numpy as np 

# --- 1. Feature Name Mapping Definitions ---
FEATURE_MAPPINGS = {
    'age': 'Age (Years)',
    'sex': 'Gender (0=Female, 1=Male)',
    'cp': 'Chest Pain Type (Level 0-3)',
    'trestbps': 'Resting Blood Pressure (mm Hg)',
    'chol': 'Serum Cholesterol (mg/dL)',
    'fbs': 'Fasting Blood Sugar > 120 (High Risk Y/N)',
    'restecg': 'Resting ECG Results',
    'thalach': 'Maximum Heart Rate Achieved',
    'exang': 'Exercise Induced Angina (Y/N)',
    'oldpeak': 'Exercise ST Depression',
    'slope': 'Slope of Peak Exercise ST',
    'ca': 'Major Vessels Colored (0-3)',
    'thal': 'Thalassemia Type (1, 2, or 3)'
}

# --- 2. Load Model ---
model = CatBoostClassifier()
try:
    # Load the CatBoost model
    model.load_model("heart_disease_catboost.cbm")
except Exception as e:
    st.error(f"Error loading CatBoost model: {e}")
    st.info("Please ensure you run 'python train_model.py' first to create 'heart_disease_catboost.cbm'.")
    st.stop()


# --- 3. Streamlit Page Config ---
st.set_page_config(
    page_title="Heart Disease Prediction (CatBoost)",
    page_icon="💓",
    layout="wide"
)

st.title("💓 Heart Disease Prediction (CatBoost) with Explainable AI")

# --- 4. Sidebar for User Input and View Selector ---
st.sidebar.header("Patient Information")

# Dynamic View Selector
view_choice = st.sidebar.radio(
    "Select Explanation View",
    ('Patient View', 'Doctor View'),
    index=0
)
st.sidebar.markdown("---")


def get_user_input():
    # Variables based on the UCI Heart Disease dataset
    age = st.sidebar.slider("Age", 20, 80, 50)
    trestbps = st.sidebar.slider("Resting Blood Pressure", 80, 200, 120)
    chol = st.sidebar.slider("Serum Cholesterol", 100, 400, 200)
    thalach = st.sidebar.slider("Max Heart Rate Achieved", 60, 220, 150)
    oldpeak = st.sidebar.slider("ST depression induced by exercise", 0.0, 10.0, 1.0)
    ca = st.sidebar.slider("Number of major vessels (0-3)", 0, 3, 0)

    # Categorical variables
    sex = st.sidebar.selectbox("Sex (0=F, 1=M)", [0, 1])
    cp = st.sidebar.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (0/1)", [0, 1])
    restecg = st.sidebar.selectbox("Resting ECG (0, 1, 2)", [0, 1, 2])
    exang = st.sidebar.selectbox("Exercise Induced Angina (0/1)", [0, 1])
    slope = st.sidebar.selectbox("Slope of ST segment (0, 1, 2)", [0, 1, 2])
    thal = st.sidebar.selectbox("Thalassemia (1, 2, 3)", [1, 2, 3])

    # Create DataFrame matching the model's feature names
    data = pd.DataFrame({
        'age': [age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps],
        'chol': [chol], 'fbs': [fbs], 'restecg': [restecg], 'thalach': [thalach],
        'exang': [exang], 'oldpeak': [oldpeak], 'slope': [slope],
        'ca': [ca], 'thal': [thal]
    })

    return data

# --- 5. Get Input Data and Prediction ---
input_df = get_user_input()

# Calculate probability for the positive class (index 1)
prediction_proba = model.predict_proba(input_df)[0][1]

# --- IMPLEMENTATION OF THREE-TIER RISK LEVELS ---
if prediction_proba >= 0.66:
    risk_level = "🚨 High Risk"
    risk_color = "red"
elif prediction_proba >= 0.33:
    risk_level = "⚠️ Moderate Risk"
    risk_color = "orange"
else:
    risk_level = "✅ Low Risk"
    risk_color = "green"

st.subheader("Prediction")
# Display risk level with color and bold font
st.markdown(f"<p style='font-size: 28px; font-weight: bold; color: {risk_color};'>{risk_level}</p>", unsafe_allow_html=True)
st.write(f"Probability of Heart Disease: {prediction_proba:.2f}")


# --- 6. SHAP Explanation ---
st.subheader(f"Feature Contribution ({view_choice} SHAP)")

explainer = shap.TreeExplainer(model)
shap_values_obj = explainer(input_df) 

# --- Apply feature names dynamically based on view choice (Fixes View Synchronization) ---
if view_choice == 'Patient View':
    # Use human-readable names
    display_feature_names = [FEATURE_MAPPINGS[col] for col in input_df.columns]
else:
    # Use original, technical names (Doctor View)
    display_feature_names = list(input_df.columns)

# Assign the selected names to the SHAP object before plotting
shap_values_obj.feature_names = display_feature_names


# --- CRITICAL INDEXING LOGIC (Handles 2D/3D output and fixed previous errors) ---
if len(shap_values_obj.values.shape) == 3:
    # If 3D, select the positive class (index 1) from the first sample (index 0)
    shap_data = shap_values_obj[0, :, 1]
else:
    # If 2D, just take the first sample (index 0)
    shap_data = shap_values_obj[0]


plt.figure(figsize=(10,5))
# Removed 'base_value' argument to fix the TypeError
shap.plots.waterfall(
    shap_data, 
    max_display=13, 
    show=True
)

st.pyplot(plt.gcf())
st.caption("Features listed above the dotted line push the prediction towards High Risk; features below push it towards Low Risk.")


# --- 7. Optional Styling ---
st.markdown("""
<style>
.main {background-color: #f5f7ff;}
.stButton > button {
    background-color: #ff4b4b;
    color: white;
    font-weight: bold;
    border-radius: 10px;
    padding: 0.5rem 2rem;
    border: none;
    width: 100%;
}
.stButton > button:hover {background-color: #ff3333;}
</style>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<hr>
<div style='text-align:center; color:gray'>
Made with ❤ using Streamlit · © 2025
</div>
""", unsafe_allow_html=True)