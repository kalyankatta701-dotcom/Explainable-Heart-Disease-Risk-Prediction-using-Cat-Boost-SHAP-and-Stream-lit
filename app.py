"""
Heart Disease Prediction Web Application
This Streamlit app provides an interactive interface for heart disease prediction
using CatBoost and SHAP explainability.
"""

import streamlit as st
import pandas as pd
import pickle
import os
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained CatBoost model and feature names."""
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'catboost_heart_model.pkl')
    feature_names_path = os.path.join(os.path.dirname(__file__), 'models', 'feature_names.pkl')
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(feature_names_path, 'rb') as f:
        feature_names = pickle.load(f)
    
    return model, feature_names

def get_feature_descriptions():
    """Return descriptions for each feature."""
    return {
        'age': 'Age in years',
        'sex': 'Sex (1 = male, 0 = female)',
        'cp': 'Chest pain type (0-3)',
        'trestbps': 'Resting blood pressure (mm Hg)',
        'chol': 'Serum cholesterol (mg/dl)',
        'fbs': 'Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)',
        'restecg': 'Resting ECG results (0-2)',
        'thalach': 'Maximum heart rate achieved',
        'exang': 'Exercise induced angina (1 = yes, 0 = no)',
        'oldpeak': 'ST depression induced by exercise',
        'slope': 'Slope of peak exercise ST segment (0-2)',
        'ca': 'Number of major vessels colored by fluoroscopy (0-3)',
        'thal': 'Thalassemia (0-3)'
    }

def create_input_form():
    """Create the input form for user data."""
    st.sidebar.header("Patient Information")
    
    # Create two columns for better organization
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=20, max_value=100, value=50, step=1)
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], 
                         format_func=lambda x: ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'][x])
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120, step=1)
        chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200, step=1)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], 
                          format_func=lambda x: "No" if x == 0 else "Yes")
    
    with col2:
        restecg = st.selectbox("Resting ECG", options=[0, 1, 2],
                              format_func=lambda x: ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'][x])
        thalach = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150, step=1)
        exang = st.selectbox("Exercise Induced Angina", options=[0, 1],
                            format_func=lambda x: "No" if x == 0 else "Yes")
        oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slope = st.selectbox("Slope of Peak Exercise ST", options=[0, 1, 2],
                           format_func=lambda x: ['Upsloping', 'Flat', 'Downsloping'][x])
        ca = st.selectbox("Number of Major Vessels", options=[0, 1, 2, 3])
        thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3],
                          format_func=lambda x: ['Normal', 'Fixed Defect', 'Reversible Defect', 'Unknown'][x])
    
    # Create feature dictionary
    features = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    
    return features

def plot_shap_waterfall(shap_values, features, feature_names):
    """Create SHAP waterfall plot."""
    fig = plt.figure(figsize=(10, 8))
    shap.waterfall_plot(shap_values[0], max_display=13)
    st.pyplot(fig, use_container_width=True)
    plt.close()

def plot_shap_force(shap_values, expected_value, features, feature_names):
    """Create SHAP force plot."""
    fig = plt.figure(figsize=(20, 3))
    shap.force_plot(
        expected_value,
        shap_values[0].values,
        features,
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
    st.pyplot(fig, use_container_width=True)
    plt.close()

def main():
    """Main application function."""
    
    # Title and description
    st.title("❤️ Heart Disease Risk Prediction")
    st.markdown("""
    This application uses **CatBoost** machine learning algorithm to predict the risk of heart disease 
    based on various medical parameters. The predictions are made explainable using **SHAP** 
    (SHapley Additive exPlanations) values.
    """)
    
    # Load model
    try:
        model, feature_names = load_model()
    except FileNotFoundError:
        st.error("⚠️ Model not found! Please train the model first by running: `python train_model.py`")
        st.stop()
    
    # Get user input
    features = create_input_form()
    
    # Create DataFrame from features
    input_df = pd.DataFrame([features])
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Input Summary")
        st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)
    
    with col2:
        st.subheader("ℹ️ Feature Descriptions")
        descriptions = get_feature_descriptions()
        desc_df = pd.DataFrame(list(descriptions.items()), columns=['Feature', 'Description'])
        st.dataframe(desc_df, use_container_width=True, hide_index=True)
    
    # Predict button
    if st.sidebar.button("🔮 Predict Heart Disease Risk", type="primary"):
        with st.spinner("Analyzing patient data..."):
            # Make prediction
            prediction = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)[0]
            
            # Display prediction
            st.markdown("---")
            st.header("🎯 Prediction Results")
            
            # Create two columns for results
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                if prediction == 1:
                    st.error("### ⚠️ High Risk of Heart Disease")
                    st.markdown(f"""
                    <div style='background-color: #ffebee; padding: 20px; border-radius: 10px; border-left: 5px solid #f44336;'>
                        <h3 style='color: #c62828;'>Risk Level: HIGH</h3>
                        <p style='font-size: 18px;'>The model predicts a <b>high risk</b> of heart disease.</p>
                        <p style='color: #666;'>Please consult with a healthcare professional for proper evaluation.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.success("### ✅ Low Risk of Heart Disease")
                    st.markdown(f"""
                    <div style='background-color: #e8f5e9; padding: 20px; border-radius: 10px; border-left: 5px solid #4caf50;'>
                        <h3 style='color: #2e7d32;'>Risk Level: LOW</h3>
                        <p style='font-size: 18px;'>The model predicts a <b>low risk</b> of heart disease.</p>
                        <p style='color: #666;'>Continue maintaining a healthy lifestyle.</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with res_col2:
                st.subheader("📈 Prediction Confidence")
                
                # Create gauge chart for probability
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prediction_proba[1] * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Risk Probability (%)"},
                    delta={'reference': 50},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkred" if prediction_proba[1] > 0.5 else "green"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                st.metric("No Disease Probability", f"{prediction_proba[0]:.2%}")
                st.metric("Disease Probability", f"{prediction_proba[1]:.2%}")
            
            # SHAP Explainability
            st.markdown("---")
            st.header("🔍 Model Explainability with SHAP")
            st.markdown("""
            SHAP (SHapley Additive exPlanations) values show how each feature contributes to the prediction.
            - **Red** indicates features pushing towards heart disease
            - **Blue** indicates features pushing away from heart disease
            """)
            
            # Calculate SHAP values
            with st.spinner("Computing SHAP values..."):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(input_df)
            
            # Create tabs for different SHAP visualizations
            tab1, tab2, tab3 = st.tabs(["📊 Waterfall Plot", "🔥 Force Plot", "📋 Feature Impact"])
            
            with tab1:
                st.subheader("SHAP Waterfall Plot")
                st.markdown("Shows how each feature contributes to pushing the prediction from the base value.")
                plot_shap_waterfall(shap_values, features, feature_names)
            
            with tab2:
                st.subheader("SHAP Force Plot")
                st.markdown("Visualizes the contribution of each feature as forces pushing the prediction.")
                plot_shap_force(shap_values, explainer.expected_value, input_df, feature_names)
            
            with tab3:
                st.subheader("Feature Impact Table")
                shap_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Value': input_df.values[0],
                    'SHAP Value': shap_values[0].values,
                    'Impact': ['Increases Risk' if x > 0 else 'Decreases Risk' for x in shap_values[0].values]
                })
                shap_df['Abs SHAP Value'] = abs(shap_df['SHAP Value'])
                shap_df = shap_df.sort_values('Abs SHAP Value', ascending=False)
                st.dataframe(
                    shap_df[['Feature', 'Value', 'SHAP Value', 'Impact']].style.background_gradient(
                        subset=['SHAP Value'], cmap='RdYlGn_r'
                    ),
                    use_container_width=True,
                    hide_index=True
                )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><b>Disclaimer:</b> This tool is for educational purposes only and should not be used as a substitute 
        for professional medical advice, diagnosis, or treatment.</p>
        <p>Built with ❤️ using CatBoost, SHAP, and Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
