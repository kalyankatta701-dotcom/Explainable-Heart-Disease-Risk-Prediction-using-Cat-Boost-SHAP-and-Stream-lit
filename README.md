# Explainable Heart Disease Risk Prediction

## 🎯 Project Overview

Heart disease still affects millions of people annually, and forecasting who may be vulnerable can contribute to meaningful differences in practice and outcomes for patients. This project addresses this problem in a highly advanced, user-friendly fashion using:

- **CatBoost**: High-performance gradient boosting algorithm for accurate predictions
- **SHAP**: SHapley Additive exPlanations for model interpretability
- **Streamlit**: Interactive web interface for real-time predictions

## ✨ Features

- 🔮 **Accurate Predictions**: Utilizes CatBoost algorithm for high-accuracy heart disease risk assessment
- 📊 **Interactive Dashboard**: User-friendly Streamlit interface for easy data input
- 🔍 **Model Explainability**: SHAP values provide transparent insights into prediction reasoning
- 📈 **Visual Analytics**: Multiple visualization types (waterfall plots, force plots, gauge charts)
- 🎨 **Professional UI**: Clean and intuitive design with real-time feedback

## 📋 Dataset

The project uses a heart disease dataset with the following features:

| Feature | Description |
|---------|-------------|
| age | Age in years |
| sex | Sex (1 = male, 0 = female) |
| cp | Chest pain type (0-3) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl |
| restecg | Resting ECG results (0-2) |
| thalach | Maximum heart rate achieved |
| exang | Exercise induced angina |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels (0-3) |
| thal | Thalassemia (0-3) |

**Target**: Binary classification (0 = No Disease, 1 = Disease)

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/kalyankatta701-dotcom/Explainable-Heart-Disease-Risk-Prediction-using-Cat-Boost-SHAP-and-Stream-lit.git
cd Explainable-Heart-Disease-Risk-Prediction-using-Cat-Boost-SHAP-and-Stream-lit
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Train the model**
```bash
python train_model.py
```

This will:
- Load the heart disease dataset
- Train a CatBoost classifier
- Save the trained model to `models/catboost_heart_model.pkl`
- Display training metrics and accuracy

4. **Run the Streamlit application**
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 💻 Usage

### Training the Model

The `train_model.py` script trains a CatBoost classifier with the following configuration:
- Iterations: 500
- Learning rate: 0.05
- Depth: 6
- Loss function: Logloss
- Train/Test split: 80/20

### Using the Web Application

1. **Input Patient Data**: Enter medical parameters in the sidebar
2. **Click Predict**: Press the "Predict Heart Disease Risk" button
3. **View Results**: See the prediction along with confidence scores
4. **Explore Explanations**: Analyze SHAP visualizations to understand the prediction

### SHAP Visualizations

The application provides three types of SHAP visualizations:

1. **Waterfall Plot**: Shows cumulative feature contributions
2. **Force Plot**: Displays push/pull effects of features
3. **Feature Impact Table**: Ranked list of feature importance

## 📊 Model Performance

The CatBoost model achieves high accuracy on the heart disease prediction task. Performance metrics include:
- Accuracy
- Precision and Recall
- Confusion Matrix
- Classification Report

## 🛠️ Technologies Used

- **Python 3.8+**: Programming language
- **CatBoost 1.2.3**: Gradient boosting library
- **SHAP 0.44.1**: Model explainability framework
- **Streamlit 1.32.0**: Web application framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning utilities
- **Matplotlib**: Data visualization
- **Plotly**: Interactive visualizations

## 📁 Project Structure

```
.
├── app.py                      # Streamlit web application
├── train_model.py              # Model training script
├── requirements.txt            # Python dependencies
├── data/
│   └── heart_disease.csv      # Heart disease dataset
├── models/
│   ├── catboost_heart_model.pkl    # Trained model
│   └── feature_names.pkl           # Feature names
└── README.md                   # Project documentation
```

## 🎓 Model Explanation

### CatBoost Algorithm
CatBoost (Categorical Boosting) is a gradient boosting algorithm that:
- Handles categorical features automatically
- Provides robust performance with minimal hyperparameter tuning
- Offers fast training and prediction
- Reduces overfitting through ordered boosting

### SHAP Values
SHAP provides game-theoretic explanations by:
- Computing each feature's contribution to predictions
- Offering consistent and locally accurate interpretations
- Supporting multiple visualization types
- Enabling trust through transparency

## ⚠️ Disclaimer

This application is for **educational and informational purposes only**. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available for educational purposes.

## 👨‍💻 Author

Created with ❤️ for advancing healthcare through AI and machine learning.

## 🔗 References

- [CatBoost Documentation](https://catboost.ai/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)
