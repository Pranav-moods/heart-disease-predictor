import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load model
model = joblib.load("anaconda_projects_f355062a-aedf-4f44-836a-5a7904e03694_rf_model.pkl")

st.set_page_config(page_title="Heart Disease Prediction Dashboard", layout="wide")

st.title("Personalized Health Dashboard")
st.write("Enter your health details below:")

with st.form("Health form"):
    age = st.number_input("Age", 20, 100, 45)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (trestbps)", 80, 200, 120)
    chol = st.number_input("Cholesterol (chol)", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
    restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved (thalach)", 60, 250, 150)
    exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
    oldpeak = st.number_input("ST depression (oldpeak)", 0.0, 10.0, 1.0, step=0.1)
    slope = st.selectbox("Slope", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    # Keep as DataFrame (DON’T convert to numpy)
    input_data = pd.DataFrame(
        [[age, sex, cp, trestbps, chol, fbs, restecg,
          thalach, exang, oldpeak, slope, ca, thal]],
        columns=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                 "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
    )

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease (Probability: {probability:.2%})")
    else:
        st.success(f"✅ Low Risk of Heart Disease (Probability: {probability:.2%})")

    # Feature Importance
    st.subheader("Feature Importance")
    if hasattr(model, "feature_importances_"):
        feat_importances = pd.Series(
            model.feature_importances_,
            index=input_data.columns
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=feat_importances, y=feat_importances.index, ax=ax, palette="viridis")
        ax.set_title("Feature Importance (Random Forest)")
        st.pyplot(fig)
