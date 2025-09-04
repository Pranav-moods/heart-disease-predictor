# heart-disease-predictor

## 🫀 Heart Disease Prediction Dashboard

An interactive Streamlit web application that predicts the risk of heart disease based on medical attributes such as age, cholesterol, blood pressure, and more.
The app uses a Random Forest Classifier trained on the UCI Heart Disease Dataset
 to provide personalized health insights.

## 🚀 Features

📋 User-friendly health form – enter age, sex, chest pain type, cholesterol, etc.

🔍 Real-time prediction – outputs risk status (Low Risk ✅ / High Risk ⚠️).

📊 Probability score – shows confidence of prediction.

📈 Feature importance chart – explains which health factors contribute most.

🖥️ Interactive dashboard – built using Streamlit.

## 🛠️ Tech Stack

Frontend/UI: Streamlit

Machine Learning: RandomForestClassifier (scikit-learn)

Data Processing: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Model Serialization: Joblib

## 📂 Project Structure
heart-disease-predictor/
│
├── app.py                 # Streamlit application
├── rf_model.pkl           # Trained Random Forest model
├── requirements.txt       # Dependencies
├── README.md              # Project documentation
└── images/                # (Optional) screenshots for README

## ⚙️ Installation & Usage
1️⃣ Clone the repository
git clone https://github.com/your-username/heart-disease-predictor.git
cd heart-disease-predictor

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the app
streamlit run app.py

4️⃣ Access in browser

The app will run locally at:
👉 http://localhost:8501

## 📊 Example Output

Prediction Example:

✅ Low Risk (Probability: 23%)

⚠️ High Risk (Probability: 86%)

Feature Importance Visualization:
<img width="1710" height="926" alt="image" src="https://github.com/user-attachments/assets/b2c3a2ae-32ea-4929-88a3-d1488b8c04d5" />


## 📘 Dataset

The model is trained on the UCI Heart Disease Dataset, which includes 13 key health attributes such as:

Age

Sex

Chest Pain Type (cp)

Resting Blood Pressure (trestbps)

Cholesterol (chol)

Fasting Blood Sugar (fbs)

Resting ECG (restecg)

Max Heart Rate Achieved (thalach)

Exercise Induced Angina (exang)

ST Depression (oldpeak)

Slope of ST segment (slope)

Major vessels colored by fluoroscopy (ca)

Thalassemia (thal)

## 📈 Future Improvements

🔮 Add more ML models (e.g., XGBoost, Logistic Regression) for comparison

🌐 Deploy on Streamlit Cloud / HuggingFace Spaces / Heroku

📱 Improve UI with better visuals & health insights

🏥 Connect with real-time medical datasets for continuous improvement
