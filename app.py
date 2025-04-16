import pandas as pd
import joblib
import requests
import os
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

MODEL_URL = "https://drive.google.com/uc?export=download&id=1dJ91tXtengUXBdHyB0miCIDwaunyQibH"

MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    print("Model downloaded successfully.")

model, scaler = joblib.load(MODEL_PATH)

def risk_category(prob):
    if prob >= 0.30:
        return "High Risk"
    elif prob >= 0.20:
        return "Moderate Risk"
    elif prob >= 0.15:
        return "Low Risk"
    else:
        return "Unlikely"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_data = {
        'HighBP': int(request.form['HighBP']),
        'HighChol': int(request.form['HighChol']),
        'BMI': float(request.form['BMI']),
        'HeartDiseaseorAttack': int(request.form['HeartDiseaseorAttack']),
        'PhysActivity': int(request.form['PhysActivity']),
        'GenHlth': int(request.form['GenHlth']),
        'PhysHlth': int(request.form['PhysHlth']),
        'DiffWalk': int(request.form['DiffWalk']),
        'Sex': int(request.form['Sex']),
        'Age': int(request.form['Age']),
    }

    user_df = pd.DataFrame([user_data])
    user_scaled = scaler.transform(user_df)

    user_prob = model.predict_proba(user_scaled)[0][1]
    user_score = round(user_prob * 100, 2)
    risk = risk_category(user_prob)

    return render_template('result.html',
                           risk_score=user_score,
                           risk_level=risk)

if __name__ == '__main__':
    app.run(debug=True)
