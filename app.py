from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

app = Flask(__name__)

df = pd.read_csv('diabetes_dataset.csv')

df = df[df['Diabetes_012'] != 1]
df = df.rename(columns={'Diabetes_012': 'diabetes'})

selected_cols = [
    'HighBP', 'HighChol', 'BMI', 'HeartDiseaseorAttack', 'PhysActivity',
    'GenHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'diabetes'
]
df = df[selected_cols]

X = df.drop('diabetes', axis=1)
y = df['diabetes']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

def risk_category(prob):
    if prob >= 0.20:
        return "High Risk"
    elif prob >= 0.10:
        return "Moderate Risk"
    elif prob >= 0.5:
        return "Low Risk"
    else:
        return "Unlikely"

@app.route('/predict', methods=['POST'])
def predict():
    user_data = {key: float(request.form[key]) for key in request.form}
    user_df = pd.DataFrame([user_data])

    user_scaled = scaler.transform(user_df)
    user_prob = model.predict_proba(user_scaled)[0][1]
    risk = risk_category(user_prob)

    return render_template('result.html', risk_score=f"{user_prob:.2f}", risk_level=risk)

if __name__ == '__main__':
    app.run(debug=True)
