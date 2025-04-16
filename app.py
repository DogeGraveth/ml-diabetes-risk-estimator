import pandas as pd
import joblib
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

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
    X, y, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# === Risk categorization ===
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
