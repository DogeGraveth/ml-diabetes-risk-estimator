import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

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

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

joblib.dump((model, scaler), 'model.pkl')