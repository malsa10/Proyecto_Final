from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
import streamlit as st

import os

# Crear etiquetas binaria: 1 si el precio baja >2% próximo día, sino 0
def etiquetar_datos(data):
    data['Return'] = data['Close'].pct_change().shift(-1)
    data['Label'] = (data['Return'] < -0.02).astype(int)
    data.dropna(inplace=True)
    return data

def entrenar_modelo(data):
    features = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    labels = data['Label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, labels, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    st.text("Evaluación del modelo:")
    st.text(classification_report(y_test, y_pred))

    joblib.dump(model, 'models/modelo_riesgo.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')