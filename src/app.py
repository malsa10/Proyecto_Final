# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.optimize import minimize

from data_loader import descargar_datos
from visualizer import graficar_precios
from model_trainer import etiquetar_datos, entrenar_modelo
from perfilador import obtener_perfil

st.set_page_config(page_title="App de Inversión con IA", layout="wide")
st.title("App de inversión con IA")

# --- Cargar lista de acciones
try:
    df_tickers = pd.read_csv("tickers_ampliado.csv")
except FileNotFoundError:
    st.error("Archivo tickers_ampliado.csv no encontrado.")
    st.stop()

# --- Seleccionar acción
if not df_tickers.empty:
    opciones = df_tickers["Nombre"] + " (" + df_tickers["Ticker"] + ")"
    opcion = st.selectbox("Selecciona una acción:", opciones)

    try:
        ticker = df_tickers[opciones == opcion]["Ticker"].values[0]
    except IndexError:
        st.error("Error al seleccionar el ticker.")
        st.stop()
else:
    st.error("El archivo tickers_ampliado.csv está vacío.")
    st.stop()

# --- Descargar datos
if ticker:
    datos = descargar_datos(ticker)
    graficar_precios(datos, ticker)
    datos = etiquetar_datos(datos)

    if st.button("Entrenar modelo de riesgo"):
        entrenar_modelo(datos)
        st.success("Modelo entrenado y guardado en /models")

    if st.button("Predecir riesgo de hoy"):
        try:
            modelo = joblib.load('models/modelo_riesgo.pkl')
            scaler = joblib.load('models/scaler.pkl')

            ult_dato = datos[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[-1:]
            ult_dato_scaled = scaler.transform(ult_dato)
            pred = modelo.predict(ult_dato_scaled)[0]
            prob = modelo.predict_proba(ult_dato_scaled)[0]

            st.subheader("Resultado de predicción de riesgo")
            col1, col2 = st.columns(2)

            if pred == 1:
                col1.error("ALERTA: Riesgo de caída significativa")
            else:
                col1.success("Estable: Bajo riesgo estimado")

            col2.metric("Probabilidad de riesgo", f"{prob[1]*100:.2f}%")
            col2.metric("Probabilidad de estabilidad", f"{prob[0]*100:.2f}%")

        except FileNotFoundError:
            st.error("Modelo no encontrado. Entrena primero el modelo.")

    # --- Perfil de inversión
    perfil = obtener_perfil()
    st.info(f"Tu perfil es: {perfil}")

    # --- Optimización de portafolio
    daily_returns = datos[['Close']].pct_change().dropna()
    mean_daily_ret = daily_returns.mean()
    cov_matrix = daily_returns.cov()

    def annualized_return(w):
        return np.sum(mean_daily_ret * w) * 252

    def annualized_volatility(w):
        var = np.dot(w.T, np.dot(cov_matrix, w))
        return np.sqrt(var) * np.sqrt(252)

    def optimize_portfolio(obj_fn):
        n = 1  # Solo una acción
        w0 = np.ones(n) / n
        bds = [(0, 1)] * n
        cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        res = minimize(obj_fn, w0, bounds=bds, constraints=cons)
        return res.x

    profiles = {
        "Conservador": lambda w: 0.75 * annualized_volatility(w) - 0.25 * annualized_return(w),
        "Moderado": lambda w: 0.50 * annualized_volatility(w) - 0.50 * annualized_return(w),
        "Agresivo": lambda w: 0.25 * annualized_volatility(w) - 0.75 * annualized_return(w)
    }

    w_opt = optimize_portfolio(profiles[perfil])
    retorno = annualized_return(w_opt)
    volatilidad = annualized_volatility(w_opt)

    st.subheader("Cartera recomendada")
    st.write(f"**Acción recomendada:** {ticker}")
    st.write(f"**Rentabilidad anual esperada:** {retorno:.2%}")
    st.write(f"**Volatilidad esperada:** {volatilidad:.2%}")
