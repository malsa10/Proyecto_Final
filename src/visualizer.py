import matplotlib.pyplot as plt
import streamlit as st

def graficar_precios(data, ticker):
    st.subheader(f"Precio de cierre de {ticker}")
    st.line_chart(data['Close'])