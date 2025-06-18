import streamlit as st

def obtener_perfil():
    st.subheader("Cuestionario de perfil inversor")
    edad = st.slider("Edad", 18, 80)
    experiencia = st.radio("¿Tienes experiencia invirtiendo?", ["Ninguna", "Media", "Alta"])
    horizonte = st.selectbox("Horizonte de inversión", ["< 1 año", "1-5 años", "> 5 años"])
    tolerancia = st.radio("Nivel de tolerancia al riesgo", ["Baja", "Media", "Alta"])

    puntaje = 0
    if experiencia == "Alta": puntaje += 2
    elif experiencia == "Media": puntaje += 1

    if horizonte == "> 5 años": puntaje += 2
    elif horizonte == "1-5 años": puntaje += 1

    if tolerancia == "Alta": puntaje += 2
    elif tolerancia == "Media": puntaje += 1

    if puntaje <= 2:
        perfil = "Conservador"
    elif puntaje <= 4:
        perfil = "Moderado"
    else:
        perfil = "Agresivo"

    return perfil