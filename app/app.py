# app/app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# =========================
# CONFIGURACIÓN INICIAL
# =========================
st.set_page_config(
    page_title="Transparencia Patrimonial CR",
    page_icon="💰",
    layout="wide"
)

st.title("Transparencia Patrimonial CR")
st.markdown("""
Aplicación desarrollada como parte del Desafío de Datos Abiertos PIDA.
Esta herramienta combina datos patrimoniales y registrales para detectar **patrones anómalos**
en la evolución del patrimonio de funcionarios públicos mediante modelos de *Machine Learning*.
""")

# =========================
# CARGA DE DATOS Y MODELO
# =========================
DATA_PATH1 = "../data/synthetic_cgr_declaraciones.csv"
DATA_PATH2 = "../data/synthetic_registros_propiedades.csv"
MODEL_PATH = "../models/trained_model.pkl"

@st.cache_data
def load_data():
    df1 = pd.read_csv(DATA_PATH1)
    df2 = pd.read_csv(DATA_PATH2)
    return pd.merge(df1, df2, on="id", how="inner")

@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except:
        return None

df = load_data()
model = load_model()

# =========================
# SECCIÓN 1: DATOS GENERALES
# =========================
st.header("Descripción general de los datos")

st.write("Vista previa del dataset:")
st.dataframe(df.head())

st.write("Estadísticas descriptivas:")
st.write(df.describe())

# =========================
# SECCIÓN 2: VISUALIZACIONES
# =========================
st.header("Visualizaciones exploratorias")

col1, col2 = st.columns(2)

with col1:
    fig1 = px.histogram(df, x="valor_patrimonio", nbins=30, title="Distribución del valor patrimonial")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.scatter(df, x="valor_propiedades", y="valor_patrimonio",
                      color="institucion", title="Patrimonio vs valor de propiedades")
    st.plotly_chart(fig2, use_container_width=True)

# =========================
# SECCIÓN 3: RESULTADOS DEL MODELO
# =========================
st.header("Resultados del modelo")

if "anomalia_score" in df.columns:
    st.write("Distribución de puntajes de anomalía (riesgo patrimonial):")
    fig3 = px.histogram(df, x="anomalia_score", nbins=25, color_discrete_sequence=["#EF553B"])
    st.plotly_chart(fig3, use_container_width=True)

    threshold = st.slider("Seleccionar umbral de riesgo:", 0.0, 1.0, 0.8)
    high_risk = df[df["anomalia_score"] > threshold]
    st.write(f"Funcionarios con riesgo mayor a {threshold}: {len(high_risk)} casos")
    st.dataframe(high_risk)
else:
    st.info("Aún no se han generado los puntajes de anomalía. Entrene el modelo y exporte los resultados en el notebook.")

# =========================
# SECCIÓN 4: CONCLUSIONES
# =========================
st.header("Conclusiones preliminares")

st.markdown("""
- Los datos sintéticos permiten visualizar la estructura esperada del sistema patrimonial.
- El modelo de detección de anomalías puede identificar casos con incrementos patrimoniales atípicos.
- Al incorporar los datos reales, el dashboard mostrará alertas automáticas y métricas agregadas por institución o región.
""")

st.caption("© 2025 Proyecto Transparencia Patrimonial CR – Instituto Tecnológico de Costa Rica")

