# app/app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os
from pathlib import Path
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import json
import unicodedata
from typing import Dict, Any
import subprocess
import sys
import tempfile
import shutil

try:
    import geopandas as gpd
except Exception:
    gpd = None  # Fallback a requests + px si no está disponible

try:
    import requests
except Exception:
    requests = None  # Se manejará más adelante si no está disponible

# BASE_DIR: carpeta raíz del proyecto (2 niveles arriba de este archivo)
BASE_DIR = Path(__file__).resolve().parents[1]

# Configuración de página (debe ser la primera llamada de Streamlit)
st.set_page_config(
    page_title="Transparencia Patrimonial CR", page_icon="💰", layout="wide"
)

# poetry shell
# poetry run streamlit run app/app.py

# docker-compose up --build
# docker build -t transparencia-patrimonial-cr .

# =========================
# CONFIGURACIÓN INICIAL
# =========================
ANOMALIES_FILENAME = "analisis_anomalias_territorial.csv"
RANKING_FILENAME = "ranking_provincias_anomalias.csv"
ANOMALIES_PATH = BASE_DIR / ANOMALIES_FILENAME

# Rutas de datos y modelo (definidas temprano para usarlas en la UI de carga)
DATA_PATH1 = BASE_DIR / "data" / "synthetic_cgr_declaraciones.csv"
DATA_PATH2 = BASE_DIR / "data" / "synthetic_registro_nacional.csv"
MODEL_PATH = BASE_DIR / "models" / "trained_model.pkl"


def _find_existing_file(name: str) -> Path | None:
    """Busca un archivo por nombre en ubicaciones típicas (repo root, /app, cwd)."""
    candidates = [BASE_DIR, Path("/app"), Path.cwd()]
    for d in candidates:
        try:
            p = d / name
            if p.exists():
                return p
        except Exception:
            continue
    return None


@st.cache_data
def load_anomalies():
    try:
        p = _find_existing_file(ANOMALIES_FILENAME)
        if p is None:
            return None
        return pd.read_csv(str(p))
    except Exception:
        return None


st.title("Transparencia Patrimonial CR")
st.markdown(
    """
Aplicación desarrollada como parte del Desafío de Datos Abiertos PIDA.
Esta herramienta combina datos patrimoniales y registrales para detectar **patrones anómalos**
en la evolución del patrimonio de funcionarios públicos mediante modelos de *Machine Learning*.
"""
)

# =========================
# ARCHIVOS DE ENTRADA (Sidebar)
# =========================
st.sidebar.header("Datos de entrada")
st.sidebar.caption(
    "Cargue nuevos CSV para reemplazar los datos actuales de la carpeta data/."
)

uploaded_cgr = st.sidebar.file_uploader(
    "CGR — synthetic_cgr_declaraciones.csv",
    type=["csv"],
    help="Archivo con declaraciones (anio_declaracion, activos_totales, pasivos_totales, provincia, canton, distrito, identificador_anonimo_declarante, ...)",
)

uploaded_reg = st.sidebar.file_uploader(
    "Registro — synthetic_registro_nacional.csv",
    type=["csv"],
    help="Archivo con actos registrales (periodo o anio, provincia, canton, distrito, tipo_acto_registral, valor_declarado, ...)",
)

col_btn_upload1, col_btn_upload2 = st.sidebar.columns(2)
apply_uploads = col_btn_upload1.button("Reemplazar archivos")
reset_btn = col_btn_upload2.button("Restablecer caché")

if apply_uploads:
    try:
        replaced = []
        if uploaded_cgr is not None:
            content = uploaded_cgr.read()
            Path(DATA_PATH1).write_bytes(content)
            replaced.append(Path(DATA_PATH1).name)
        if uploaded_reg is not None:
            content = uploaded_reg.read()
            Path(DATA_PATH2).write_bytes(content)
            replaced.append(Path(DATA_PATH2).name)
        if replaced:
            st.sidebar.success(f"Reemplazados: {', '.join(replaced)}")
            try:
                st.cache_data.clear()
            except Exception:
                pass
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()
    except Exception as e:
        st.sidebar.error(f"Error al reemplazar archivos: {e}")

# =========================
# CARGA DE DATOS Y MODELO
# =========================


@st.cache_data
def load_data():
    # Verificar que los archivos existen antes de leer
    p1 = Path(DATA_PATH1)
    p2 = Path(DATA_PATH2)
    if not p1.exists() or not p2.exists():
        missing = []
        if not p1.exists():
            missing.append(str(p1))
        if not p2.exists():
            missing.append(str(p2))
        # No usar st.error aquí directamente para poder testear la función; la llamadora mostrará el mensaje.
        raise FileNotFoundError(f"Faltan archivos de datos: {', '.join(missing)}")

    df1 = pd.read_csv(str(DATA_PATH1))
    df2 = pd.read_csv(str(DATA_PATH2))

    # Crear columnas derivadas en df1
    df1["valor_patrimonio"] = (df1["activos_totales"] - df1["pasivos_totales"]).clip(
        lower=0
    )
    df1["valor_propiedades"] = (df1["distrib_inmuebles_%"] / 100) * df1[
        "activos_totales"
    ]

    # Promedio de valor declarado por provincia y año en df2
    # Derivar año desde 'periodo' o usar 'anio' si ya existe
    if "periodo" in df2.columns:
        df2["anio"] = pd.to_datetime(df2["periodo"], errors="coerce").dt.year
    elif "anio" not in df2.columns:
        df2["anio"] = pd.NaT
    df2_grouped = df2.groupby(["provincia", "anio"], as_index=False)[
        "valor_declarado"
    ].mean()
    df2_grouped.rename(
        columns={"valor_declarado": "valor_medio_registral"}, inplace=True
    )

    # Unir ambos por provincia y año
    df1["anio"] = df1["anio_declaracion"]
    df = pd.merge(
        df1, df2_grouped, on=["provincia", "anio"], how="left", validate="m:1"
    )

    # Rellenar valores faltantes
    df["valor_medio_registral"].fillna(0, inplace=True)

    # Si el modelo ya generó scores
    if "anomalia_score" in df.columns:
        df["anomalia_score"] = df["anomalia_score"].astype(float)

    return df


def load_raw_inputs():
    """Cargar los dos CSVs originales (CGR y Registro Nacional)."""
    p1 = Path(DATA_PATH1)
    p2 = Path(DATA_PATH2)
    if not p1.exists() or not p2.exists():
        missing = []
        if not p1.exists():
            missing.append(str(p1))
        if not p2.exists():
            missing.append(str(p2))
        raise FileNotFoundError(f"Faltan archivos de datos: {', '.join(missing)}")

    df_cgr = pd.read_csv(str(p1))
    df_reg = pd.read_csv(str(p2))
    return df_cgr, df_reg


def aggregate_registro(df_reg):
    # periodo -> anio
    if "periodo" in df_reg.columns:
        df_reg["anio"] = pd.to_datetime(df_reg["periodo"], errors="coerce").dt.year
    elif "anio" not in df_reg.columns:
        # intentar inferir de alguna columna
        df_reg["anio"] = pd.NaT

    grp = (
        df_reg.groupby(["anio", "provincia", "canton", "distrito"])
        .agg(
            valor_total_bienes=("valor_declarado", "sum"),
            valor_medio_bienes=("valor_declarado", "mean"),
            cant_actos_registrales=("tipo_acto_registral", "count"),
        )
        .reset_index()
    )
    return grp


def aggregate_cgr(df_cgr):
    # crear patrimonio neto
    if "patrimonio_neto" not in df_cgr.columns:
        if "activos_totales" in df_cgr.columns and "pasivos_totales" in df_cgr.columns:
            df_cgr["patrimonio_neto"] = (
                df_cgr["activos_totales"] - df_cgr["pasivos_totales"]
            )
        else:
            df_cgr["patrimonio_neto"] = 0

    df_cgr_agg = (
        df_cgr.groupby(["anio_declaracion", "provincia", "canton", "distrito"])
        .agg(
            patrimonio_neto=("patrimonio_neto", "sum"),
            patrimonio_medio=("patrimonio_neto", "mean"),
            activos_totales=("activos_totales", "sum"),
            activos_medios=("activos_totales", "mean"),
            pasivos_totales=("pasivos_totales", "sum"),
            pasivos_medios=("pasivos_totales", "mean"),
            cant_funcionarios=("identificador_anonimo_declarante", "count"),
        )
        .reset_index()
    )

    # renombrar para merge
    df_cgr_agg = df_cgr_agg.rename(columns={"anio_declaracion": "anio"})
    return df_cgr_agg


def build_features_and_merge(df_cgr_agg, df_reg_agg):
    df_merged = pd.merge(
        df_cgr_agg,
        df_reg_agg,
        on=["anio", "provincia", "canton", "distrito"],
        how="inner",
        validate="one_to_one",
    )

    # Ratios
    df_merged["ratio_bienes_vs_patrimonio"] = df_merged[
        "valor_total_bienes"
    ] / df_merged["patrimonio_neto"].replace(0, np.nan)
    df_merged["ratio_bienes_vs_activos"] = df_merged["valor_total_bienes"] / df_merged[
        "activos_totales"
    ].replace(0, np.nan)
    df_merged["densidad_actos_por_func"] = df_merged[
        "cant_actos_registrales"
    ] / df_merged["cant_funcionarios"].replace(0, np.nan)

    df_merged = df_merged.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Variaciones por provincia (ordenar primero)
    df_merged = df_merged.sort_values(["provincia", "anio"])
    for col in [
        "valor_total_bienes",
        "patrimonio_neto",
        "activos_totales",
        "pasivos_totales",
    ]:
        df_merged[f"var_{col}"] = (
            df_merged.groupby("provincia")[col].pct_change().fillna(0)
        )

    df_merged["dif_var_bienes_patrimonio"] = (
        df_merged["var_valor_total_bienes"] - df_merged["var_patrimonio_neto"]
    )

    return df_merged


def run_isolation_pipeline(df_merged, contamination=0.05, n_estimators=200):
    features = [
        "ratio_bienes_vs_patrimonio",
        "ratio_bienes_vs_activos",
        "densidad_actos_por_func",
        "var_valor_total_bienes",
        "var_patrimonio_neto",
        "dif_var_bienes_patrimonio",
    ]

    X = df_merged[features].fillna(0).values
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(X)

    iso = IsolationForest(
        n_estimators=int(n_estimators), contamination=contamination, random_state=42
    )
    preds = iso.fit_predict(x_scaled)
    decision = iso.decision_function(x_scaled)

    # Añadir columnas
    df_merged["score_anomalia"] = preds
    df_merged["anomaly_score"] = decision

    # Ranking por provincia (promedio de anomaly_score)
    ranking = (
        df_merged.groupby("provincia")["anomaly_score"]
        .mean()
        .sort_values()
        .reset_index()
    )
    ranking["ranking"] = np.arange(1, len(ranking) + 1)

    # Guardar resultados
    out_analisis = BASE_DIR / "analisis_anomalias_territorial.csv"
    out_ranking = BASE_DIR / "ranking_provincias_anomalias.csv"
    df_merged.to_csv(str(out_analisis), index=False)
    ranking.to_csv(str(out_ranking), index=False)

    return df_merged, ranking


# -------------------------
# Utilidades para mapa por provincia
# -------------------------
def _normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    # remover acentos
    text = unicodedata.normalize("NFKD", text)
    text = "".join([c for c in text if not unicodedata.combining(c)])
    return text


def _canton_code_name_map() -> Dict[int, str]:
    """Mapa de código de cantón (101..706) a nombre oficial."""
    m: Dict[int, str] = {}
    m.update(
        {
            101: "San José",
            102: "Escazú",
            103: "Desamparados",
            104: "Puriscal",
            105: "Tarrazú",
            106: "Aserrí",
            107: "Mora",
            108: "Goicoechea",
            109: "Santa Ana",
            110: "Alajuelita",
            111: "Vásquez de Coronado",
            112: "Acosta",
            113: "Tibás",
            114: "Moravia",
            115: "Montes de Oca",
            116: "Turrubares",
            117: "Dota",
            118: "Curridabat",
            119: "Pérez Zeledón",
            120: "León Cortés",
        }
    )
    m.update(
        {
            201: "Alajuela",
            202: "San Ramón",
            203: "Grecia",
            204: "San Mateo",
            205: "Atenas",
            206: "Naranjo",
            207: "Palmares",
            208: "Poás",
            209: "Orotina",
            210: "San Carlos",
            211: "Zarcero",
            212: "Sarchí",
            213: "Upala",
            214: "Los Chiles",
            215: "Guatuso",
        }
    )
    m.update(
        {
            301: "Cartago",
            302: "Paraíso",
            303: "La Unión",
            304: "Jiménez",
            305: "Turrialba",
            306: "Alvarado",
            307: "Oreamuno",
            308: "El Guarco",
        }
    )
    m.update(
        {
            401: "Heredia",
            402: "Barva",
            403: "Santo Domingo",
            404: "Santa Bárbara",
            405: "San Rafael",
            406: "San Isidro",
            407: "Belén",
            408: "Flores",
            409: "San Pablo",
            410: "Sarapiquí",
        }
    )
    m.update(
        {
            501: "Liberia",
            502: "Nicoya",
            503: "Santa Cruz",
            504: "Bagaces",
            505: "Carrillo",
            506: "Cañas",
            507: "Abangares",
            508: "Tilarán",
            509: "Nandayure",
            510: "La Cruz",
            511: "Hojancha",
        }
    )
    m.update(
        {
            601: "Puntarenas",
            602: "Esparza",
            603: "Buenos Aires",
            604: "Montes de Oro",
            605: "Osa",
            606: "Quepos",
            607: "Golfito",
            608: "Coto Brus",
            609: "Parrita",
            610: "Corredores",
            611: "Garabito",
        }
    )
    m.update(
        {
            701: "Limón",
            702: "Pococí",
            703: "Siquirres",
            704: "Talamanca",
            705: "Matina",
            706: "Guácimo",
        }
    )
    return m


def load_province_geojson() -> Dict[str, Any]:
    """Construye el GeoJSON de provincias usando GeoPandas si está disponible; si no, usa requests como fallback.

    Estructura solicitada:
    - geopandas.read_file("GeoJSON:{url}") para 7 URLs
    - nombres_provincias en el mismo orden
    - concatenación a un único GeoDataFrame
    - merge con ranking por nombre de provincia en minúsculas
    """
    # Intento preferido: GeoPandas
    if gpd is not None:
        try:
            urls_provincias = [
                "https://raw.githubusercontent.com/schweini/CR_distritos_geojson/master/geojson/1.geojson",  # San José
                "https://raw.githubusercontent.com/schweini/CR_distritos_geojson/master/geojson/2.geojson",  # Alajuela
                "https://raw.githubusercontent.com/schweini/CR_distritos_geojson/master/geojson/3.geojson",  # Cartago
                "https://raw.githubusercontent.com/schweini/CR_distritos_geojson/master/geojson/4.geojson",  # Heredia
                "https://raw.githubusercontent.com/schweini/CR_distritos_geojson/master/geojson/5.geojson",  # Guanacaste
                "https://raw.githubusercontent.com/schweini/CR_distritos_geojson/master/geojson/6.geojson",  # Puntarenas
                "https://raw.githubusercontent.com/schweini/CR_distritos_geojson/master/geojson/7.geojson",  # Limón
            ]
            nombres_provincias = [
                "san josé",
                "alajuela",
                "cartago",
                "heredia",
                "guanacaste",
                "puntarenas",
                "limón",
            ]

            geo_list = []
            for i, url in enumerate(urls_provincias):
                gdf = gpd.read_file(f"GeoJSON:{url}")
                gdf["provincia"] = nombres_provincias[i]
                geo_list.append(gdf)
            geo_provincias = pd.concat(geo_list, ignore_index=True)

            # Normalizar nombres para join robusto
            geo_provincias["provincia_norm"] = geo_provincias["provincia"].apply(
                _normalize_text
            )
            # Devolver GeoJSON dict listo para choropleth
            geojson = json.loads(geo_provincias.to_json())
            return geojson
        except Exception:
            pass  # Fallback más abajo

    # Fallback: requests + construcción manual del GeoJSON (como estaba antes)
    try:
        features = []
        for idx, name in enumerate(
            [
                "san jose",
                "alajuela",
                "cartago",
                "heredia",
                "guanacaste",
                "puntarenas",
                "limon",
            ],
            start=1,
        ):
            url = f"https://raw.githubusercontent.com/schweini/CR_distritos_geojson/master/geojson/{idx}.geojson"
            if requests is None:
                return {}
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            gj = resp.json()
            for feat in gj.get("features", []):
                if "properties" not in feat:
                    feat["properties"] = {}
                feat["properties"]["provincia"] = name
                feat["properties"]["provincia_norm"] = _normalize_text(name)
                features.append(feat)
        return {"type": "FeatureCollection", "features": features}
    except Exception:
        return {}


def render_province_map(
    ranking_df: pd.DataFrame, titulo: str = "Mapa de anomalías por provincia"
):
    try:
        if ranking_df is None or ranking_df.empty:
            return
        gjson = load_province_geojson()
        if not gjson or not isinstance(gjson, dict):
            st.info("No se pudo cargar el mapa de provincias (GeoJSON no disponible).")
            return

        df_map = ranking_df.copy()
        # Normalizar nombres para el join
        df_map["provincia_norm"] = df_map["provincia"].apply(_normalize_text)
        # px.choropleth hará match con properties.provincia_norm del geojson
        import plotly.express as px  # asegurar contexto

        feature_key = (
            "properties.provincia_norm"
            if any(
                f.get("properties", {}).get("provincia_norm") is not None
                for f in gjson.get("features", [])
            )
            else "properties.provincia"
        )
        fig = px.choropleth(
            df_map,
            geojson=gjson,
            locations="provincia_norm",
            featureidkey=feature_key,
            color="anomaly_score",
            color_continuous_scale="RdYlGn",
            hover_name="provincia",
            title=titulo,
        )
        fig.update_geos(fitbounds="locations", visible=False, bgcolor="rgba(0,0,0,0)")
        fig.update_layout(
            margin={"r": 0, "t": 50, "l": 0, "b": 0},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(f"No se pudo renderizar el mapa: {e}")


@st.cache_resource
def load_model():
    try:
        if not Path(MODEL_PATH).exists():
            return None
        return joblib.load(str(MODEL_PATH))
    except Exception:
        return None


# Cargar datos y manejar faltantes de forma amigable para el usuario
try:
    df = load_data()
except FileNotFoundError as e:
    st.error(
        "No se pudieron cargar los CSV necesarios. "
        + "Asegúrese de que la carpeta `data/` existe y contiene los archivos esperados: \n"
        + str(e)
    )
    st.stop()

model = load_model()

# =========================
# SECCIÓN 1: DATOS GENERALES
# =========================
st.header("Descripción general de los datos")

# KPIs relacionados con el análisis del modelo
col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
col_kpi1.metric("Registros base (CGR x año)", f"{len(df):,}")
try:
    anio_min, anio_max = int(df["anio"].min()), int(df["anio"].max())
    col_kpi2.metric("Cobertura temporal", f"{anio_min}–{anio_max}")
except Exception:
    col_kpi2.metric("Cobertura temporal", "ND")

col_kpi3.metric("Provincias", df["provincia"].nunique())
col_kpi4.metric("Ámbito geográfico", "Provincias, Cantones y Distritos")

# Métricas de variables clave que alimentan el modelo
# with st.expander("Ver resumen de variables relevantes"):
#    cols_v = st.columns(2)
#    cols_v[0].write("Promedios (aprox.)")
#    cols_v[0].write({
#        "valor_patrimonio_prom": round(float(df["valor_patrimonio"].mean()), 2),
#        "valor_propiedades_prom": round(float(df["valor_propiedades"].mean()), 2),
#        "valor_medio_registral": round(float(df["valor_medio_registral"].mean()), 2),
#    })
#    cols_v[1].write("Vista previa (primeras filas)")
#    cols_v[1].dataframe(df.head())

# Si hay resultados del análisis territorial, mostrar un resumen breve
analisis_path = BASE_DIR / "analisis_anomalias_territorial.csv"
if analisis_path.exists():
    try:
        df_analisis = pd.read_csv(str(analisis_path))
        st.subheader("Resumen del análisis territorial (última ejecución)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Filas territoriales", f"{len(df_analisis):,}")
        if "anomaly_score" in df_analisis.columns:
            c2.metric("Score medio", f"{df_analisis['anomaly_score'].mean():.4f}")
        # Métricas de MAD si existen
        if "mad_score" in df_analisis.columns:
            c3.metric("MAD promedio", f"{df_analisis['mad_score'].mean():.4f}")
        if "es_anomalia_mad" in df_analisis.columns:
            prop_mad = df_analisis["es_anomalia_mad"].mean() * 100
            c4.metric("% filas anómalas (MAD)", f"{prop_mad:.2f}%")

        if "anio" in df_analisis.columns:
            try:
                c3.metric(
                    "Años en análisis",
                    f"{int(df_analisis['anio'].min())}–{int(df_analisis['anio'].max())}",
                )
            except Exception:
                c3.metric("Años en análisis", "ND")
        if "provincia" in df_analisis.columns:
            c4.metric("Provincias (análisis)", df_analisis["provincia"].nunique())

        if {"provincia", "anomaly_score"}.issubset(df_analisis.columns):
            top5 = (
                df_analisis.groupby("provincia")["anomaly_score"]
                .mean()
                .sort_values()
                .head(5)
                .reset_index()
            )
        
            top5["ranking"] = range(1, len(top5) + 1)
            st.write("Provincias con mayor atipicidad (score más bajo = más anómalo):")
            st.dataframe(top5)
                    # Ranking simple por MAD (promedio de mad_score por provincia)
        if {"provincia", "mad_score"}.issubset(df_analisis.columns):
            top5_mad = (
                df_analisis.groupby("provincia")["mad_score"]
                .mean()
                .sort_values(ascending=False)  # mayor MAD = más extremo
                .head(5)
                .reset_index()
            )
            top5_mad["ranking_mad"] = range(1, len(top5_mad) + 1)
            st.write("Provincias más extremas según MAD (promedio de mad_score):")
            st.dataframe(top5_mad)

    except Exception:
        pass

## (Sección de visualizaciones exploratorias removida a solicitud del usuario)

# =========================
# SECCIÓN 3: ANÁLISIS TERRITORIAL Y RANKINGS
# =========================
st.header("Análisis territorial — detección de zonas de riesgo")

# Botón para ejecutar el análisis (sin parámetros configurables)
run_btn = st.sidebar.button("Ejecutar análisis territorial")

# Parámetros de visualización
top_n = st.sidebar.slider("Mostrar Top N cantones/distritos", 5, 25, 10, step=1)
order_opt = st.sidebar.radio("Orden", ["Más atípicas primero", "Alfabético"], index=0)
year_filter = None

# Si ya existe ranking generado previamente, mostrarlo en pestañas (Provincias/Cantones/Distritos)
ranking_path = _find_existing_file(RANKING_FILENAME)
if ranking_path is not None and ranking_path.exists() and not run_btn:
    st.subheader("Rankings territoriales (cargados desde CSV)")
    ranking_df = pd.read_csv(str(ranking_path))

    df_src = None
    if analisis_path.exists():
        try:
            df_src = pd.read_csv(str(analisis_path))
        except Exception:
            df_src = None

    tab_prov, tab_cant, tab_dist = st.tabs(["Provincias", "Cantones", "Distritos"])

    # Provincias
    with tab_prov:
        # Filtro de año si hay detalle
        if df_src is not None and "anio" in df_src.columns:
            years = sorted([int(y) for y in df_src["anio"].dropna().unique()])
            if years:
                year_filter = st.selectbox(
                    "Filtrar por año (opcional)", ["Todos"] + years, index=0
                )
                if year_filter != "Todos":
                    df_year = df_src[df_src["anio"] == int(year_filter)]
                    ranking_df = (
                        df_year.groupby("provincia")["anomaly_score"]
                        .mean()
                        .sort_values()
                        .reset_index()
                    )
                    ranking_df["ranking"] = range(1, len(ranking_df) + 1)

        if order_opt == "Más atípicas primero":
            ranking_ordered = (
                ranking_df.sort_values(["ranking"])
                if "ranking" in ranking_df.columns
                else ranking_df.sort_values("anomaly_score")
            )
        else:
            ranking_ordered = ranking_df.sort_values("provincia")
        ranking_view = ranking_ordered.head(top_n)

        st.dataframe(ranking_view)
        fig_rank = px.bar(
            ranking_view,
            x="provincia",
            y="anomaly_score",
            color="anomaly_score",
            color_continuous_scale="RdYlGn",
            text="ranking" if "ranking" in ranking_view.columns else None,
            title="Ranking de provincias por score de anomalía (promedio)",
        )
        fig_rank.update_layout(
            xaxis_title="Provincia", yaxis_title="Score de anomalía (↓ = más atípico)"
        )
        st.plotly_chart(fig_rank, use_container_width=True)

        # Mapa por provincia (usar todo el ranking para asegurar cobertura completa)
        try:
            render_province_map(ranking_df, titulo="Mapa de anomalías por provincia")
        except Exception:
            pass
                # Comparación Isolation Forest vs MAD por provincia (si existen columnas)
        if df_src is not None and {"provincia", "anomaly_score", "mad_score"}.issubset(df_src.columns):
            comp = df_src.groupby("provincia").agg(
                anomaly_score_mean=("anomaly_score", "mean"),
                mad_score_mean=("mad_score", "mean"),
                mad_anom_frac=("es_anomalia_mad", "mean"),
            ).reset_index()
            comp["mad_anom_frac"] = (comp["mad_anom_frac"] * 100).round(2)
            st.markdown(
                        """
                    **¿Qué es MAD?**

                    El método MAD (*Median Absolute Deviation*) es un algoritmo estadístico robusto para detectar valores atípicos.
                    En lugar de usar la media y la desviación estándar, calcula:

                    - La **mediana** de los datos.
                    - La desviación absoluta de cada valor respecto a esa mediana.
                    - La **mediana de esas desviaciones** (MAD).

                    A partir de eso se construye un puntaje de atipicidad (*score MAD*):  
                    valores más altos indican que una provincia, cantón o distrito se aleja más del patrón “normal”
                    de comportamiento patrimonial y registral.
                    """
                )

            st.markdown("#### Comparación por provincia: Isolation Forest vs MAD")
            st.dataframe(comp.sort_values("mad_score_mean", ascending=False).head(top_n))
            st.markdown("### Distribución de MAD Score")

            fig_box = px.box(
                df_src,
                x="provincia",
                y="mad_score",
                title="Distribución de MAD Score por provincia",
                labels={"mad_score": "MAD Score", "provincia": "Province"},
                color="provincia"
            )

            fig_box.update_layout(showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)


    # Cantones
    with tab_cant:
        if df_src is None:
            st.info(
                "Para ver ranking de cantones, ejecute el análisis o asegúrese de que 'analisis_anomalias_territorial.csv' exista."
            )
        else:
            prov_filter = st.selectbox(
                "Filtrar por provincia (opcional)",
                ["Todas"] + sorted(df_src["provincia"].unique().tolist()),
            )
            df_c = df_src.copy()
            if prov_filter != "Todas":
                df_c = df_c[df_c["provincia"] == prov_filter]
            if year_filter and year_filter != "Todos" and "anio" in df_c.columns:
                df_c = df_c[df_c["anio"] == int(year_filter)]
            rk_cant = (
                df_c.groupby(["provincia", "canton"])["anomaly_score"]
                .mean()
                .sort_values()
                .reset_index()
            )
            rk_cant["ranking"] = range(1, len(rk_cant) + 1)
            rk_cant_view = rk_cant.head(top_n)
            st.dataframe(rk_cant_view)
            fig_cant = px.bar(
                rk_cant_view,
                x="canton",
                y="anomaly_score",
                color="anomaly_score",
                color_continuous_scale="RdYlGn",
                text="ranking",
                title="Ranking de cantones por score de anomalía (promedio)",
            )
            fig_cant.update_layout(
                xaxis_title="Cantón", yaxis_title="Score de anomalía (↓ = más atípico)"
            )
            st.plotly_chart(fig_cant, use_container_width=True)
                        # Comparación Isolation Forest vs MAD por cantón (si existen columnas)
            if {"provincia", "canton", "anomaly_score", "mad_score", "es_anomalia_mad"}.issubset(df_c.columns):
                comp_cant = (
                    df_c.groupby(["provincia", "canton"])
                    .agg(
                        anomaly_score_mean=("anomaly_score", "mean"),
                        mad_score_mean=("mad_score", "mean"),
                        mad_anom_frac=("es_anomalia_mad", "mean"),
                    )
                    .reset_index()
                )
                comp_cant["mad_anom_frac"] = (comp_cant["mad_anom_frac"] * 100).round(2)

                st.markdown(
                        """
                    **¿Qué es MAD?**

                    El método MAD (*Median Absolute Deviation*) es un algoritmo estadístico robusto para detectar valores atípicos.
                    En lugar de usar la media y la desviación estándar, calcula:

                    - La **mediana** de los datos.
                    - La desviación absoluta de cada valor respecto a esa mediana.
                    - La **mediana de esas desviaciones** (MAD).

                    A partir de eso se construye un puntaje de atipicidad (*score MAD*):  
                    valores más altos indican que una provincia, cantón o distrito se aleja más del patrón “normal”
                    de comportamiento patrimonial y registral.
                    """
                )
                st.markdown("### Comparación por cantón: Isolation Forest vs MAD")
                st.dataframe(
                    comp_cant.sort_values("mad_score_mean", ascending=False).head(top_n)
                )
                st.subheader("Distribución de MAD Score")

                top10_cantones = (
                    df_src.groupby("canton")["mad_score"]
                    .mean()
                    .sort_values(ascending=False)
                    .head(10)
                    .index.tolist()
                )

                df_top_cant = df_src[df_src["canton"].isin(top10_cantones)]

                fig_mad_cant = px.box(
                    df_top_cant,
                    x="canton",
                    y="mad_score",
                    color="canton",
                    points="all",
                    title="Distribución de MAD Score por cantón",
                )

                fig_mad_cant.update_layout(xaxis_title="Cantón", yaxis_title="MAD Score")
                st.plotly_chart(fig_mad_cant, use_container_width=True)

    # Distritos
    with tab_dist:
        if df_src is None:
            st.info(
                "Para ver ranking de distritos, ejecute el análisis o asegúrese de que 'analisis_anomalias_territorial.csv' exista."
            )
        else:
            prov_filter_d = st.selectbox(
                "Filtrar por provincia (opcional)",
                ["Todas"] + sorted(df_src["provincia"].unique().tolist()),
                key="prov_dist",
            )
            df_d = df_src.copy()
            if prov_filter_d != "Todas":
                df_d = df_d[df_d["provincia"] == prov_filter_d]
            if year_filter and year_filter != "Todos" and "anio" in df_d.columns:
                df_d = df_d[df_d["anio"] == int(year_filter)]
            rk_dist = (
                df_d.groupby(["provincia", "canton", "distrito"])["anomaly_score"]
                .mean()
                .sort_values()
                .reset_index()
            )
            rk_dist["ranking"] = range(1, len(rk_dist) + 1)
            rk_dist_view = rk_dist.head(top_n)
            st.dataframe(rk_dist_view)
            fig_dist = px.bar(
                rk_dist_view,
                x="distrito",
                y="anomaly_score",
                color="anomaly_score",
                color_continuous_scale="RdYlGn",
                text="ranking",
                title="Ranking de distritos por score de anomalía (promedio)",
            )
            fig_dist.update_layout(
                xaxis_title="Distrito",
                yaxis_title="Score de anomalía (↓ = más atípico)",
            )
            st.plotly_chart(fig_dist, use_container_width=True)
                        # Comparación Isolation Forest vs MAD por distrito (si existen columnas)
            if {"provincia", "canton", "distrito", "anomaly_score", "mad_score", "es_anomalia_mad"}.issubset(df_d.columns):
                comp_dist = (
                    df_d.groupby(["provincia", "canton", "distrito"])
                    .agg(
                        anomaly_score_mean=("anomaly_score", "mean"),
                        mad_score_mean=("mad_score", "mean"),
                        mad_anom_frac=("es_anomalia_mad", "mean"),
                    )
                    .reset_index()
                )
                comp_dist["mad_anom_frac"] = (comp_dist["mad_anom_frac"] * 100).round(2)
                st.markdown(
                        """
                    **¿Qué es MAD?**

                    El método MAD (*Median Absolute Deviation*) es un algoritmo estadístico robusto para detectar valores atípicos.
                    En lugar de usar la media y la desviación estándar, calcula:

                    - La **mediana** de los datos.
                    - La desviación absoluta de cada valor respecto a esa mediana.
                    - La **mediana de esas desviaciones** (MAD).

                    A partir de eso se construye un puntaje de atipicidad (*score MAD*):  
                    valores más altos indican que una provincia, cantón o distrito se aleja más del patrón “normal”
                    de comportamiento patrimonial y registral.
                    """
                )
                st.markdown("### Comparación por distrito: Isolation Forest vs MAD")
                st.dataframe(
                    comp_dist.sort_values("mad_score_mean", ascending=False).head(top_n)
                )
                st.subheader("Distribución de MAD Score")

                top10_dist = (
                    df_d.groupby("distrito")["mad_score"]
                    .mean()
                    .sort_values(ascending=False)
                    .head(10)
                    .index.tolist()
                )

                df_top_dist = df_d[df_d["distrito"].isin(top10_dist)]

                fig_mad_dist = px.box(
                    df_top_dist,
                    x="distrito",
                    y="mad_score",
                    color="distrito",
                    points="all",
                    title="Distribución de MAD Score por distrito",
                )

                fig_mad_dist.update_layout(xaxis_title="Distrito", yaxis_title="MAD Score")
                st.plotly_chart(fig_mad_dist, use_container_width=True)


if run_btn:
    try:
        with st.spinner("Ejecutando script de detección de anomalías..."):
            # Localizar script en distintas ubicaciones posibles
            script_candidates = [
                BASE_DIR / "scripts" / "deteccion_anomala.py",
                BASE_DIR / "scripts" / "deteccion_anomalias.py",
                Path("/scripts") / "deteccion_anomala.py",
                Path("/scripts") / "deteccion_anomalias.py",
            ]
            script_path = next((p for p in script_candidates if p.exists()), None)
            if script_path is None:
                raise FileNotFoundError(
                    "No se encontró el script en 'scripts/deteccion_anomala.py' ni 'scripts/deteccion_anomalias.py'."
                )

            # Elegir cwd para que ./data sea accesible y los outputs se persistan
            if (BASE_DIR / "data").exists() and (BASE_DIR / "scripts").exists():
                cwd_run = BASE_DIR
            elif Path("/app/data").exists() and Path("/scripts").exists():
                cwd_run = Path("/app")
            elif Path("/data").exists() and Path("/scripts").exists():
                cwd_run = Path("/")
            else:
                cwd_run = BASE_DIR

            # Si el usuario subió archivos pero no los reemplazó, usaremos temporales y los pasamos como argumentos
            tmp_files = []
            cmd = [sys.executable, str(script_path)]
            try:
                tmp_dir = Path(
                    tempfile.mkdtemp(prefix="tmp_uploads_", dir=str(BASE_DIR / "data"))
                )
            except Exception:
                tmp_dir = Path(tempfile.mkdtemp(prefix="tmp_uploads_"))

            try:
                if uploaded_cgr is not None:
                    tmp_cgr = tmp_dir / "uploaded_cgr.csv"
                    tmp_cgr.write_bytes(uploaded_cgr.getvalue())
                    tmp_files.append(tmp_cgr)
                    cmd += ["--cgr", str(tmp_cgr)]
                if uploaded_reg is not None:
                    tmp_reg = tmp_dir / "uploaded_registro.csv"
                    tmp_reg.write_bytes(uploaded_reg.getvalue())
                    tmp_files.append(tmp_reg)
                    cmd += ["--registro", str(tmp_reg)]

                result = subprocess.run(
                    cmd, cwd=str(cwd_run), capture_output=True, text=True
                )
            finally:
                # Limpiar archivos temporales
                try:
                    if tmp_dir.exists():
                        shutil.rmtree(tmp_dir, ignore_errors=True)
                except Exception:
                    pass
            if result.returncode != 0:
                raise RuntimeError(f"Fallo al ejecutar el script:\n{result.stderr}")

            # Cargar resultados generados por el script desde el cwd usado
            out_analisis = (
                (cwd_run / ANOMALIES_FILENAME)
                if (cwd_run / ANOMALIES_FILENAME).exists()
                else _find_existing_file(ANOMALIES_FILENAME)
            )
            out_ranking = (
                (cwd_run / RANKING_FILENAME)
                if (cwd_run / RANKING_FILENAME).exists()
                else _find_existing_file(RANKING_FILENAME)
            )
            if out_analisis is None or out_ranking is None:
                raise FileNotFoundError(
                    "El script terminó sin generar los archivos de salida esperados."
                )
            df_results = pd.read_csv(str(out_analisis))
            ranking = pd.read_csv(str(out_ranking))

        st.success("Análisis completado — resultados guardados en CSV")
        st.subheader("Rankings territoriales")
        tab_prov2, tab_cant2, tab_dist2 = st.tabs(
            ["Provincias", "Cantones", "Distritos"]
        )

        # Provincias
        with tab_prov2:
            years_new = (
                sorted([int(y) for y in df_results["anio"].dropna().unique()])
                if "anio" in df_results.columns
                else []
            )
            ranking_to_show = ranking.copy()
            if years_new:
                year_filter = st.selectbox(
                    "Filtrar por año (opcional)",
                    ["Todos"] + years_new,
                    index=0,
                    key="year_after_run",
                )
                if year_filter != "Todos":
                    df_year = df_results[df_results["anio"] == int(year_filter)]
                    ranking_to_show = (
                        df_year.groupby("provincia")["anomaly_score"]
                        .mean()
                        .sort_values()
                        .reset_index()
                    )
                    ranking_to_show["ranking"] = range(1, len(ranking_to_show) + 1)

            if order_opt == "Más atípicas primero":
                ranking_to_show = (
                    ranking_to_show.sort_values(["ranking"])
                    if "ranking" in ranking_to_show.columns
                    else ranking_to_show.sort_values("anomaly_score")
                )
            else:
                ranking_to_show = ranking_to_show.sort_values("provincia")
            ranking_view = ranking_to_show.head(top_n)

            st.dataframe(ranking_view)
            fig_rank = px.bar(
                ranking_view,
                x="provincia",
                y="anomaly_score",
                color="anomaly_score",
                color_continuous_scale="RdYlGn",
                text="ranking" if "ranking" in ranking_view.columns else None,
                title="Ranking de provincias por score de anomalía (promedio)",
            )
            fig_rank.update_layout(
                xaxis_title="Provincia",
                yaxis_title="Score de anomalía (↓ = más atípico)",
            )
            st.plotly_chart(fig_rank, use_container_width=True)

            # Mapa por provincia (usar todo el ranking para asegurar cobertura completa)
            try:
                render_province_map(ranking, titulo="Mapa de anomalías por provincia")
            except Exception:
                pass

        # Cantones
        with tab_cant2:
            prov_filter2 = st.selectbox(
                "Filtrar por provincia (opcional)",
                ["Todas"] + sorted(df_results["provincia"].unique().tolist()),
                key="prov_cant_after",
            )
            df_c2 = df_results.copy()
            if prov_filter2 != "Todas":
                df_c2 = df_c2[df_c2["provincia"] == prov_filter2]
            if year_filter and year_filter != "Todos" and "anio" in df_c2.columns:
                df_c2 = df_c2[df_c2["anio"] == int(year_filter)]
            rk_c2 = (
                df_c2.groupby(["provincia", "canton"])["anomaly_score"]
                .mean()
                .sort_values()
                .reset_index()
            )
            rk_c2["ranking"] = range(1, len(rk_c2) + 1)
            rk_c2_view = rk_c2.head(top_n)
            st.dataframe(rk_c2_view)
            fig_c2 = px.bar(
                rk_c2_view,
                x="canton",
                y="anomaly_score",
                color="anomaly_score",
                color_continuous_scale="RdYlGn",
                text="ranking",
                title="Ranking de cantones por score de anomalía (promedio)",
            )
            fig_c2.update_layout(
                xaxis_title="Cantón", yaxis_title="Score de anomalía (↓ = más atípico)"
            )
            st.plotly_chart(fig_c2, use_container_width=True)

        # Distritos
        with tab_dist2:
            prov_filter3 = st.selectbox(
                "Filtrar por provincia (opcional)",
                ["Todas"] + sorted(df_results["provincia"].unique().tolist()),
                key="prov_dist_after",
            )
            df_d2 = df_results.copy()
            if prov_filter3 != "Todas":
                df_d2 = df_d2[df_d2["provincia"] == prov_filter3]
            if year_filter and year_filter != "Todos" and "anio" in df_d2.columns:
                df_d2 = df_d2[df_d2["anio"] == int(year_filter)]
            rk_d2 = (
                df_d2.groupby(["provincia", "canton", "distrito"])["anomaly_score"]
                .mean()
                .sort_values()
                .reset_index()
            )
            rk_d2["ranking"] = range(1, len(rk_d2) + 1)
            rk_d2_view = rk_d2.head(top_n)
            st.dataframe(rk_d2_view)
            fig_d2 = px.bar(
                rk_d2_view,
                x="distrito",
                y="anomaly_score",
                color="anomaly_score",
                color_continuous_scale="RdYlGn",
                text="ranking",
                title="Ranking de distritos por score de anomalía (promedio)",
            )
            fig_d2.update_layout(
                xaxis_title="Distrito",
                yaxis_title="Score de anomalía (↓ = más atípico)",
            )
            st.plotly_chart(fig_d2, use_container_width=True)

            st.subheader(
                "Top distritos con mayor anomalía (valores más bajos = más atípicos)"
            )
            top_distritos = df_results.sort_values("anomaly_score").head(20)[
                ["provincia", "canton", "distrito", "anio", "anomaly_score"]
            ]
            st.dataframe(top_distritos)

    except FileNotFoundError as e:
        st.error(f"No se pudo ejecutar el análisis: {e}")
    except Exception as e:
        st.error(f"Error durante el análisis: {e}")

# =========================
# SECCIÓN 4: CONCLUSIONES
# =========================
st.header("Conclusiones preliminares")

st.markdown(
    """
- Los datos sintéticos permiten visualizar la estructura esperada del sistema patrimonial.
- El modelo de detección de anomalías puede identificar casos con incrementos patrimoniales atípicos.
- Al incorporar los datos reales, el dashboard mostrará alertas automáticas y métricas agregadas por región.
"""
)

st.caption(
    "© 2025 Proyecto Transparencia Patrimonial CR – Instituto Tecnológico de Costa Rica"
)