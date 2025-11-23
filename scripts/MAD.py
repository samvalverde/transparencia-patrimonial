# mad.py
# ===============================================
# MÉTODO ESTADÍSTICO ROBUSTO: MAD (Median Absolute Deviation)
# ===============================================
# Este módulo implementa el cálculo de MAD robusto como método de detección
# de anomalías, diseñado para ser totalmente independiente de Streamlit.
#
# Compatible con el pipeline:
# - aggregate_cgr()
# - aggregate_registro()
# - build_features_and_merge()
#
# Uso:
#   from mad import run_mad
#   df_mad, ranking_mad = run_mad(df_merged)
#
# Devuelve:
#   df_mad : df con columnas mad_score, es_anomalia_mad
#   ranking_mad : ranking por provincia basado en mad_score promedio
# ===============================================

import numpy as np
import pandas as pd
from pathlib import Path

# -------------------------------------------------------------------------
# Cálculo robusto del MAD univariado para una serie numérica
# -------------------------------------------------------------------------
def modified_zscore_series(series: pd.Series) -> np.ndarray:
    """
    Calcula el Modified Z-Score (Iglewicz & Hoaglin) de una serie numérica.
    Si la MAD es 0 (todos iguales), devuelve ceros.
    """
    x = series.to_numpy(dtype=float)
    median = np.median(x)
    mad = np.median(np.abs(x - median))

    if mad == 0 or np.isnan(mad):
        return np.zeros_like(x)

    # Iglewicz & Hoaglin: factor 0.6745 para escala consistente
    return 0.6745 * (x - median) / mad


# -------------------------------------------------------------------------
# Agregar MAD univariado por feature
# -------------------------------------------------------------------------
def compute_mad_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Agrega columnas mz_<feature> para cada variable en feature_cols.
    También agrega mad_score = promedio del valor absoluto de todos los MZ-scores.
    """
    df = df.copy()

    mz_cols = []
    for col in feature_cols:
        mzcol = f"mz_{col}"
        df[mzcol] = modified_zscore_series(df[col])
        mz_cols.append(mzcol)

    # Puntaje agregado (score MAD robusto)
    df["mad_score"] = df[mz_cols].abs().mean(axis=1)

    return df


# -------------------------------------------------------------------------
# Etiquetado de anomalías por MAD
# -------------------------------------------------------------------------
def classify_mad(df: pd.DataFrame, threshold: float = 3.5) -> pd.DataFrame:
    """
    Etiqueta como anómala cada fila cuyo mad_score excede el threshold.
    """
    df = df.copy()
    df["es_anomalia_mad"] = df["mad_score"] > threshold
    return df


# -------------------------------------------------------------------------
# Ranking por provincia (o cualquier nivel territorial)
# -------------------------------------------------------------------------
def build_mad_ranking(df: pd.DataFrame, level: str = "provincia") -> pd.DataFrame:
    """
    Construye un ranking de unidades territoriales según mad_score promedio.
    Por defecto usa la columna 'provincia'.
    """
    if level not in df.columns:
        raise KeyError(f"La columna {level} no se encuentra en el DataFrame.")

    ranking = (
        df.groupby(level)["mad_score"]
        .mean()
        .sort_values(ascending=False)  # mayor MAD = más extremo
        .reset_index()
    )
    ranking["ranking_mad"] = range(1, len(ranking) + 1)
    return ranking


# -------------------------------------------------------------------------
# Método principal para integración en el pipeline
# -------------------------------------------------------------------------
def run_mad(df_merged: pd.DataFrame, output_dir: str | Path | None = None,
            threshold: float = 3.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calcula MAD, etiquetas y ranking para el DF consolidado (df_merged)
    que proviene de build_features_and_merge().

    Parámetros:
        df_merged: DataFrame con variables territoriales y features numéricas.
        output_dir: carpeta donde guardar CSV opcionalmente.
        threshold: umbral para clasificar anomalías (default = 3.5).

    Retorna:
        df_mad : DataFrame con columnas mz_*, mad_score, es_anomalia_mad.
        ranking_mad : ranking por provincia.
    """

    # Mismas features que usa Isolation Forest en tu pipeline
    feature_cols = [
        "ratio_bienes_vs_patrimonio",
        "ratio_bienes_vs_activos",
        "densidad_actos_por_func",
        "var_valor_total_bienes",
        "var_patrimonio_neto",
        "dif_var_bienes_patrimonio",
    ]

    # Asegurar que estén presentes
    for c in feature_cols:
        if c not in df_merged.columns:
            raise KeyError(f"Falta la columna requerida para MAD: {c}")

    # 1. Calcular MZ-scores y score MAD
    df_mad = compute_mad_features(df_merged, feature_cols)

    # 2. Clasificar anomalías
    df_mad = classify_mad(df_mad, threshold=threshold)

    # 3. Ranking territorial por provincia
    ranking_mad = build_mad_ranking(df_mad, level="provincia")

    # 4. Guardar (opcional)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df_mad.to_csv(output_dir / "analisis_mad_territorial.csv", index=False)
        ranking_mad.to_csv(output_dir / "ranking_provincias_mad.csv", index=False)

    return df_mad, ranking_mad
