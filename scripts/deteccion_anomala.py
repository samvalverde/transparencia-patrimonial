import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

df_registro = pd.read_csv("./data/synthetic_registro_nacional.csv")
df_cgr = pd.read_csv("./data/synthetic_cgr_declaraciones.csv")

# REGISTRO NACIONAL

# transformaciones de datos del registro nacional
# asignar tipos de datos correspondientes
df_registro["periodo"] = df_registro["periodo"].astype(str)
df_registro["provincia"] = df_registro["provincia"].astype(str)
df_registro["canton"] = df_registro["canton"].astype(str)
df_registro["distrito"] = df_registro["distrito"].astype(str)
df_registro["tipo_acto_registral"] = df_registro["tipo_acto_registral"].astype(str)
df_registro["tipo_bien"] = df_registro["tipo_bien"].astype(str)
df_registro["valor_declarado"] = df_registro["valor_declarado"].astype(float)
df_registro["categoria_adquirente"] = df_registro["categoria_adquirente"].astype(str)
df_registro["entidad_publica_participante"] = df_registro["entidad_publica_participante"].astype(str)
df_registro["anio"] = df_registro["anio"].astype(int)
df_registro["mes"] = df_registro["mes"].astype(int)

# dataset agregados registro nacional
df_registro_agg = df_registro.groupby(["anio", "provincia", "canton", "distrito"]).agg(
    valor_total_bienes=("valor_declarado", "sum"),
    valor_medio_bienes=("valor_declarado", "mean"),
    cant_actos_registrales=("tipo_acto_registral", "count"),
).reset_index()

columns_round = ["valor_total_bienes", "valor_medio_bienes"]
df_registro_agg[columns_round] = df_registro_agg[columns_round].round(4)


# CONTRALORIA GENERAL DE LA REPÚBLICA

# transformaciones de datos de las declaraciones de la CGR
# asignar tipos de datos correspondientes
df_cgr["anio_declaracion"] = df_cgr["anio_declaracion"].astype(int)
df_cgr["identificador_anonimo_declarante"] = df_cgr["identificador_anonimo_declarante"].astype(str)
df_cgr["institucion"] = df_cgr["institucion"].astype(str)
df_cgr["tipo_cargo"] = df_cgr["tipo_cargo"].astype(str)
df_cgr["activos_totales"] = df_cgr["activos_totales"].astype(float)
df_cgr["pasivos_totales"] = df_cgr["pasivos_totales"].astype(float)
df_cgr["provincia"] = df_cgr["provincia"].astype(str)
df_cgr["canton"] = df_cgr["canton"].astype(str)
df_cgr["distrito"] = df_cgr["distrito"].astype(str)
df_cgr["distrib_inmuebles_%"] = df_cgr["distrib_inmuebles_%"].astype(float)
df_cgr["distrib_vehículos_%"] = df_cgr["distrib_vehículos_%"].astype(float)
df_cgr["distrib_depósitos_%"] = df_cgr["distrib_depósitos_%"].astype(float)
df_cgr["distrib_inversiones_%"] = df_cgr["distrib_inversiones_%"].astype(float)
df_cgr["distrib_otros_%"] = df_cgr["distrib_otros_%"].astype(float)


# calcular patrimonio neto
df_cgr["patrimonio_neto"] = df_cgr["activos_totales"] - df_cgr["pasivos_totales"]

df_cgr_agg = df_cgr.groupby(["anio_declaracion", "provincia", "canton", "distrito"]).agg(
    patrimonio_neto=("patrimonio_neto", "sum"),
    patrimonio_medio=("patrimonio_neto", "mean"),
    activos_totales=("activos_totales", "sum"),
    activos_medios=("activos_totales", "mean"),
    pasivos_totales=("pasivos_totales", "sum"),
    pasivos_medios=("pasivos_totales", "mean"),
    cant_funcionarios=("identificador_anonimo_declarante", "count"),
).reset_index()

# Round the relevant columns to 6 decimal places
cols_to_round = ["patrimonio_neto", "patrimonio_medio", "activos_totales", "activos_medios", "pasivos_totales", "pasivos_medios"]
df_cgr_agg[cols_to_round] = df_cgr_agg[cols_to_round].round(4)



# union de los dos datasets agregados
df_registro_cgr_anio_provincia_canton_distrito = pd.merge(
    df_cgr_agg,
    df_registro_agg,
    left_on=["anio_declaracion", "provincia", "canton", "distrito"],
    right_on=["anio", "provincia", "canton", "distrito"],
    how="inner"
)


# estructura para el modelo de Isolation Forest

# Renombrar columnas clave para consistencia
df_cgr_agg = df_cgr_agg.rename(columns={"anio_declaracion": "anio"})
df_registro_agg = df_registro_agg.copy()

# Merge por año y provincia (sin institución)
df_merged = pd.merge(
    df_cgr_agg,
    df_registro_agg,
    on=["anio", "provincia", "canton", "distrito"],
    how="inner"
)

# Ratios de consistencia patrimonial
df_merged["ratio_bienes_vs_patrimonio"] = df_merged["valor_total_bienes"] / df_merged["patrimonio_neto"]
df_merged["ratio_bienes_vs_activos"]     = df_merged["valor_total_bienes"] / df_merged["activos_totales"]
df_merged["densidad_actos_por_func"]     = df_merged["cant_actos_registrales"] / df_merged["cant_funcionarios"]

# Limpieza de infinitos o nulos
df_merged = df_merged.replace([np.inf, -np.inf], np.nan).fillna(0)

# Variaciones por provincia
df_merged = df_merged.sort_values(["provincia", "anio"])

for col in ["valor_total_bienes", "patrimonio_neto", "activos_totales", "pasivos_totales"]:
    df_merged[f"var_{col}"] = df_merged.groupby("provincia")[col].pct_change()

# Diferencia entre variaciones (patrimonio vs bienes)
df_merged["dif_var_bienes_patrimonio"] = df_merged["var_valor_total_bienes"] - df_merged["var_patrimonio_neto"]


# Selección de features para el modelo
features = [
    "ratio_bienes_vs_patrimonio",
    "ratio_bienes_vs_activos",
    "densidad_actos_por_func",
    "var_valor_total_bienes",
    "var_patrimonio_neto",
    "dif_var_bienes_patrimonio"
]

X = df_merged[features].fillna(0)
X_scaled = MinMaxScaler().fit_transform(X)

# Modelo de anomalías
iso = IsolationForest(
    n_estimators=200, 
    contamination=0.05, 
    random_state=42
    )
df_merged["score_anomalia"] = iso.fit_predict(X_scaled)
df_merged["anomaly_score"] = iso.decision_function(X_scaled)

# Ranking de anomalías por distrito
ranking_distrito = (
    df_merged.groupby("distrito")["anomaly_score"]
    .mean()
    .sort_values()
    .reset_index()
)
ranking_distrito["ranking"] = np.arange(1, len(ranking_distrito) + 1)

# Ranking de anomalías por provincia
ranking = (
    df_merged.groupby("provincia")["anomaly_score"]
    .mean()
    .sort_values()
    .reset_index()
)
ranking["ranking"] = np.arange(1, len(ranking) + 1)

# Guardar resultados
df_merged.to_csv("analisis_anomalias_territorial.csv", index=False)
ranking.to_csv("ranking_provincias_anomalias.csv", index=False)