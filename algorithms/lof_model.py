from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

def detect_anomalies_lof(df, features, n_neighbors=20, contamination=0.05):
    """
    Aplica Local Outlier Factor (LOF) sobre el dataframe territorial.
    
    Retorna:
      - df con columnas 'lof_score' y 'lof_flag'
      - modelo entrenado
    """
    X = df[features].fillna(0).values
    
    # Escalamiento
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=False  # porque predecimos sobre el mismo set
    )
    
    # LOF produce:
    # - negative_outlier_factor_: valores más pequeños = más anomalía
    preds = lof.fit_predict(X_scaled)
    scores = lof.negative_outlier_factor_
    
    df = df.copy()
    df["lof_score"] = -scores  # invertimos: mayor = más anomalía (coherente con IF)
    df["lof_flag"] = (preds == -1).astype(int)
    
    return df, lof

'''
Features recomendados (idénticos a los de Isolation Forest) 

features = [
    "ratio_bienes_vs_patrimonio",
    "ratio_bienes_vs_activos",
    "densidad_actos_por_func",
    "var_valor_total_bienes",
    "var_patrimonio_neto",
    "dif_var_bienes_patrimonio"
]

'''

'''
Cómo integrarlo al pipeline:

df_lof, lof_model = detect_anomalies_lof(df_merged, features)
df_lof.head()

'''