# Proyecto: Evaluación de Modified Z-Score para Detección de Anomalías Patrimoniales

Este proyecto (para paper académico) implementa el método **Modified Z-Score** sobre los datos sintéticos del desafío PIDA con el fin de comparar en trabajos posteriores contra otros métodos (Z-Score clásico, Isolation Forest, Local Outlier Factor).

## 1. Objetivo
Detectar observaciones territoriales (año–provincia–cantón–distrito) atípicas en la evolución patrimonial y actividad registral mediante un indicador robusto basado en la mediana y el MAD.

## 2. Datos de entrada
Se reutilizan los mismos archivos sintéticos del repositorio raíz:
- `data/synthetic_cgr_declaraciones.csv`
- `data/synthetic_registro_nacional.csv`

## 3. Agregación y Features
Se construye una tabla unificada agregada por: `anio, provincia, canton, distrito`.
Del lado CGR:
- patrimonio_neto = activos_totales - pasivos_totales
- métricas agregadas (sum/mean) de patrimonio, activos, pasivos, cant_funcionarios.
Del lado Registro:
- valor_total_bienes, valor_medio_bienes, cant_actos_registrales.

Features derivadas para el MZ-Score:
- ratio_bienes_vs_patrimonio = valor_total_bienes / patrimonio_neto
- ratio_bienes_vs_activos = valor_total_bienes / activos_totales
- densidad_actos_por_func = cant_actos_registrales / cant_funcionarios
- var_valor_total_bienes (pct_change por provincia ordenada por año)
- var_patrimonio_neto
- var_activos_totales
- var_pasivos_totales
- dif_var_bienes_patrimonio = var_valor_total_bienes - var_patrimonio_neto

## 4. Modified Z-Score
Para cada feature numérica seleccionada se calcula:
```
MZ_i = 0.6745 * (x_i - mediana_x) / MAD_x
```
Donde:
- mediana_x: mediana de la serie
- MAD_x: mediana(|x_i - mediana_x|)
- 0.6745: factor que hace MAD comparable a la desviación estándar en datos normales

Umbral típico para marcar outliers: |MZ_i| > 3.5.

## 5. Combinación Multivariante
Se calcula para cada fila:
- `max_abs_mz`: máximo de |MZ| entre las features seleccionadas.
- `avg_top3_abs_mz`: promedio de las tres mayores magnitudes (si hay >=3 features).
- `combined_score`: por defecto `avg_top3_abs_mz` si hay >=3 features, otherwise `max_abs_mz`.
- `anomaly_flag`: 1 si `combined_score > threshold` (threshold por defecto 3.5), sino 0.

## 6. Salidas
Archivos generados en el directorio de salida (por defecto raíz del repo):
- `mzscore_detalle.csv`: dataset completo con columnas originales + MZ de cada feature + combined_score + anomaly_flag.
- `mzscore_ranking_provincias.csv`: promedio de combined_score por provincia y ranking (orden descendente, más alto = más atípico).

## 7. Uso
### Ejecución básica
```bash
python paper_mzscore/mzscore_pipeline.py
```
### Con argumentos explícitos
```bash
python paper_mzscore/mzscore_pipeline.py \
  --cgr data/synthetic_cgr_declaraciones.csv \
  --registro data/synthetic_registro_nacional.csv \
  --out-dir . \
  --threshold 3.5
```
### Seleccionar subconjunto de features
```bash
python paper_mzscore/mzscore_pipeline.py --features ratio_bienes_vs_patrimonio densidad_actos_por_func var_valor_total_bienes
```

## 8. Futuras Extensiones
- Implementar Z-Score clásico y comparar falsos positivos.
- Isolation Forest y LOF en la misma tabla para análisis de correlación entre métodos.
- Curvas de precisión vs cobertura usando datos con etiquetas simuladas.

## 9. Referencias
- Iglewicz, B., & Hoaglin, D. C. (1993). *How to Detect and Handle Outliers*.
- MAD robust scaling factor (0.6745) aplicado para normalizar respecto a distribución normal.

## 10. Licencia
Uso interno académico para elaboración de paper comparativo.
