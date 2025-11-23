import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# =========================
# Utilidades Modified Z-Score
# =========================

def modified_z_score(series: pd.Series) -> pd.Series:
    """Calcula Modified Z-Score para una serie (vectorizada).
    MZ_i = 0.6745 * (x_i - mediana) / MAD
    Maneja MAD=0 devolviendo 0s para evitar división por cero.
    """
    x = series.astype(float).values
    median = np.median(x)
    abs_dev = np.abs(x - median)
    mad = np.median(abs_dev)
    if mad == 0:
        return pd.Series(np.zeros_like(x), index=series.index)
    mz = 0.6745 * (x - median) / mad
    return pd.Series(mz, index=series.index)

# =========================
# Carga y agregación de datos
# =========================

def load_cgr(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Tipos esenciales
    df["activos_totales"] = df["activos_totales"].astype(float)
    df["pasivos_totales"] = df["pasivos_totales"].astype(float)
    df["anio_declaracion"] = df["anio_declaracion"].astype(int)
    # Patrimonio neto
    df["patrimonio_neto"] = df["activos_totales"] - df["pasivos_totales"]
    agg = (df.groupby(["anio_declaracion", "provincia", "canton", "distrito"])\
             .agg(
                patrimonio_neto=("patrimonio_neto", "sum"),
                activos_totales=("activos_totales", "sum"),
                pasivos_totales=("pasivos_totales", "sum"),
                cant_funcionarios=("identificador_anonimo_declarante", "count")
             )
             .reset_index())
    agg = agg.rename(columns={"anio_declaracion": "anio"})
    return agg

def load_registro(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Derivar año si sólo hay periodo
    if "anio" not in df.columns and "periodo" in df.columns:
        df["anio"] = pd.to_datetime(df["periodo"], errors="coerce").dt.year
    else:
        df["anio"] = df["anio"].astype(int)
    df["valor_declarado"] = df["valor_declarado"].astype(float)
    agg = (df.groupby(["anio", "provincia", "canton", "distrito"])\
             .agg(
                valor_total_bienes=("valor_declarado", "sum"),
                valor_medio_bienes=("valor_declarado", "mean"),
                cant_actos_registrales=("tipo_acto_registral", "count")
             )
             .reset_index())
    return agg

def build_features(cgr: pd.DataFrame, reg: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(cgr, reg, on=["anio", "provincia", "canton", "distrito"], how="inner", validate="one_to_one")
    # Ratios robustos (evitar división por cero)
    df["ratio_bienes_vs_patrimonio"] = df["valor_total_bienes"] / df["patrimonio_neto"].replace(0, np.nan)
    df["ratio_bienes_vs_activos"] = df["valor_total_bienes"] / df["activos_totales"].replace(0, np.nan)
    df["densidad_actos_por_func"] = df["cant_actos_registrales"] / df["cant_funcionarios"].replace(0, np.nan)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    # Variaciones por provincia (orden temporal)
    df = df.sort_values(["provincia", "anio"])  # orden estable
    for col in ["valor_total_bienes", "patrimonio_neto", "activos_totales", "pasivos_totales"]:
        df[f"var_{col}"] = df.groupby("provincia")[col].pct_change().fillna(0)
    df["dif_var_bienes_patrimonio"] = df["var_valor_total_bienes"] - df["var_patrimonio_neto"]
    return df

# =========================
# Cálculo Modified Z multivariante
# =========================

def compute_modified_z_scores(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    result = df.copy()
    mz_cols = []
    for col in feature_cols:
        mz_col = f"mz_{col}"
        result[mz_col] = modified_z_score(result[col])
        mz_cols.append(mz_col)
    # Valores absolutos
    abs_matrix = result[mz_cols].abs().values
    # Máximo absoluto
    result["max_abs_mz"] = abs_matrix.max(axis=1)
    # Promedio top3
    sorted_abs = np.sort(abs_matrix, axis=1)[:, ::-1]  # descendente
    if len(mz_cols) >= 3:
        top3 = sorted_abs[:, :3]
        result["avg_top3_abs_mz"] = top3.mean(axis=1)
        result["combined_score"] = result["avg_top3_abs_mz"]
    else:
        result["combined_score"] = result["max_abs_mz"]
    return result

# =========================
# Ranking y exportación
# =========================

def export_results(df: pd.DataFrame, out_dir: Path, threshold: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    detalle_path = out_dir / "mzscore_detalle.csv"
    ranking_path = out_dir / "mzscore_ranking_provincias.csv"
    df["anomaly_flag"] = (df["combined_score"].abs() > threshold).astype(int)
    df.to_csv(detalle_path, index=False)
    ranking = (df.groupby("provincia")["combined_score"].mean().sort_values(ascending=False).reset_index())
    ranking["ranking"] = np.arange(1, len(ranking) + 1)
    ranking.to_csv(ranking_path, index=False)
    return detalle_path, ranking_path

# =========================
# CLI principal
# =========================

def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline Modified Z-Score para anomalías territoriales")
    parser.add_argument("--cgr", type=str, default="data/synthetic_cgr_declaraciones.csv", help="Ruta CSV CGR")
    parser.add_argument("--registro", type=str, default="data/synthetic_registro_nacional.csv", help="Ruta CSV Registro Nacional")
    parser.add_argument("--out-dir", type=str, default=".", help="Directorio de salida")
    parser.add_argument("--threshold", type=float, default=3.5, help="Umbral |MZ| para anomaly_flag")
    parser.add_argument("--features", nargs="*", default=[
        "ratio_bienes_vs_patrimonio",
        "ratio_bienes_vs_activos",
        "densidad_actos_por_func",
        "var_valor_total_bienes",
        "var_patrimonio_neto",
        "dif_var_bienes_patrimonio"
    ], help="Lista de columnas de features a incluir en Modified Z-Score")
    return parser.parse_args()


def main():
    args = parse_args()
    cgr_path = Path(args.cgr)
    reg_path = Path(args.registro)
    out_dir = Path(args.out_dir)

    if not cgr_path.exists() or not reg_path.exists():
        raise FileNotFoundError("No se encuentran los CSV de entrada especificados.")

    cgr = load_cgr(cgr_path)
    reg = load_registro(reg_path)
    df = build_features(cgr, reg)

    # Verificar que todas las features solicitadas existen
    missing = [f for f in args.features if f not in df.columns]
    if missing:
        raise ValueError(f"Features faltantes en dataset agregado: {missing}")

    mz_df = compute_modified_z_scores(df, args.features)
    detalle_path, ranking_path = export_results(mz_df, out_dir, args.threshold)

    print("Proceso completado.")
    print(f"Detalle: {detalle_path}")
    print(f"Ranking provincias: {ranking_path}")


if __name__ == "__main__":
    main()
