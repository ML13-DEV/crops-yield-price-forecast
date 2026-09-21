import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from prediccion_campo import calcular_prediccion_campo

RUTA_LOTES = "lotes_productor.csv"
RUTA_NDVI_DIR = Path("ndvi_lotes")
MODELS_DIR = Path("../models/")
RUTA_VENTANAS = "../data/raw/ventanas_fenologicas.csv"
RUTA_MAPA_ZONA_PAS = "mapa_departamento_zona.csv"
RUTA_MAPA_ZONA_MANI = "zona_mani_por_departamento.csv"


def cargar_modelo(nombre_pkl, nombre_features):
    modelo = joblib.load(MODELS_DIR / nombre_pkl)
    with open(MODELS_DIR / nombre_features, "r", encoding="utf-8") as f:
        features = json.load(f)
    return modelo, features


def main():
    lotes = pd.read_csv(RUTA_LOTES)
    lotes["fecha_siembra"] = pd.to_datetime(lotes["fecha_siembra"])

    modelo_con_te, features_con_te = cargar_modelo(
        "modelo_rendimiento_rf.pkl", "features_modelo_rendimiento.json"
    )
    modelo_sin_te, features_sin_te = cargar_modelo(
        "modelo_rendimiento_rf_sin_TE.pkl", "features_sin_TE.json"
    )

    tabla_te = pd.read_csv(MODELS_DIR / "promedio_rendimiento_departamento_cultivo.csv")
    tabla_te["Departamento_id"] = tabla_te["Departamento_id"].apply(lambda x: str(int(float(x))).zfill(5))
    tabla_te = tabla_te.set_index(["Departamento_id", "Cultivo"])["Rinde(qq/Ha)"]
    promedio_global = tabla_te.mean()

    df_ventanas = pd.read_csv(RUTA_VENTANAS)
    mapa_zona_pas = pd.read_csv(RUTA_MAPA_ZONA_PAS)
    mapa_zona_pas["departamento_id"] = mapa_zona_pas["departamento_id"].apply(lambda x: str(int(float(x))).zfill(5))
    mapa_zona_mani = pd.read_csv(RUTA_MAPA_ZONA_MANI)
    mapa_zona_mani["departamento_id"] = mapa_zona_mani["departamento_id"].apply(lambda x: str(int(float(x))).zfill(5))

    resultados = []
    for _, lote in lotes.iterrows():
        lote_id = str(lote["lote_id"])
        archivo_ndvi = RUTA_NDVI_DIR / f"lote_{lote_id}.csv"
        if not archivo_ndvi.exists():
            continue
        df_ndvi_punto = pd.read_csv(archivo_ndvi, parse_dates=["fecha_media"])

        anio_inicio = lote["fecha_siembra"].year if lote["fecha_siembra"].month > 6 else lote["fecha_siembra"].year - 1
        campania = f"{anio_inicio}/{anio_inicio + 1}"

        try:
            pred_con_te, depto_id, depto_nombre, zona, _ = calcular_prediccion_campo(
                lote["lat"], lote["lon"], lote["cultivo"], campania, lote["fecha_siembra"],
                df_ventanas, df_ndvi_punto, modelo_con_te, features_con_te,
                mapa_zona_pas, mapa_zona_mani, tabla_te, promedio_global,
            )
            pred_sin_te, _, _, _, _ = calcular_prediccion_campo(
                lote["lat"], lote["lon"], lote["cultivo"], campania, lote["fecha_siembra"],
                df_ventanas, df_ndvi_punto, modelo_sin_te, features_sin_te,
                mapa_zona_pas, mapa_zona_mani, tabla_te, promedio_global,
            )
        except Exception as e:
            print(f"Lote {lote_id}: ERROR -- {e}")
            continue

        real = lote["rinde_real_qqha"]
        resultados.append({
            "lote_id": lote_id, "cultivo": lote["cultivo"], "real": real,
            "pred_con_TE": round(pred_con_te, 1), "error_con_TE": round(abs(pred_con_te - real), 1),
            "pred_sin_TE": round(pred_sin_te, 1), "error_sin_TE": round(abs(pred_sin_te - real), 1),
        })

    df_resultados = pd.DataFrame(resultados)
    if df_resultados.empty:
        print("Ningún lote se pudo procesar.")
        return

    print(df_resultados.to_string(index=False))
    print()
    print(f"MAE con TE:  {df_resultados['error_con_TE'].mean():.2f} qq/ha | "
          f"desvío de las predicciones: {df_resultados['pred_con_TE'].std():.1f}")
    print(f"MAE sin TE:  {df_resultados['error_sin_TE'].mean():.2f} qq/ha | "
          f"desvío de las predicciones: {df_resultados['pred_sin_TE'].std():.1f}")
    print(f"Desvío de los valores REALES: {df_resultados['real'].std():.1f}")

    df_resultados.to_csv("comparacion_con_sin_TE.csv", index=False)


if __name__ == "__main__":
    main()
