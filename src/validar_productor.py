import json
import joblib
import pandas as pd
from pathlib import Path
from prediccion_campo import calcular_prediccion_campo

RUTA_LOTES = "lotes_productor.csv"
RUTA_NDVI_DIR = Path("ndvi_lotes")
MODELS_DIR = Path("../models/")
RUTA_VENTANAS = "../data/raw/ventanas_fenologicas.csv"
RUTA_MAPA_ZONA_PAS = "mapa_departamento_zona.csv"
RUTA_MAPA_ZONA_MANI = "zona_mani_por_departamento.csv"


def main():
    lotes = pd.read_csv(RUTA_LOTES)
    lotes["fecha_siembra"] = pd.to_datetime(lotes["fecha_siembra"])

    modelo_final = joblib.load(MODELS_DIR / "modelo_rendimiento_rf.pkl")
    with open(MODELS_DIR / "features_modelo_rendimiento.json", "r", encoding="utf-8") as f:
        features = json.load(f)

    tabla_te = pd.read_csv(
        MODELS_DIR / "promedio_rendimiento_departamento_cultivo.csv",
    )
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
            print(f"Lote {lote_id}: sin NDVI descargado todavía, salteando.")
            continue

        df_ndvi_punto = pd.read_csv(archivo_ndvi, parse_dates=["fecha_media"])

        anio_inicio = lote["fecha_siembra"].year if lote["fecha_siembra"].month > 6 else lote["fecha_siembra"].year - 1
        campania = f"{anio_inicio}/{anio_inicio + 1}"

        try:
            prediccion, depto_id, depto_nombre, zona, _ = calcular_prediccion_campo(
                lote["lat"], lote["lon"], lote["cultivo"], campania, lote["fecha_siembra"],
                df_ventanas, df_ndvi_punto, modelo_final, features,
                mapa_zona_pas, mapa_zona_mani, tabla_te, promedio_global,
            )
        except Exception as e:
            print(f"Lote {lote_id}: ERROR -- {e}")
            continue

        real = lote["rinde_real_qqha"]
        error_abs = abs(prediccion - real)
        error_pct = 100 * error_abs / real

        resultados.append({
            "lote_id": lote_id, "cultivo": lote["cultivo"],
            "departamento": depto_nombre, "zona": zona,
            "prediccion": round(prediccion, 1), "real": real,
            "error_abs": round(error_abs, 1), "error_pct": round(error_pct, 1),
        })

    df_resultados = pd.DataFrame(resultados)
    if df_resultados.empty:
        print("Ningún lote se pudo procesar -- revisá los errores de arriba antes de seguir.")
        return

    print(df_resultados.to_string(index=False))
    print()
    print(f"MAE sobre {len(df_resultados)} lotes: {df_resultados['error_abs'].mean():.2f} qq/ha")
    print(f"Error % promedio: {df_resultados['error_pct'].mean():.1f}%")

    df_resultados.to_csv("resultados_validacion_productor.csv", index=False)


if __name__ == "__main__":
    main()