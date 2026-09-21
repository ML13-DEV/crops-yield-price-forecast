import pandas as pd
from pathlib import Path

CLIMA_DIR = Path(r"..\data\raw\clima_departamentos")
FILL_VALUE = -999
parametros = ["PRECTOTCORR", "T2M", "T2M_MAX", "T2M_MIN", "ALLSKY_SFC_SW_DWN", "RH2M"]

resumen = []

for archivo in sorted(CLIMA_DIR.glob("clima_*.csv")):
    depto_id = archivo.stem.replace("clima_", "")
    df = pd.read_csv(archivo, index_col=0, parse_dates=True)

    n_fill = (df[parametros] == FILL_VALUE).sum().sum()

    rango_esperado = pd.date_range(df.index.min(), df.index.max(), freq="D")
    faltantes = rango_esperado.difference(df.index)

    nan_por_param = df[parametros].isna().sum()
    nan_por_param = nan_por_param[nan_por_param > 0]

    ok = (n_fill == 0) and (len(faltantes) == 0) and (len(nan_por_param) == 0)

    resumen.append({
        "departamento_id": depto_id,
        "filas": len(df),
        "fecha_min": df.index.min(),
        "fecha_max": df.index.max(),
        "fill_residual": n_fill,
        "fechas_faltantes": len(faltantes),
        "nan_maximo_por_param": nan_por_param.max() if len(nan_por_param) else 0,
        "ok": ok,
    })

df_resumen = pd.DataFrame(resumen)

print(f"Total departamentos revisados: {len(df_resumen)}")
print(f"Sin problemas: {df_resumen['ok'].sum()}")
print(f"Con algún problema: {(~df_resumen['ok']).sum()}")
print()

problemas = df_resumen[~df_resumen["ok"]]
if len(problemas) > 0:
    print("--- Departamentos con problemas ---")
    print(problemas.to_string(index=False))
else:
    print("Todo limpio.")

df_resumen.to_csv("resumen_validacion_clima.csv", index=False)