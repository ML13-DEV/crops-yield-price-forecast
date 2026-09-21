import pandas as pd
import json
import getpass
from pathlib import Path
from prediccion_campo import pedir_ndvi_punto

RUTA_LOTES = "lotes_productor.csv"
RUTA_TASK_IDS = "ndvi_lotes_task_ids.json"


def conseguir_token(usuario, contrasena):
    import requests
    resp = requests.post("https://appeears.earthdatacloud.nasa.gov/api/login", auth=(usuario, contrasena))
    resp.raise_for_status()
    return resp.json()["token"]


def main():
    lotes = pd.read_csv(RUTA_LOTES)
    lotes["fecha_siembra"] = pd.to_datetime(lotes["fecha_siembra"])

    usuario = input("Usuario Earthdata Login: ")
    contrasena = getpass.getpass("Contraseña: ")
    token = conseguir_token(usuario, contrasena)

    if Path(RUTA_TASK_IDS).exists():
        with open(RUTA_TASK_IDS, "r", encoding="utf-8") as f:
            task_ids = json.load(f)
    else:
        task_ids = {}

    for _, lote in lotes.iterrows():
        lote_id = str(lote["lote_id"])
        if lote_id in task_ids:
            print(f"Lote {lote_id}: ya enviado, salteando.")
            continue

        # rango amplio: desde la siembra hasta fin de la campaña, para
        # tener de sobra los compuestos de floración y llenado
        fecha_inicio = lote["fecha_siembra"].strftime("%m-%d-%Y")
        fecha_fin = f"12-31-{lote['fecha_siembra'].year + 1}"

        task_id = pedir_ndvi_punto(
            lote["lat"], lote["lon"], fecha_inicio, fecha_fin, token, campo_id=f"lote_{lote_id}"
        )
        task_ids[lote_id] = task_id

        with open(RUTA_TASK_IDS, "w", encoding="utf-8") as f:
            json.dump(task_ids, f, indent=2)

    print(f"\n{len(task_ids)} de {len(lotes)} lotes con request enviado. IDs guardados en {RUTA_TASK_IDS}.")


if __name__ == "__main__":
    main()
