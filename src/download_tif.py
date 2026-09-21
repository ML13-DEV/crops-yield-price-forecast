import requests
import json
import getpass
from pathlib import Path

API = "https://appeears.earthdatacloud.nasa.gov/api/"
RUTA_TASK_IDS = "appeears_task_ids.json"
OUTPUT_DIR = Path(r"..\data\raw\ndvi_departamentos")


def conseguir_token(usuario, contrasena):
    login = requests.post(f"{API}login", auth=(usuario, contrasena))
    login.raise_for_status()
    return login.json()["token"]


def revisar_y_descargar(usuario, contrasena):
    with open(RUTA_TASK_IDS, "r", encoding="utf-8") as f:
        task_ids = json.load(f)

    token = conseguir_token(usuario, contrasena)
    headers = {"Authorization": f"Bearer {token}"}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    estados = {"done": 0, "pending": 0, "processing": 0, "otro": 0}

    for clave, task_id in task_ids.items():
        estado = requests.get(f"{API}task/{task_id}", headers=headers).json()
        status = estado.get("status", "otro")
        estados[status if status in estados else "otro"] += 1

        if status != "done":
            print(f"{clave}: {status} -- todavía no listo")
            continue

        bundle = requests.get(f"{API}bundle/{task_id}", headers=headers).json()

        # el nombre del archivo de estadísticas cambia según la capa:
        # NDVI  -> "...Statistics.csv" (sin QA en el nombre)
        # pixel_reliability -> "...Statistics-QA.csv" (SÍ tiene QA, es el nombre real
        #   del archivo que queremos, no uno secundario a descartar)
        es_pixel_reliability = "reliability" in clave.lower()
        if es_pixel_reliability:
            archivo_stats = next(
                (a for a in bundle["files"] if a["file_name"].lower().endswith("-qa.csv") and "reliability" in a["file_name"].lower()),
                None
            )
        else:
            archivo_stats = next(
                (a for a in bundle["files"] if a["file_name"].endswith("Statistics.csv") and "QA" not in a["file_name"]),
                None
            )
        if archivo_stats is None:
            print(f"{clave}: 'done' pero no encontré el Statistics.csv en el bundle -- revisar a mano.")
            continue

        destino = OUTPUT_DIR / f"{clave}_Statistics.csv"
        if destino.exists():
            print(f"{clave}: ya descargado, salteando.")
            continue

        resp = requests.get(f"{API}bundle/{task_id}/{archivo_stats['file_id']}", headers=headers)
        with open(destino, "wb") as f:
            f.write(resp.content)
        print(f"{clave}: descargado -> {destino}")

    print()
    print("Resumen de estados:", estados)
    if estados["done"] < len(task_ids):
        print(f"Todavía faltan {len(task_ids) - estados['done']} de {len(task_ids)} -- volvé a correr este script más tarde para esos.")

    print()
    print("--- Desglose por capa (para confirmar si falta una capa entera, no solo años sueltos) ---")
    for capa_nombre in set(clave.rsplit("_", 1)[0] for clave in task_ids):
        archivos_esperados = [c for c in task_ids if c.startswith(capa_nombre)]
        archivos_bajados = list(OUTPUT_DIR.glob(f"{capa_nombre}_*_Statistics.csv"))
        print(f"{capa_nombre}: {len(archivos_bajados)} de {len(archivos_esperados)} descargados")


if __name__ == "__main__":
    usuario = input("Usuario Earthdata Login: ")
    contrasena = getpass.getpass("Contraseña: ")
    revisar_y_descargar(usuario, contrasena)