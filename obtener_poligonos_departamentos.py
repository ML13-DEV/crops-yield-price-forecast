import requests
import json
import pandas as pd

URL_DEPARTAMENTOS = "https://apis.datos.gob.ar/georef/api/departamentos.geojson"


def descargar_departamentos_completo(ruta_cache="departamentos_completo.geojson"):
    """
    Descarga UNA sola vez el geojson nacional completo (~500 departamentos,
    con geometría real de cada uno) y lo cachea en disco. No hace falta
    pedir departamento por departamento -- este endpoint ya trae todo.
    """
    from pathlib import Path
    if Path(ruta_cache).exists():
        print(f"Ya existe {ruta_cache} en disco, no vuelvo a descargar.")
        with open(ruta_cache, "r", encoding="utf-8") as f:
            return json.load(f)

    print("Descargando departamentos.geojson completo (puede tardar, es un archivo grande)...")
    resp = requests.get(URL_DEPARTAMENTOS, timeout=300)
    resp.raise_for_status()
    data = resp.json()

    with open(ruta_cache, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    print(f"Guardado en {ruta_cache} -- {len(data['features'])} departamentos en total.")
    return data


def filtrar_departamentos(geojson_completo, ids_departamentos):
    """
    ids_departamentos: lista de IDs (columna 'departamento_id' de tu csv
    de estimaciones agrícolas, con padding a 5 dígitos -- georef usa
    id de 5 caracteres, ej. '06427', no 6427).
    """
    ids_set = set(str(i).zfill(5) for i in ids_departamentos)

    features_filtradas = [
        f for f in geojson_completo["features"]
        if f["properties"]["id"] in ids_set
    ]

    encontrados = {f["properties"]["id"] for f in features_filtradas}
    faltantes = ids_set - encontrados
    if faltantes:
        print(f"ATENCIÓN: {len(faltantes)} IDs de tu lista no se encontraron en georef: {faltantes}")

    return {"type": "FeatureCollection", "features": features_filtradas}


if __name__ == "__main__":
    # 1) tu lista de departamentos filtrados (la que armamos con el 95% de superficie)
    deptos_filtrados = pd.read_csv("departamentos_filtrados.csv")

    # 2) descargar (o leer de cache) el geojson nacional completo
    geojson_completo = descargar_departamentos_completo()

    # 3) filtrar solo a los que importan
    geojson_final = filtrar_departamentos(geojson_completo, deptos_filtrados["departamento_id"])

    # 4) guardar en el mismo formato que ya usás (zonas_boxes.geojson), listo para
    #    NASA POWER (armar bounding box a partir de la geometría) y AppEEARS
    with open("departamentos_boxes.geojson", "w", encoding="utf-8") as f:
        json.dump(geojson_final, f, ensure_ascii=False)

    print(f"\nListo: {len(geojson_final['features'])} de {len(deptos_filtrados)} departamentos con polígono real guardados en departamentos_boxes.geojson")
