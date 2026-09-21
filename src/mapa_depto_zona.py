import json
import pandas as pd
import numpy as np

with open(r"..\data\raw\departamentos_boxes.geojson", "r", encoding="utf-8") as f:
    deptos_gj = json.load(f)

zonas = pd.read_csv(r"..\data\raw\zonas_bounding_boxes.csv", delimiter=";", index_col=False)
zonas["centro_lat"] = (zonas["lat_min"] + zonas["lat_max"]) / 2
zonas["centro_lon"] = (zonas["lon_min"] + zonas["lon_max"]) / 2


def zona_mas_cercana(lat, lon):
    """Distancia euclídea simple (no geodésica) al centro de cada zona --
    de sobra de precisión para desempatar entre cajas de cientos de km."""
    dist = np.sqrt((zonas["centro_lat"] - lat) ** 2 + (zonas["centro_lon"] - lon) ** 2)
    return zonas.loc[dist.idxmin(), "Zona"]


filas = []
for feat in deptos_gj["features"]:
    depto_id = feat["properties"]["id"]
    depto_nombre = feat["properties"]["nombre"]
    lat = feat["properties"]["centroide"]["lat"]
    lon = feat["properties"]["centroide"]["lon"]

    matches = zonas[
        (zonas["lat_min"] <= lat) & (lat <= zonas["lat_max"]) &
        (zonas["lon_min"] <= lon) & (lon <= zonas["lon_max"])
    ]

    if len(matches) == 1:
        zona_asignada = matches["Zona"].iloc[0]
        metodo = "match_unico"
    else:
        zona_asignada = zona_mas_cercana(lat, lon)
        metodo = "0_candidatas" if len(matches) == 0 else "desempate_por_distancia"

    filas.append({
        "departamento_id": depto_id,
        "departamento": depto_nombre,
        "zona_pas": zona_asignada,
        "metodo": metodo,
        "n_zonas_candidatas_originales": len(matches),
    })

mapa_depto_zona = pd.DataFrame(filas)

print(mapa_depto_zona["metodo"].value_counts())
print()
print("--- casos resueltos por desempate, para revisar si tienen sentido geográfico ---")
print(mapa_depto_zona[mapa_depto_zona["metodo"] != "match_unico"][["departamento", "zona_pas", "metodo"]].to_string(index=False))

mapa_depto_zona.to_csv("mapa_departamento_zona.csv", index=False)