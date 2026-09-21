import json
import pandas as pd

with open(r"..\data\raw\departamentos_boxes.geojson", "r", encoding="utf-8") as f:
    deptos_gj = json.load(f)

zonas = pd.read_csv(r"..\data\raw\zonas_bounding_boxes.csv", delimiter=";", index_col=False)

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

    filas.append({
        "departamento_id": depto_id,
        "departamento": depto_nombre,
        "zona_pas": matches["Zona"].iloc[0] if len(matches) >= 1 else None,
        "n_zonas_candidatas": len(matches),
        "zonas_candidatas": list(matches["Zona"]) if len(matches) > 1 else None,
    })

mapa_depto_zona = pd.DataFrame(filas)

sin_zona = mapa_depto_zona[mapa_depto_zona["n_zonas_candidatas"] == 0]
ambiguos = mapa_depto_zona[mapa_depto_zona["n_zonas_candidatas"] > 1]

print(f"Total departamentos: {len(mapa_depto_zona)}")
print(f"Sin ninguna zona PAS que los contenga: {len(sin_zona)}")
print(f"Con más de una zona candidata (superposición): {len(ambiguos)}")
if len(ambiguos) > 0:
    print(ambiguos[["departamento", "zonas_candidatas"]].to_string(index=False))

mapa_depto_zona.to_csv("mapa_departamento_zona.csv", index=False)