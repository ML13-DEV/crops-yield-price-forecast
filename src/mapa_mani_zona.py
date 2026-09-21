import json
import pandas as pd

with open(r"..\data\raw\departamentos_boxes.geojson", "r", encoding="utf-8") as f:
    deptos_gj = json.load(f)

# Los 10 departamentos del cinturón manisero (todos en Córdoba).
cinturon_manisero = [
    "Río Cuarto", "General Roca", "Juárez Celman", "Tercero Arriba",
    "Presidente Roque Sáenz Peña", "General San Martín", "Unión",
    "Río Segundo", "Marcos Juárez", "Río Primero",
]

filas = []
for feat in deptos_gj["features"]:
    p = feat["properties"]
    if p["nombre"] in cinturon_manisero:
        filas.append({
            "departamento_id": p["id"],
            "departamento": p["nombre"],
            "lon_centroide": p["centroide"]["lon"],
        })

df = pd.DataFrame(filas)

faltantes = set(cinturon_manisero) - set(df["departamento"])
if faltantes:
    raise ValueError(f"No se encontraron en el geojson: {faltantes}")
if len(df) != len(cinturon_manisero):
    raise ValueError(f"Se esperaban {len(cinturon_manisero)} departamentos, se encontraron {len(df)}")

# Terciles de longitud: oeste = longitud más negativa (más al oeste),
# este = longitud menos negativa (más al este). pd.qcut asigna los grupos
# en el mismo orden que 'labels' según el valor ordenado ascendentemente,
# por lo que el primer label cae en los valores más bajos (más al oeste).
df = df.sort_values("lon_centroide").reset_index(drop=True)
df["zona_mani"] = pd.qcut(df["lon_centroide"], 3, labels=["Maní Oeste", "Maní Núcleo", "Maní Este"])

print(df["zona_mani"].value_counts())
print()
print(df.to_string(index=False))

df[["departamento_id", "departamento", "zona_mani"]].to_csv("zona_mani_por_departamento.csv", index=False)
