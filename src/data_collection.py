import requests
import pandas as pd
import numpy as np
import time
import json
from pathlib import Path

URL_POINT = "https://power.larc.nasa.gov/api/temporal/daily/point"
comunidad = "AG"

# Valor de relleno que usa NASA POWER cuando no hay dato para un dia.
FILL_VALUE = -999

parametros = ["PRECTOTCORR", "T2M", "T2M_MAX", "T2M_MIN", "ALLSKY_SFC_SW_DWN", "RH2M"]

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw" / "clima_departamentos"


def cargar_departamentos(ruta_geojson=r"..\data\raw\departamentos_boxes.geojson"):
    """
    Ya no hace falta el polígono completo ni el bbox -- el endpoint 'point'
    solo necesita el centroide de cada departamento. Se deja el geojson
    como fuente porque ya tiene el centroide calculado por georef.
    """
    with open(ruta_geojson, "r", encoding="utf-8") as f:
        gj = json.load(f)

    departamentos = []
    for feat in gj["features"]:
        departamentos.append({
            "id": feat["properties"]["id"],
            "nombre": feat["properties"]["nombre"],
            "lat": feat["properties"]["centroide"]["lat"],
            "lon": feat["properties"]["centroide"]["lon"],
        })
    return pd.DataFrame(departamentos)


def descargar_parametro_por_departamento(depto, parametro, fecha_inicio, fecha_fin):
    nombre = depto["nombre"]
    print(f"REQUEST -> depto={nombre}, parametro={parametro}, inicio={fecha_inicio}, fin={fecha_fin}")
    payload = {
        "parameters": parametro,
        "community":  comunidad,
        "longitude":  depto["lon"],
        "latitude":   depto["lat"],
        "start":      fecha_inicio,
        "end":        fecha_fin,
        "format":     "JSON",
    }
    try:
        response = requests.get(URL_POINT, params=payload, timeout=120)
        response.raise_for_status()
        data = response.json()

        valores = data.get("properties", {}).get("parameter", {}).get(parametro)
        if not valores:
            print(f"Advertencia: sin datos de {parametro} para {nombre}.")
            return None

        serie = pd.Series(valores)
        serie.index = pd.to_datetime(serie.index, format="%Y%m%d")

        n_fill = (serie == FILL_VALUE).sum()
        if n_fill > 0:
            print(f"  -> {n_fill} valores de relleno ({FILL_VALUE}) encontrados y descartados.")
            serie = serie.replace(FILL_VALUE, np.nan)

        serie.name = parametro
        return serie
    except requests.RequestException as e:
        print(f"Error fetching data for {nombre} - {parametro}: {e}")
        return None


def conseguir_parametro_por_departamento(depto, parametro, historico=False, fecha_inicio=None, fecha_fin=None):
    if historico:
        series_list = []
        for anio in range(2000, 2027):
            fi, ff = f"{anio}0101", f"{anio}1231"
            try:
                serie = descargar_parametro_por_departamento(depto, parametro, fi, ff)
                if serie is not None:
                    series_list.append(serie)
            except requests.RequestException as e:
                print(f"Error fetching data for {depto['nombre']} - {parametro}: {e}")
                return None
            time.sleep(3)
        if len(series_list) == 0:
            return None
        combined = pd.concat(series_list)
        combined.name = parametro
        return combined
    else:
        return descargar_parametro_por_departamento(depto, parametro, fecha_inicio, fecha_fin)


def descargar_todos_los_departamentos():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    departamentos = cargar_departamentos()

    for _, depto in departamentos.iterrows():
        archivo = OUTPUT_DIR / f"clima_{depto['id']}.csv"

        if archivo.exists():
            data = pd.read_csv(archivo, index_col=0, parse_dates=True)
            parametros_actuales = [c for c in data.columns if c not in ('Departamento', 'Departamento_id')]
            if set(parametros_actuales) == set(parametros):
                print(f"Archivo {archivo} ya existe y contiene todos los parámetros. Saltando descarga.")
                continue
            else:
                columnas_faltantes = set(parametros) - set(parametros_actuales)
                for parametro in columnas_faltantes:
                    fecha_inicio = (pd.to_datetime(data[parametro].index.max()) + pd.Timedelta(days=1)).strftime("%Y%m%d")
                    fecha_fin = pd.Timestamp.today().strftime("%Y%m%d")
                    serie = conseguir_parametro_por_departamento(depto, parametro, historico=False, fecha_inicio=fecha_inicio, fecha_fin=fecha_fin)
                    if serie is not None:
                        data[parametro] = serie
                    time.sleep(3)
                data['Departamento'] = depto['nombre']
                data['Departamento_id'] = depto['id']
                if not data.empty:
                    data.to_csv(archivo)
                    print(f"Datos guardados en {archivo}.")
        else:
            print(f"Archivo {archivo} no existe. Descargando histórico completo para {depto['nombre']} (id {depto['id']}).")
            series = []
            for parametro in parametros:
                serie = conseguir_parametro_por_departamento(depto, parametro, historico=True)
                if serie is not None:
                    series.append(serie)
                time.sleep(3)
            if len(series) > 0:
                data = pd.concat(series, axis=1)
                data['Departamento'] = depto['nombre']
                data['Departamento_id'] = depto['id']
                if not data.empty:
                    data.to_csv(archivo)
                    print(f"Datos guardados en {archivo}.")


descargar_todos_los_departamentos()