import json
import requests
import pandas as pd
import numpy as np
from shapely.geometry import shape, Point
from features_comunes import calcular_features_climaticas, calcular_features_ndvi

API = "https://appeears.earthdatacloud.nasa.gov/api/"
FILL_VALUE_POWER = -999

# mismo mapeo que usa notebook 03 (celda que arma df['cultivo_ventana']) --
# Soja 1ra y 2da comparten la ventana fenológica 'Soja', las demás son 1 a 1.
CULTIVO_A_VENTANA = {
    'girasol': 'Girasol', 'maíz': 'Maíz', 'maní': 'Maní',
    'soja 1ra': 'Soja', 'soja 2da': 'Soja', 'trigo total': 'Trigo',
}


def pedir_ndvi_punto(lat, lon, fecha_inicio, fecha_fin, token, campo_id="campo_prueba"):
    """
    Envía un request tipo 'point' a AppEEARS para un lat/lon puntual.
    fecha_inicio/fecha_fin en formato 'MM-DD-YYYY' (AppEEARS point usa ese
    formato, distinto del 'YYYY-MM-DD' que ves en el CSV de salida).
    Devuelve el task_id. El pedido queda "processing" un rato — no está
    listo apenas se envía, hay que consultarlo después con descargar_ndvi_punto.
    """
    headers = {"Authorization": f"Bearer {token}"}

    task = {
        "task_type": "point",
        "task_name": campo_id,
        "params": {
            "dates": [{"startDate": fecha_inicio, "endDate": fecha_fin}],
            "layers": [{"layer": "_250m_16_days_NDVI", "product": "MOD13Q1.061"}],
            "coordinates": [{"latitude": lat, "longitude": lon, "id": campo_id}],
        },
    }

    resp = requests.post(f"{API}task", json=task, headers=headers)
    resp.raise_for_status()
    task_id = resp.json()["task_id"]
    print(f"Request enviado, task_id={task_id}. Consultá el estado antes de bajar el resultado.")
    return task_id


def descargar_ndvi_punto(task_id, token):
    """
    Chequea el estado del task. Si ya terminó ('done'), baja el CSV de
    resultado y lo devuelve como DataFrame con columnas ['fecha_media', 'ndvi'].
    Si todavía está procesando, devuelve None — hay que reintentar más tarde,
    no bloquea esperando.
    """
    headers = {"Authorization": f"Bearer {token}"}

    estado = requests.get(f"{API}task/{task_id}", headers=headers).json()
    if estado["status"] != "done":
        print(f"Task {task_id} todavía en estado '{estado['status']}', probá de nuevo en un rato.")
        return None

    bundle = requests.get(f"{API}bundle/{task_id}", headers=headers).json()
    archivo_csv = next(
        a for a in bundle["files"]
        if a["file_name"].endswith(".csv") and "results" in a["file_name"].lower()
    )

    resp = requests.get(f"{API}bundle/{task_id}/{archivo_csv['file_id']}", headers=headers)
    from io import StringIO
    df = pd.read_csv(StringIO(resp.text))

    columna_ndvi = [c for c in df.columns if "NDVI" in c and "_MOD13Q1_061" in c][0]
    df_ndvi = df[["Date", columna_ndvi]].rename(columns={"Date": "fecha", columna_ndvi: "ndvi"})
    df_ndvi["fecha"] = pd.to_datetime(df_ndvi["fecha"])
    df_ndvi["fecha_media"] = df_ndvi["fecha"] + pd.Timedelta(days=8)

    return df_ndvi[["fecha_media", "ndvi"]]


def obtener_clima_punto(lat, lon, fecha_inicio, fecha_fin):
    """
    NASA POWER tiene un endpoint 'point' dedicado — a diferencia del
    'regional' que usaste para las 15 zonas, este devuelve directo la serie
    del punto exacto, sin necesidad de promediar varios píxeles ni filtrar
    océano (es un solo punto, no una grilla).
    fecha_inicio/fecha_fin en formato 'YYYYMMDD'.
    """
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    parametros = ["PRECTOTCORR", "T2M", "T2M_MAX", "T2M_MIN", "ALLSKY_SFC_SW_DWN", "RH2M"]

    payload = {
        "parameters": ",".join(parametros),
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": fecha_inicio,
        "end": fecha_fin,
        "format": "JSON",
    }

    resp = requests.get(url, params=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()["properties"]["parameter"]

    df_clima = pd.DataFrame(data)
    df_clima.index = pd.to_datetime(df_clima.index, format="%Y%m%d")
    df_clima = df_clima.replace(FILL_VALUE_POWER, np.nan)
    df_clima.index.name = "Fecha"
    return df_clima.reset_index()


def _pivotear_a_una_fila(resultados, prefijo_valor, valores):
    """
    Mismo patrón de pivot_table que ya usa el notebook 03 para clima/NDVI,
    aplicado a una sola fila (una campaña, un campo). Convierte el resultado
    de calcular_features_climaticas/ndvi (una fila por Etapa) en una sola
    fila con columnas tipo 'precipitacion_total_floracion'.
    Pivotea por Departamento_id (no por Zona) -- así quedó en notebook 03
    después de la migración, porque el clima/NDVI viven a nivel departamento.
    """
    df = pd.DataFrame(resultados)
    pivot = df.pivot_table(
        index=["Cultivo", "Departamento_id", "Campaña"], columns="Etapa", values=valores, aggfunc="first"
    )
    pivot.reset_index(inplace=True)
    pivot.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in pivot.columns]
    return pivot


def calcular_fechas_ventana(ventana, anio_inicio):
    """
    Convierte mes/día de una fila de ventanas_fenologicas.csv en fechas
    reales, cruzando el año si la etapa cae en la primera mitad del año
    calendario (mismo criterio que ya usan calcular_features_climaticas y
    calcular_features_ndvi — factorizado acá para no repetirlo una vez más).
    """
    anio_inicio_etapa = anio_inicio + 1 if ventana["mes_inicio"] <= 6 else anio_inicio
    fecha_inicio = pd.Timestamp(year=anio_inicio_etapa, month=ventana["mes_inicio"], day=ventana["dia_inicio"])

    anio_fin_etapa = anio_inicio + 1 if ventana["mes_fin"] <= 6 else anio_inicio
    fecha_fin = pd.Timestamp(year=anio_fin_etapa, month=ventana["mes_fin"], day=ventana["dia_fin"])

    return fecha_inicio, fecha_fin


def cargar_config_oni(path="../models/oni_categoria_config.json"):
    """
    NOTA (ver notebook 05, 5.11 y 5.14): el ONI (continuo y categórico) se
    investigó a fondo y se DESCARTÓ del modelo final de producción — no
    aportaba mejora neta una vez que el modelo estaba bien tuneado y con
    depto_target_encoding. `models/oni_categoria_config.json` ya NO existe
    (se borra en 5.14). Esta función y las tres siguientes (cargar_oni_por_mes,
    categorizar_oni, calcular_categoria_oni_ventana) quedan en el archivo
    solo como referencia del caso de estudio — el modelo vigente no las
    necesita ni las llama.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cargar_oni_por_mes(archivo_oni, config_oni):
    """Ver nota de cargar_config_oni — no usado por el modelo final."""
    df_oni = pd.read_csv(archivo_oni, sep=r"\s+")
    df_oni["mes_central"] = df_oni["SEAS"].map(config_oni["seas_a_mes_central"])
    return {
        (int(fila["YR"]), int(fila["mes_central"])): fila["ANOM"]
        for _, fila in df_oni.iterrows()
    }


def categorizar_oni(valor, config_oni):
    """
    Ver nota de cargar_config_oni — no usado por el modelo final.
    IMPORTANTE (se mantiene por referencia): esto NO es la definición
    oficial de NOAA de un episodio ENSO (que exige que el umbral se cumpla
    durante 5 temporadas SEAS consecutivas) — es una simplificación que
    categoriza el promedio de una ventana de floración/llenado puntual.
    """
    if valor >= config_oni["umbral_nino"]:
        return "Niño"
    elif valor <= config_oni["umbral_nina"]:
        return "Niña"
    return "Neutral"


def calcular_categoria_oni_ventana(fecha_inicio, fecha_fin, oni_por_mes, config_oni):
    """Ver nota de cargar_config_oni — no usado por el modelo final."""
    meses_ventana = pd.period_range(fecha_inicio, fecha_fin, freq="M")
    valores = [oni_por_mes.get((p.year, p.month)) for p in meses_ventana]
    valores = [v for v in valores if v is not None]
    if not valores:
        return np.nan, None
    promedio = float(np.mean(valores))
    return promedio, categorizar_oni(promedio, config_oni)


def determinar_etapa_actual(cultivo, zona, fecha_siembra, df_ventanas, fecha_consulta=None):
    """
    Dado cultivo, zona y fecha de siembra, determina en qué etapa
    fenológica (floración, llenado, o fuera de ventana) cae fecha_consulta
    (por default, hoy). Usa el mismo calendario fijo de
    ventanas_fenologicas.csv que ya usa calcular_features_climaticas — no
    ajusta las ventanas a la fecha de siembra real de este campo puntual,
    solo determina en qué tramo del calendario fijo de la zona cae la
    fecha de consulta.
    """
    fecha_siembra = pd.Timestamp(fecha_siembra)
    fecha_consulta = pd.Timestamp(fecha_consulta) if fecha_consulta is not None else pd.Timestamp.today()

    anio_inicio = fecha_siembra.year if fecha_siembra.month > 6 else fecha_siembra.year - 1

    ventanas_filtradas = df_ventanas[
        (df_ventanas["cultivo"] == cultivo) & (df_ventanas["zona"] == zona)
    ]
    if ventanas_filtradas.empty:
        raise ValueError(f"No hay ventana fenológica definida para {cultivo} en {zona}.")

    etapas = sorted(
        [(ventana["etapa"], *calcular_fechas_ventana(ventana, anio_inicio))
         for _, ventana in ventanas_filtradas.iterrows()],
        key=lambda e: e[1],
    )

    for etapa, f_inicio, f_fin in etapas:
        if f_inicio <= fecha_consulta <= f_fin:
            return {"etapa": etapa, "inicio": f_inicio, "fin": f_fin}

    if fecha_consulta < etapas[0][1]:
        return {"etapa": f"antes de {etapas[0][0]} (vegetativo)", "inicio": None, "fin": etapas[0][1]}
    if fecha_consulta > etapas[-1][2]:
        return {"etapa": f"después de {etapas[-1][0]} (cosecha o posterior)", "inicio": etapas[-1][2], "fin": None}

    for (etapa_a, _, fin_a), (etapa_b, inicio_b, _) in zip(etapas, etapas[1:]):
        if fin_a < fecha_consulta < inicio_b:
            return {"etapa": f"entre {etapa_a} y {etapa_b}", "inicio": fin_a, "fin": inicio_b}

    return {"etapa": "sin determinar", "inicio": None, "fin": None}


def determinar_departamento(lat, lon, ruta_geojson="../data/raw/departamentos_boxes.geojson"):
    """
    Point-in-polygon contra el polígono REAL de cada uno de los 158
    departamentos (no un rectángulo aproximado). A diferencia de las 15
    zonas PAS viejas, los polígonos administrativos reales no se
    superponen entre sí, así que acá no hay ambigüedad posible: o el
    punto cae en un departamento, o no cae en ninguno de los 158
    filtrados (campo fuera del área de cobertura del proyecto).
    """
    with open(ruta_geojson, "r", encoding="utf-8") as f:
        gj = json.load(f)

    punto = Point(lon, lat)
    for feat in gj["features"]:
        poligono = shape(feat["geometry"])
        if poligono.contains(punto):
            return feat["properties"]["id"], feat["properties"]["nombre"]

    raise ValueError(
        f"El punto ({lat}, {lon}) no cae en ninguno de los 158 departamentos filtrados -- "
        f"campo fuera del área de cobertura del proyecto (ver notebook 03, filtro de superficie sembrada)."
    )


def obtener_zona_para_departamento(departamento_id, cultivo, mapa_zona_pas, mapa_zona_mani):
    """
    La zona depende del cultivo, no solo del departamento (ver notebook 03) --
    maní usa mapa_zona_mani (las 3 zonas Oeste/Núcleo/Este por longitud),
    los otros 5 cultivos usan mapa_zona_pas (zona PAS heredada, con
    match_unico o desempate_por_distancia documentado en el propio archivo).
    """
    if cultivo == "maní":
        fila = mapa_zona_mani[mapa_zona_mani["departamento_id"] == departamento_id]
        if fila.empty:
            raise ValueError(f"Departamento {departamento_id} no está en el cinturón manisero filtrado.")
        return fila.iloc[0]["zona_mani"]
    else:
        fila = mapa_zona_pas[mapa_zona_pas["departamento_id"] == departamento_id]
        if fila.empty:
            raise ValueError(f"Departamento {departamento_id} no tiene zona PAS asignada en mapa_departamento_zona.csv.")
        return fila.iloc[0]["zona_pas"]


def obtener_target_encoding(departamento_id, cultivo, tabla_promedio, promedio_global):
    """
    Busca el promedio histórico de Rinde(qq/Ha) para (Departamento, Cultivo)
    en la tabla guardada por el notebook 05 (5.14). Si la combinación nunca
    se vio en el histórico (cultivo nuevo en ese departamento), usa el
    promedio_global como fallback -- mismo criterio que ya usa
    entrenar_con_kfold para las combinaciones no vistas en cada fold.
    """
    clave = (departamento_id, cultivo)
    if clave in tabla_promedio.index:
        return tabla_promedio.loc[clave]
    print(f"AVISO: sin historial de {cultivo} en departamento {departamento_id} -- usando promedio global como fallback.")
    return promedio_global


def calcular_prediccion_campo(lat, lon, cultivo, campania, fecha_siembra, df_ventanas,
                                df_ndvi_punto, modelo_final, features,
                                mapa_zona_pas, mapa_zona_mani, tabla_target_encoding, promedio_global):
    """
    Versión actualizada para el modelo final (5.14): Random Forest + Zona +
    Departamento(TE), sin ninguna columna de ONI (ver notebook 05, 5.11 --
    el caso de estudio completo de por qué el ONI se descartó).

    departamento y zona ya NO se piden como parámetro -- se resuelven acá
    adentro a partir de (lat, lon) y cultivo, de forma no ambigua (polígono
    real de departamento, no las 15 cajas superpuestas de antes).

    IMPORTANTE: df_ndvi_punto tiene que venir YA DESCARGADO, resuelto antes
    de llamar a esta función con pedir_ndvi_punto() + descargar_ndvi_punto()
    por separado (AppEEARS es asíncrono). El clima sí se pide acá adentro
    porque el endpoint 'point' de NASA POWER responde al instante.
    """
    departamento_id, departamento_nombre = determinar_departamento(lat, lon)
    zona = obtener_zona_para_departamento(departamento_id, cultivo, mapa_zona_pas, mapa_zona_mani)

    anio_inicio = int(campania.split("/")[0])
    fecha_siembra = pd.Timestamp(fecha_siembra)
    fecha_fin_datos = pd.Timestamp(year=anio_inicio + 1, month=12, day=31)

    df_clima_punto = obtener_clima_punto(
        lat, lon, fecha_siembra.strftime("%Y%m%d"), fecha_fin_datos.strftime("%Y%m%d")
    )
    clima_por_depto = {departamento_id: df_clima_punto}
    ndvi_por_depto = {departamento_id: df_ndvi_punto}

    df_camp_fila = pd.DataFrame([{
        "Cultivo": cultivo,
        "cultivo_ventana": CULTIVO_A_VENTANA[cultivo],
        "Zona": zona,
        "Departamento_id": departamento_id,
        "Campaña": campania,
        "Anio_Inicio": anio_inicio,
    }])

    resultados_clima = calcular_features_climaticas(df_camp_fila, clima_por_depto, df_ventanas)
    resultados_ndvi = calcular_features_ndvi(df_camp_fila, ndvi_por_depto, df_ventanas)

    cols_clima = ["precipitacion_total", "deficit_hidrico_extremo", "temperatura_media",
                  "radiacion_solar_promedio", "amplitud_termica", "estres_termico", "helada_agro"]
    cols_ndvi = ["ndvi_promedio", "ndvi_n_observaciones"]

    fila_clima = _pivotear_a_una_fila(resultados_clima, "clima", cols_clima)
    fila_ndvi = _pivotear_a_una_fila(resultados_ndvi, "ndvi", cols_ndvi)
    fila_features = fila_clima.merge(fila_ndvi, on=["Cultivo", "Departamento_id", "Campaña"])

    fila_features["depto_target_encoding"] = obtener_target_encoding(
        departamento_id, cultivo, tabla_target_encoding, promedio_global
    )

    for col in features:
        if col.startswith("Cultivo_") or col.startswith("Zona_"):
            fila_features[col] = 1 if col in (f"Cultivo_{cultivo}", f"Zona_{zona}") else 0
        elif col not in fila_features.columns:
            fila_features[col] = np.nan  # feature climática/NDVI que no se pudo calcular

    prediccion = modelo_final.predict(fila_features[features])[0]
    return prediccion, departamento_id, departamento_nombre, zona, campania


if __name__ == "__main__":
    from validar_productor import main

    main()