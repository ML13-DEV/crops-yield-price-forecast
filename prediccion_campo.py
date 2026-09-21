import json
import requests
import pandas as pd
import numpy as np

API = "https://appeears.earthdatacloud.nasa.gov/api/"
FILL_VALUE_POWER = -999


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


def determinar_zona(lat, lon, zonas_df):
    """
    zonas_df: el mismo zonas_bounding_boxes.csv de siempre (columnas Zona,
    lat_min, lat_max, lon_min, lon_max).
    Point-in-box contra las 15 cajas. Si el punto cae en más de una zona
    (las franjas superpuestas que vimos en el mapa al principio del
    proyecto), no elige una al azar — devuelve la lista completa y avisa,
    porque no hay una regla de desempate definida todavía.
    """
    matches = zonas_df[
        (zonas_df["lat_min"] <= lat) & (lat <= zonas_df["lat_max"]) &
        (zonas_df["lon_min"] <= lon) & (lon <= zonas_df["lon_max"])
    ]

    if len(matches) == 0:
        raise ValueError(f"El punto ({lat}, {lon}) no cae en ninguna de las 15 zonas PAS.")
    if len(matches) > 1:
        print(f"ATENCIÓN: el punto cae en {len(matches)} zonas superpuestas: {list(matches['Zona'])}. "
              f"No hay regla de desempate definida — devolviendo la primera, revisar a mano.")

    return matches.iloc[0]["Zona"]


def _pivotear_a_una_fila(resultados, prefijo_valor, valores):
    """
    Mismo patrón de pivot_table que ya usás en el notebook para clima/NDVI,
    aplicado a una sola fila (una campaña, un campo). Convierte el resultado
    de calcular_features_climaticas/ndvi (una fila por Etapa) en una sola
    fila con columnas tipo 'precipitacion_total_floracion'.
    """
    df = pd.DataFrame(resultados)
    pivot = df.pivot_table(
        index=["Cultivo", "Zona", "Campaña"], columns="Etapa", values=valores, aggfunc="first"
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
    Carga los umbrales, la fuente del ONI y el mapeo SEAS->mes central desde
    el archivo que guarda el notebook 05 al entrenar el modelo final (ver
    notebook 03, sección 3.2.9, y notebook 05, sección 5.9.3/5.12.1 para por
    qué se usa la versión categórica del ONI y no la continua). Los umbrales
    NO se hardcodean acá — si se cambian en el notebook, este archivo cambia
    y esta función recoge el valor nuevo sin tocar código.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cargar_oni_por_mes(archivo_oni, config_oni):
    """
    Parsea el archivo ONI de NOAA CPC (columnas SEAS YR TOTAL ANOM, ver
    config_oni['fuente']) y arma un diccionario (año, mes) -> ANOM, mapeando
    cada código SEAS de 3 letras a su mes central con
    config_oni['seas_a_mes_central']. Misma lógica que el notebook 03.
    """
    df_oni = pd.read_csv(archivo_oni, sep=r"\s+")
    df_oni["mes_central"] = df_oni["SEAS"].map(config_oni["seas_a_mes_central"])
    return {
        (int(fila["YR"]), int(fila["mes_central"])): fila["ANOM"]
        for _, fila in df_oni.iterrows()
    }


def categorizar_oni(valor, config_oni):
    """
    Niño/Niña/Neutral según el promedio de ONI de una ventana puntual, con
    los umbrales de config_oni (±0.5°C por default). IMPORTANTE: esto NO es
    la definición oficial de NOAA de un episodio ENSO (que exige que el
    umbral se cumpla durante 5 temporadas SEAS consecutivas) — es la misma
    simplificación que ya documentan notebook 03 (3.2.9) y notebook 05
    (5.9.3): categoriza el promedio de una ventana de floración/llenado
    puntual, no una racha de 5 temporadas. Ver config_oni['nota'].
    """
    if valor >= config_oni["umbral_nino"]:
        return "Niño"
    elif valor <= config_oni["umbral_nina"]:
        return "Niña"
    return "Neutral"


def calcular_categoria_oni_ventana(fecha_inicio, fecha_fin, oni_por_mes, config_oni):
    """
    Promedia el ONI de los meses calendario dentro de [fecha_inicio, fecha_fin]
    y lo categoriza — mismo cálculo que calcular_features_oni en el notebook
    03 (promedio simple por mes, sin ponderar por días), aplicado a una sola
    ventana (floración o llenado) de un campo puntual. Devuelve
    (promedio, categoria); ambos NaN/None si ningún mes de la ventana tiene
    ONI disponible (por ejemplo, una fecha futura sin dato todavía publicado).
    """
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


def calcular_prediccion_campo(lat, lon, cultivo, zona, campania, fecha_siembra, df_ventanas,
                                df_ndvi_punto, modelo_final, features):
    """
    Arma la fila de features para un campo puntual y devuelve la predicción
    del modelo ya entrenado. Reutiliza calcular_features_climaticas y
    calcular_features_ndvi tal cual están.

    zona y campania se pasan directo (ya los tenés en tu excel de
    productores) — no se infieren, evita el problema de la superposición
    de zonas hasta que haya otra forma de resolverlo.

    fecha_siembra define desde cuándo pedirle datos a NASA POWER / AppEEARS
    (más ajustado que pedir desde el 1° de enero) — pero las ventanas de
    floración/llenado en sí siguen saliendo del calendario fijo de
    ventanas_fenologicas.csv, no de esta fecha real.

    IMPORTANTE: df_ndvi_punto tiene que venir YA DESCARGADO, resuelto antes
    de llamar a esta función con pedir_ndvi_punto() + descargar_ndvi_punto()
    por separado (AppEEARS es asíncrono, no se puede resolver adentro de una
    sola llamada sync). El clima sí se pide acá adentro porque el endpoint
    'point' de NASA POWER responde al instante.

    PENDIENTE — esta función todavía NO arma una columna que el modelo
    guardado en notebook 05 (5.14) espera en `features`:
    - `depto_target_encoding` (5.12): requiere resolver el `Departamento_id`
      de (lat, lon) contra los 158 departamentos y el promedio histórico de
      `promedio_rendimiento_departamento_cultivo.csv` — no está resuelto acá
      (esta función sigue trabajando con las 15 zonas PAS, no con
      departamento).
    Si `features` viene de `features_modelo_rendimiento.json` (el modelo
    actual), esta función va a fallar o rellenar esa columna con NaN — no
    asumir que el resultado es una predicción válida del modelo vigente
    hasta que se complete esta integración.

    NO PENDIENTE — descartado, no falta conectar: el ONI (continuo o
    categórico) se evaluó a fondo en notebook 05 (5.9.3, 5.12.1, 5.13.3,
    5.13.4) y se **descartó** del modelo de producción — en las
    comparaciones parejas (mismo tuneo de hiperparámetros en ambos lados),
    el ONI categórico nunca aportó una mejora neta de MAE sobre no tenerlo,
    con o sin `depto_target_encoding` (ver la matriz 2x2 en notebook 05,
    sección 5.11). El modelo guardado en 5.14 ya no incluye columnas de
    ONI en `features`, así que esta función no necesita construirlas.
    `cargar_config_oni`, `cargar_oni_por_mes`, `categorizar_oni` y
    `calcular_categoria_oni_ventana` se dejan en este archivo como
    referencia de esa investigación, no porque haga falta llamarlas acá.
    """
    anio_inicio = int(campania.split("/")[0])
    fecha_siembra = pd.Timestamp(fecha_siembra)
    fecha_fin_datos = pd.Timestamp(year=anio_inicio + 1, month=12, day=31)

    df_clima_punto = obtener_clima_punto(
        lat, lon, fecha_siembra.strftime("%Y%m%d"), fecha_fin_datos.strftime("%Y%m%d")
    )
    df_clima_punto["Zona"] = zona
    df_ndvi_punto = df_ndvi_punto.copy()
    df_ndvi_punto["zona"] = zona

    df_camp_fila = pd.DataFrame([{
        "Cultivo": cultivo, "Zona": zona, "Campaña": campania, "Anio_Inicio": anio_inicio,
    }])

    resultados_clima = calcular_features_climaticas(df_camp_fila, df_clima_punto, df_ventanas)
    resultados_ndvi = calcular_features_ndvi(df_camp_fila, df_ndvi_punto, df_ventanas)

    cols_clima = ["precipitacion_total", "deficit_hidrico_extremo", "temperatura_media",
                  "radiacion_solar_promedio", "amplitud_termica", "estres_termico", "helada_agro"]
    cols_ndvi = ["ndvi_promedio", "ndvi_n_observaciones"]

    fila_clima = _pivotear_a_una_fila(resultados_clima, "clima", cols_clima)
    fila_ndvi = _pivotear_a_una_fila(resultados_ndvi, "ndvi", cols_ndvi)
    fila_features = fila_clima.merge(fila_ndvi, on=["Cultivo", "Zona", "Campaña"])

    # dummies de Cultivo/Zona: el modelo espera las mismas columnas exactas
    # del entrenamiento (14 zonas + 4 cultivos) — para esta fila, todas en 0
    # salvo la que corresponde a este campo
    for col in features:
        if col.startswith("Cultivo_") or col.startswith("Zona_"):
            fila_features[col] = 1 if col in (f"Cultivo_{cultivo}", f"Zona_{zona}") else 0
        elif col not in fila_features.columns:
            fila_features[col] = np.nan  # feature climática/NDVI que no se pudo calcular

    prediccion = modelo_final.predict(fila_features[features])[0]
    return prediccion, zona, campania
