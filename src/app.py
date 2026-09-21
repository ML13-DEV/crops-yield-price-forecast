import json
import joblib
import pandas as pd
import streamlit as st
import folium
import plotly.express as px
import shap
from streamlit_folium import st_folium
from prediccion_campo import obtener_target_encoding, calcular_fechas_ventana

# Duplicado a propósito, no importado de prediccion_campo.py -- no tenía
# forma de confirmar que ese diccionario sigue ahí después de los cambios
# de Claude Code en el caso de estudio del ONI. Si en algún momento cambia
# el mapeo cultivo->ventana en un lado, hay que actualizar el otro también.
CULTIVO_A_VENTANA = {
    "girasol": "Girasol", "maíz": "Maíz", "maní": "Maní",
    "soja 1ra": "Soja", "soja 2da": "Soja", "trigo total": "Trigo",
}

st.set_page_config(page_title="DataSoma — Predicción de rendimiento", layout="wide")

RUTA_GEOJSON = "../data/raw/departamentos_boxes.geojson"
RUTA_DATASET = "../data/processed/dataset_modelo_rendimiento.csv"
RUTA_MODELO = "../models/modelo_rendimiento_rf.pkl"
RUTA_FEATURES = "../models/features_modelo_rendimiento.json"
RUTA_TABLA_TE = "../models/promedio_rendimiento_departamento_cultivo.csv"

# IMPORTANTE: números finales de la validación cruzada del notebook 05
# (sección 5.11, modelo sin ONI) -- no se recalculan en vivo (correr
# GroupKFold+RandomizedSearchCV acá sería lento e innecesario para una demo).
# Reemplazá estos None por los valores reales de tu notebook antes de grabar.
METRICAS_FINALES = pd.DataFrame([
    {"Cultivo": "Girasol",  "MAE": 3.325,  "R2": 0.366, "Mejora_vs_baseline": 9.7},
    {"Cultivo": "Maíz",     "MAE": 10.425, "R2": 0.601, "Mejora_vs_baseline": 24.6},
    {"Cultivo": "Maní",     "MAE": 5.608,  "R2": 0.135, "Mejora_vs_baseline": 12.7},
    {"Cultivo": "Soja 1ra", "MAE": 4.533,  "R2": 0.535, "Mejora_vs_baseline": 21.9},
    {"Cultivo": "Soja 2da", "MAE": 4.259,  "R2": 0.435, "Mejora_vs_baseline": 18.3},
    {"Cultivo": "Trigo",    "MAE": 5.579,  "R2": 0.631, "Mejora_vs_baseline": 14.0},
])

# configuración final de Track A (notebook 04, celda 43) -- qué modelo se
# usa para cada cultivo/horizonte de precio. 'naive' = último precio conocido.
CONFIGURACION_PRECIO = {
    "girasol": {"15d": "rf",       "30d": "xgb",   "60d": "rf",    "90d": "naive"},
    "maiz":    {"15d": "naive",    "30d": "naive", "60d": "naive", "90d": "naive"},
    "soja":    {"15d": "reglineal", "30d": "naive", "60d": "naive", "90d": "naive"},
    "trigo":   {"15d": "naive",    "30d": "naive", "60d": "naive", "90d": "xgb"},
}
COLUMNAS_FEATURES_PRECIO = ["lag_1", "lag_2", "lag_4", "lag_8", "lag_13",
                             "rolling_mean_4", "rolling_mean_8", "rolling_mean_13",
                             "rolling_std_4", "return_1"]

# nombre técnico de columna -> nombre presentable para gráficos
NOMBRES_PRESENTABLES = {
    "ndvi_promedio_floracion": "NDVI en floración",
    "ndvi_promedio_llenado": "NDVI en llenado",
    "temperatura_media_floracion": "Temperatura media (floración)",
    "temperatura_media_llenado": "Temperatura media (llenado)",
    "precipitacion_total_floracion": "Precipitación (floración)",
    "precipitacion_total_llenado": "Precipitación (llenado)",
    "deficit_hidrico_extremo_floracion": "Días secos consecutivos (floración)",
    "deficit_hidrico_extremo_llenado": "Días secos consecutivos (llenado)",
    "radiacion_solar_promedio_floracion": "Radiación solar (floración)",
    "radiacion_solar_promedio_llenado": "Radiación solar (llenado)",
    "amplitud_termica_floracion": "Amplitud térmica (floración)",
    "amplitud_termica_llenado": "Amplitud térmica (llenado)",
    "estres_termico_floracion": "Estrés térmico (floración)",
    "estres_termico_llenado": "Estrés térmico (llenado)",
    "helada_agro_floracion": "Riesgo de helada (floración)",
    "helada_agro_llenado": "Riesgo de helada (llenado)",
    "depto_target_encoding": "Promedio histórico de la zona",
}


def nombre_legible(columna):
    """Traduce el nombre técnico de una columna a algo presentable. Las
    dummies de Cultivo_X/Zona_X se arman dinámicamente (son demasiadas
    para tenerlas todas a mano en NOMBRES_PRESENTABLES)."""
    if columna in NOMBRES_PRESENTABLES:
        return NOMBRES_PRESENTABLES[columna]
    if columna.startswith("Cultivo_"):
        return f"Cultivo: {columna.replace('Cultivo_', '').capitalize()}"
    if columna.startswith("Zona_"):
        return f"Zona: {columna.replace('Zona_', '')}"
    return columna


# ---------- carga (cacheada, se ejecuta una sola vez) ----------

@st.cache_data
def cargar_geojson():
    with open(RUTA_GEOJSON, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def cargar_dataset():
    df = pd.read_csv(RUTA_DATASET)
    df["Departamento_id"] = df["Departamento_id"].apply(lambda x: str(int(float(x))).zfill(5))
    return df


@st.cache_resource
def cargar_modelo():
    modelo = joblib.load(RUTA_MODELO)
    with open(RUTA_FEATURES, "r", encoding="utf-8") as f:
        features = json.load(f)
    return modelo, features


@st.cache_resource
def cargar_explicador(_modelo):
    # el guion bajo en _modelo le dice a Streamlit que no intente cachear
    # por el contenido del modelo (no es fácil de hashear) -- lo trata
    # como un recurso fijo, igual que ya hace con el modelo en sí.
    return shap.TreeExplainer(_modelo)


@st.cache_data
def cargar_tabla_te():
    tabla = pd.read_csv(RUTA_TABLA_TE)
    tabla["Departamento_id"] = tabla["Departamento_id"].apply(lambda x: str(int(float(x))).zfill(5))
    tabla = tabla.set_index(["Departamento_id", "Cultivo"])["Rinde(qq/Ha)"]
    return tabla


@st.cache_data
def cargar_ventanas():
    return pd.read_csv("../data/raw/ventanas_fenologicas.csv")


@st.cache_data
def cargar_precio_features(cultivo):
    """
    Recalcula las features de lag/rolling sobre la serie completa de precio
    -- misma lógica que calcular_features del notebook 04, pero SIN el
    dropna() de las columnas target (esas tienen NaN al final de la serie
    a propósito, porque el "futuro" todavía no pasó -- justo la última
    fila es la que necesitamos para predecir hacia adelante).
    """
    ruta = f"../data/processed/Precios_{cultivo}_semanal.csv"
    df = pd.read_csv(ruta, parse_dates=["Fecha"]).rename(columns={"Fecha": "ds", "Precio en USD": "y"})
    df["y"] = df["y"].astype(float)
    df["lag_1"] = df["y"].shift(1)
    df["lag_2"] = df["y"].shift(2)
    df["lag_4"] = df["y"].shift(4)
    df["lag_8"] = df["y"].shift(8)
    df["lag_13"] = df["y"].shift(13)
    df["rolling_mean_4"] = df["y"].rolling(4).mean()
    df["rolling_mean_8"] = df["y"].rolling(8).mean()
    df["rolling_mean_13"] = df["y"].rolling(13).mean()
    df["rolling_std_4"] = df["y"].rolling(4).std()
    df["return_1"] = (df["y"] - df["y"].shift(1)) / df["y"].shift(1)
    return df


@st.cache_resource
def cargar_modelo_precio(cultivo, horizonte, tipo_modelo):
    if tipo_modelo == "naive":
        return None
    return joblib.load(f"../models/{cultivo}_{horizonte}_{tipo_modelo}.pkl")


def construir_fila_prediccion(fila, cultivo, zona, departamento_id, features, tabla_te, promedio_global):
    """
    Arma la fila completa que el modelo espera a partir de una fila ya
    existente del dataset (que solo tiene las columnas climáticas/NDVI
    'naturales'): agrega las dummies de Cultivo/Zona y el target encoding
    de Departamento -- mismo patrón que calcular_prediccion_campo en
    prediccion_campo.py, para no duplicar la lógica.
    """
    valores = {}
    for col in features:
        if col.startswith("Cultivo_"):
            valores[col] = 1 if col == f"Cultivo_{cultivo}" else 0
        elif col.startswith("Zona_"):
            valores[col] = 1 if col == f"Zona_{zona}" else 0
        elif col == "depto_target_encoding":
            valores[col] = obtener_target_encoding(departamento_id, cultivo, tabla_te, promedio_global)
        else:
            valores[col] = fila[col]
    return pd.DataFrame([valores])[features]


geojson = cargar_geojson()
df = cargar_dataset()
modelo, features = cargar_modelo()
explicador = cargar_explicador(modelo)
tabla_te = cargar_tabla_te()
promedio_global = tabla_te.mean()
df_ventanas = cargar_ventanas()
nombre_a_id = {f["properties"]["nombre"]: f["properties"]["id"] for f in geojson["features"]}


# ---------- pestañas ----------

tab_prediccion, tab_precio, tab_metricas = st.tabs(["🌾 Rendimiento", "💰 Precio", "📊 Métricas y limitaciones"])

with tab_prediccion:
    st.title("Predicción de rendimiento por departamento")

    col_sel, col_mapa = st.columns([1, 2])

    with col_sel:
        departamento_nombre = st.selectbox("Departamento", sorted(nombre_a_id.keys()))
        departamento_id = nombre_a_id[departamento_nombre]

        cultivos_disponibles = sorted(df[df["Departamento_id"] == departamento_id]["Cultivo"].unique())
        if not cultivos_disponibles:
            st.warning("Este departamento no tiene datos de ningún cultivo en el dataset.")
            st.stop()
        cultivo = st.selectbox("Cultivo", cultivos_disponibles, format_func=lambda c: c.capitalize())

        campanias_disponibles = sorted(
            df[(df["Departamento_id"] == departamento_id) & (df["Cultivo"] == cultivo)]["Campaña"].unique(),
            reverse=True,
        )
        campania = st.selectbox("Campaña", campanias_disponibles)

    with col_mapa:
        centro = next(f["properties"]["centroide"] for f in geojson["features"] if f["properties"]["id"] == departamento_id)
        # tiles="OpenStreetMap" -- gratis, sin API key. cartodbpositron
        # empezó a exigir key propia y dejaba la marca de agua en el mapa.
        mapa = folium.Map(location=[centro["lat"], centro["lon"]], zoom_start=7, tiles="OpenStreetMap")

        def estilo(feature):
            seleccionado = feature["properties"]["id"] == departamento_id
            return {
                "fillColor": "#2ca25f" if seleccionado else "#a1d99b",
                "color": "#005824" if seleccionado else "#74c476",
                "weight": 3 if seleccionado else 1,
                "fillOpacity": 0.7 if seleccionado else 0.2,
            }

        folium.GeoJson(geojson, style_function=estilo,
                        tooltip=folium.GeoJsonTooltip(fields=["nombre"])).add_to(mapa)
        st_folium(mapa, width=700, height=450, returned_objects=[])

    st.divider()

    fila = df[(df["Departamento_id"] == departamento_id) & (df["Cultivo"] == cultivo) & (df["Campaña"] == campania)]

    if fila.empty:
        st.warning("No hay datos para esta combinación.")
    else:
        fila = fila.iloc[0]
        zona = fila["Zona"]
        X = construir_fila_prediccion(fila, cultivo, zona, departamento_id, features, tabla_te, promedio_global)
        prediccion = modelo.predict(X)[0]
        real = fila["Rinde(qq/Ha)"]

        # fecha de inicio de floración estimada -- el dataset no tiene una
        # fecha de siembra real observada (la fuente, MAGyP, no la incluye),
        # esto es lo más cercano: el arranque del calendario fenológico fijo
        # para este cultivo/zona/campaña, calculado con la misma lógica que
        # ya usa el resto del proyecto (calcular_fechas_ventana).
        cultivo_ventana = CULTIVO_A_VENTANA[cultivo]
        ventana_floracion = df_ventanas[
            (df_ventanas["cultivo"] == cultivo_ventana) &
            (df_ventanas["zona"] == zona) &
            (df_ventanas["etapa"] == "floracion")
        ]
        if not ventana_floracion.empty:
            fecha_inicio_floracion, _ = calcular_fechas_ventana(ventana_floracion.iloc[0], int(fila["Anio_Inicio"]))
            st.caption(f"📅 Inicio estimado de floración: {fecha_inicio_floracion.strftime('%d/%m/%Y')} "
                       f"(calendario fenológico de {zona} para {cultivo.capitalize()} — no es una fecha de "
                       f"siembra observada, el dataset no tiene ese dato)")

        col1, col2, col3 = st.columns(3)
        col1.metric("Predicción del modelo", f"{prediccion:.1f} qq/ha")
        col2.metric("Rendimiento real", f"{real:.1f} qq/ha")
        col3.metric("Error", f"{abs(prediccion - real):.1f} qq/ha", f"{100*abs(prediccion-real)/real:.1f}%")

        st.subheader("Qué pesó para ESTA predicción puntual")
        shap_values = explicador.shap_values(X)[0]
        serie_shap = pd.Series(shap_values, index=features)
        top8 = serie_shap.reindex(serie_shap.abs().sort_values(ascending=True).tail(8).index)
        top8.index = [nombre_legible(c) for c in top8.index]

        etiqueta_direccion = ["Sube la predicción" if v > 0 else "Baja la predicción" for v in top8.values]
        fig = px.bar(
            top8, orientation="h", color=etiqueta_direccion,
            color_discrete_map={"Sube la predicción": "#2ca25f", "Baja la predicción": "#de2d26"},
            labels={"value": "Impacto en qq/ha", "index": ""},
        )
        fig.update_layout(yaxis_title="", xaxis_title="Impacto en la predicción (qq/ha)", legend_title="")
        st.plotly_chart(fig, use_container_width=True)


with tab_precio:
    st.title("Predicción de precio")
    st.caption("A diferencia de la pestaña de rendimiento, esto sí es una predicción hacia adelante real, "
               "usando el último dato de precio disponible en el dataset.")

    col_sel, col_res = st.columns([1, 2])

    with col_sel:
        cultivo_precio = st.selectbox("Cultivo", ["girasol", "maiz", "soja", "trigo"], format_func=lambda c: c.capitalize())
        horizonte = st.selectbox("Horizonte", ["15d", "30d", "60d", "90d"])

    tipo_modelo = CONFIGURACION_PRECIO[cultivo_precio][horizonte]
    df_precio = cargar_precio_features(cultivo_precio)
    ultima_fila = df_precio.iloc[-1]
    precio_actual = ultima_fila["y"]
    fecha_dato = ultima_fila["ds"]

    with col_res:
        if tipo_modelo == "naive":
            st.metric(f"Predicción a {horizonte} (dato del {fecha_dato.date()})", f"USD {precio_actual:.2f}")
            st.info(
                f"Para {cultivo_precio.capitalize()} a {horizonte}, el modelo ganador fue el **naive** "
                f"-- la predicción **es** el último precio conocido, a propósito, no un valor sin calcular. "
                f"En la validación del notebook 04, ningún modelo (Random Forest, XGBoost, regresión lineal) "
                f"le ganó de forma consistente a \"el precio de mañana es el precio de hoy\" para esta "
                f"combinación -- los precios de commodities argentinos se comportan como una caminata "
                f"aleatoria en la mayoría de los casos (test de Dickey-Fuller, notebook 04)."
            )
        else:
            st.metric(f"Último precio conocido ({fecha_dato.date()})", f"USD {precio_actual:.2f}")

            modelo_precio = cargar_modelo_precio(cultivo_precio, horizonte, tipo_modelo)
            X_precio = ultima_fila[COLUMNAS_FEATURES_PRECIO].to_frame().T.astype(float)
            retorno_predicho = modelo_precio.predict(X_precio)[0]
            precio_predicho = precio_actual * (1 + retorno_predicho)
            nombres_modelo = {"rf": "Random Forest", "xgb": "XGBoost", "reglineal": "Regresión Lineal"}
            st.caption(f"Modelo: {nombres_modelo[tipo_modelo]} (le ganó al naive para esta combinación)")

            st.metric(f"Precio estimado a {horizonte}", f"USD {precio_predicho:.2f}",
                       f"{100*(precio_predicho-precio_actual)/precio_actual:.1f}%")


with tab_metricas:
    st.header("Qué tan bien funciona el modelo")

    st.subheader("Resultados por cultivo (validación cruzada, 26 campañas)")
    tabla = METRICAS_FINALES.copy()
    tabla["MAE ≤ 5 qq/ha"] = tabla["MAE"].apply(lambda x: "✅" if x is not None and x <= 5 else "❌")
    tabla["R² ≥ 0.65"] = tabla["R2"].apply(lambda x: "✅" if x is not None and x >= 0.65 else "❌")
    tabla["Mejora ≥ 20% vs. baseline"] = tabla["Mejora_vs_baseline"].apply(
        lambda x: "✅" if x is not None and x >= 20 else "❌"
    )
    st.dataframe(tabla, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("La limitación real: campos dentro de una misma zona")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Variación real de rendimiento entre campos de la misma zona", "24.5 qq/ha")
    with col2:
        st.metric("Variación que el modelo logra captar hoy", "6-8 qq/ha")

    st.markdown("""
    El modelo predice bien el **nivel típico de una zona** — no diferencia
    bien un campo excepcional de uno mediocre dentro de esa misma zona,
    porque nunca vio ejemplos de rendimiento a nivel de lote individual
    durante el entrenamiento, solo promedios de departamento.

    La agronomía ya tiene una respuesta aproximada de cuánto de esa
    diferencia es alcanzable con datos de clima y suelo: **el suelo, el
    clima y la geografía explican entre 60% y 70% del rendimiento en años
    climáticamente normales** — ese es el techo que buscamos alcanzar con
    más datos reales. El 30-40% restante depende del manejo del productor,
    y ningún dato satelital lo puede capturar.
    """)

    st.divider()
    st.subheader("¿Sos productor y querés ayudar a cerrar esa brecha?")
    st.markdown("""
    Buscamos productores dispuestos a compartir el rendimiento histórico
    de sus campos (varias campañas del mismo lote, idealmente con mapa de
    rinde de cosechadora) para mejorar la precisión del modelo a nivel de
    campo individual. Es solo para testeo interno — no se publica ni se
    comparte con nadie.
    """)