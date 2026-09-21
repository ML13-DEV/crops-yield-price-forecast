import pandas as pd
import numpy as np


def calcular_features_climaticas(df_camp, clima_por_depto, df_ventanas):
    """
    Extraída tal cual del notebook 03 (celda que arma df_resultados_pivot).
    df_camp tiene que traer 'Cultivo', 'cultivo_ventana', 'Zona',
    'Departamento_id', 'Anio_Inicio', 'Campaña' -- 'cultivo_ventana' es
    necesaria porque Soja 1ra/2da comparten la misma ventana ('Soja' en
    ventanas_fenologicas.csv) aunque tengan Cultivo distinto.
    """
    if df_camp is None or clima_por_depto is None or df_ventanas is None:
        raise ValueError("Los parámetros de entrada no pueden ser None")

    resultados = []

    for _, fila in df_camp.iterrows():

        cultivo_ventana = fila['cultivo_ventana']
        zona = fila['Zona']
        depto_id = fila['Departamento_id']
        anio_inicio = fila['Anio_Inicio']
        campania = fila['Campaña']

        ventanas_filtradas = df_ventanas[
            (df_ventanas['cultivo'] == cultivo_ventana) &
            (df_ventanas['zona'] == zona)
        ]

        df_depto = clima_por_depto.get(depto_id)

        for _, ventana in ventanas_filtradas.iterrows():

            if ventana['mes_inicio'] <= 6:
                anio_inicio_etapa = anio_inicio + 1
            else:
                anio_inicio_etapa = anio_inicio
            fecha_inicio = pd.Timestamp(year=anio_inicio_etapa,
                                        month=ventana['mes_inicio'],
                                        day=ventana['dia_inicio'])

            if ventana['mes_fin'] <= 6:
                anio_fin_etapa = anio_inicio + 1
            else:
                anio_fin_etapa = anio_inicio
            fecha_fin = pd.Timestamp(year=anio_fin_etapa,
                                    month=ventana['mes_fin'],
                                    day=ventana['dia_fin'])

            if df_depto is None:
                precipitacion_total = temperatura_media = radiacion_solar_promedio = np.nan
                amplitud_termica = deficit_hidrico_extremo = np.nan
                estres_termico = helada_agro = np.nan
            else:
                df_filtrado = df_depto[
                    (df_depto['Fecha'] >= fecha_inicio) &
                    (df_depto['Fecha'] <= fecha_fin)
                ]

                precipitacion_total = df_filtrado['PRECTOTCORR'].sum()
                temperatura_media = df_filtrado['T2M'].mean()
                radiacion_solar_promedio = df_filtrado['ALLSKY_SFC_SW_DWN'].mean()
                amplitud_termica = (df_filtrado['T2M_MAX'] - df_filtrado['T2M_MIN']).mean()

                condicion = df_filtrado['PRECTOTCORR'] < 1
                grupos = (~condicion).cumsum()
                rachas = condicion.groupby(grupos).sum()
                deficit_hidrico_extremo = rachas.max() if len(rachas) > 0 else 0

                estres_termico = 0
                helada_agro = 0
                cultivo = fila['Cultivo']

                if cultivo == 'maíz':
                    condicion = df_filtrado['T2M_MAX'] > 32
                    grupos = (~condicion).cumsum()
                    rachas = condicion.groupby(grupos).sum()
                    estres_termico = rachas.max() if len(rachas) > 0 else 0
                elif cultivo in ('soja 1ra', 'soja 2da', 'girasol'):
                    condicion = df_filtrado['T2M_MAX'] > 35
                    grupos = (~condicion).cumsum()
                    rachas = condicion.groupby(grupos).sum()
                    estres_termico = rachas.max() if len(rachas) > 0 else 0
                elif cultivo == 'trigo total':
                    if ventana['etapa'] == 'floracion':
                        condicion = df_filtrado['T2M_MIN'] <= 2
                        grupos = (~condicion).cumsum()
                        rachas = condicion.groupby(grupos).sum()
                        helada_agro = rachas.max() if len(rachas) > 0 else 0

            calculos = {
                'Cultivo': fila['Cultivo'],
                'Departamento_id': depto_id,
                'Campaña': campania,
                'Etapa': ventana['etapa'],
                'Fecha_Inicio': fecha_inicio,
                'Fecha_Fin': fecha_fin,
                'precipitacion_total': precipitacion_total,
                'deficit_hidrico_extremo': deficit_hidrico_extremo,
                'temperatura_media': temperatura_media,
                'radiacion_solar_promedio': radiacion_solar_promedio,
                'amplitud_termica': amplitud_termica,
                'estres_termico': estres_termico,
                'helada_agro': helada_agro
            }

            resultados.append(calculos)

    return resultados


def calcular_features_ndvi(df_camp, ndvi_por_depto, df_ventanas):
    """Extraída tal cual del notebook 03 -- mismo criterio que la de clima."""
    if df_camp is None or ndvi_por_depto is None or df_ventanas is None:
        raise ValueError("Los parámetros de entrada no pueden ser None")

    resultados = []

    for _, fila in df_camp.iterrows():

        cultivo_ventana = fila['cultivo_ventana']
        zona = fila['Zona']
        depto_id = fila['Departamento_id']
        anio_inicio = fila['Anio_Inicio']
        campania = fila['Campaña']

        ventanas_filtradas = df_ventanas[
            (df_ventanas['cultivo'] == cultivo_ventana) &
            (df_ventanas['zona'] == zona)
        ]

        df_depto = ndvi_por_depto.get(depto_id)

        for _, ventana in ventanas_filtradas.iterrows():

            if ventana['mes_inicio'] <= 6:
                anio_inicio_etapa = anio_inicio + 1
            else:
                anio_inicio_etapa = anio_inicio
            fecha_inicio = pd.Timestamp(year=anio_inicio_etapa,
                                        month=ventana['mes_inicio'],
                                        day=ventana['dia_inicio'])

            if ventana['mes_fin'] <= 6:
                anio_fin_etapa = anio_inicio + 1
            else:
                anio_fin_etapa = anio_inicio
            fecha_fin = pd.Timestamp(year=anio_fin_etapa,
                                    month=ventana['mes_fin'],
                                    day=ventana['dia_fin'])

            if df_depto is None:
                ndvi_promedio, n_observaciones = np.nan, 0
            else:
                df_filtrado = df_depto[
                    (df_depto['fecha_media'] >= fecha_inicio) &
                    (df_depto['fecha_media'] <= fecha_fin)
                ]
                ndvi_promedio = df_filtrado['ndvi'].mean()
                n_observaciones = len(df_filtrado)

            calculos = {
                'Cultivo': fila['Cultivo'],
                'Departamento_id': depto_id,
                'Campaña': campania,
                'Etapa': ventana['etapa'],
                'ndvi_promedio': ndvi_promedio,
                'ndvi_n_observaciones': n_observaciones,
            }

            resultados.append(calculos)

    return resultados
