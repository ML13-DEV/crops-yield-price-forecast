import json
import os

archivo_entrada = '02_data_understanding (3).ipynb'
archivo_salida = '02_data_understanding_Actualizado.ipynb'

with open(archivo_entrada, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# ---------------------------------------------------------
# CELDA 13: Carga de Clima por Departamentos
# ---------------------------------------------------------
nb['cells'][13]['source'] = [
    "dfs = []\n",
    "for file in os.listdir(r\"..\\data\\raw\\clima_departamentos\"):\n",
    "    if file.endswith(\".csv\"):\n",
    "        file_path = os.path.join(r\"..\\data\\raw\\clima_departamentos\", file)\n",
    "        temp_df = pd.read_csv(file_path)\n",
    "        \n",
    "        # Mapeamos Departamento a Zona para que los gráficos del EDA sigan funcionando\n",
    "        if 'Unnamed: 0' in temp_df.columns:\n",
    "            temp_df = temp_df.rename(columns={'Unnamed: 0': 'Fecha'})\n",
    "            \n",
    "        if 'Departamento' in temp_df.columns:\n",
    "            temp_df['Zona'] = temp_df['Departamento'].astype(str)\n",
    "        elif 'Departamento_id' in temp_df.columns:\n",
    "            temp_df['Zona'] = temp_df['Departamento_id'].astype(str)\n",
    "            \n",
    "        temp_df['Fecha'] = pd.to_datetime(temp_df['Fecha'], errors='coerce')\n",
    "        dfs.append(temp_df)\n",
    "\n",
    "df_clima = pd.concat(dfs, ignore_index=True)\n",
    "print(f\"Total registros clima: {len(df_clima)}\")\n",
    "df_clima.head()"
]

# ---------------------------------------------------------
# CELDA 15: Carga de Estimaciones Agrícolas
# ---------------------------------------------------------
nb['cells'][15]['source'] = [
    "df_camp = pd.read_csv(r\"..\\data\\raw\\estimaciones-agricolas-2026-03.csv\", encoding='latin1')\n",
    "\n",
    "# Renombramos columnas nuevas para mantener compatibilidad absoluta con el EDA\n",
    "df_camp = df_camp.rename(columns={\n",
    "    'departamento': 'Zona', \n",
    "    'campania': 'Campaña',\n",
    "    'superficie_sembrada_ha': 'Sembrado(Ha)',\n",
    "    'superficie_cosechada_ha': 'Cosechado(Ha)',\n",
    "    'produccion_tm': 'Producción(MTn)'\n",
    "})\n",
    "\n",
    "df_camp['Cultivo'] = df_camp['cultivo'].astype(str).str.title()\n",
    "df_camp['Rinde(qq/Ha)'] = df_camp['rendimiento_kgxha'] / 100\n",
    "df_camp['Perdido(Ha)'] = df_camp['Sembrado(Ha)'] - df_camp['Cosechado(Ha)']\n",
    "df_camp = df_camp[df_camp['Zona'].str.upper() != 'TOTAL'].copy()\n",
    "\n",
    "df_camp.head(5)"
]

# CELDA 16: Anulamos el renombre viejo
nb['cells'][16]['source'] = ["# (Celda automatizada: la limpieza y renombre se hizo en la celda superior)"]

# ---------------------------------------------------------
# CELDA 18: Carga de Dólar
# ---------------------------------------------------------
nb['cells'][18]['source'] = [
    "df_dolard = pd.read_excel(r\"..\\data\\raw\\cotizaciones_dolar.xlsx\")\n",
    "df_dolard['Fecha'] = pd.to_datetime(df_dolard['Fecha'], errors='coerce').dt.normalize()\n",
    "df_dolard['Divisa Compra'] = pd.to_numeric(df_dolard['Divisa Compra'], errors='coerce')\n",
    "df_dolard = df_dolard[['Fecha', 'Divisa Compra']]\n",
    "df_dolard.head()"
]

# CELDA 20: Anulamos el filtrado viejo
nb['cells'][20]['source'] = ["# (Celda automatizada: el filtrado se hizo en la celda superior)"]

# ---------------------------------------------------------
# CELDA 22: Carga de Precios y Conversión a USD
# ---------------------------------------------------------
nb['cells'][22]['source'] = [
    "base_path = r\"..\\data\\raw\\precios_cultivos\"\n",
    "dfs_precios = {}\n",
    "for folder in sorted(os.listdir(base_path)):\n",
    "    folder_path = os.path.join(base_path, folder)\n",
    "    if not os.path.isdir(folder_path): continue\n",
    "\n",
    "    parts = []\n",
    "    for fname in sorted(os.listdir(folder_path)):\n",
    "        fpath = os.path.join(folder_path, fname)\n",
    "        if fname.lower().endswith((\".csv\", \".xlsx\", \".xls\")):\n",
    "            try:\n",
    "                if fname.lower().endswith(\".csv\"): df_part = pd.read_csv(fpath)\n",
    "                else: df_part = pd.read_excel(fpath)\n",
    "                \n",
    "                if 'Fecha de operación' in df_part.columns:\n",
    "                    df_part = df_part.rename(columns={'Fecha de operación': 'Fecha'})\n",
    "                if 'Precio' in df_part.columns:\n",
    "                    df_part = df_part.rename(columns={'Precio': 'Precio_Pesos'})\n",
    "                \n",
    "                df_part['Fecha'] = pd.to_datetime(df_part['Fecha']).dt.normalize()\n",
    "                df_part['Precio_Pesos'] = pd.to_numeric(df_part['Precio_Pesos'], errors='coerce')\n",
    "                parts.append(df_part)\n",
    "            except: pass\n",
    "\n",
    "    if parts:\n",
    "        df_concat = pd.concat(parts, ignore_index=True)\n",
    "        df_concat = df_concat.groupby('Fecha')['Precio_Pesos'].mean().reset_index()\n",
    "        \n",
    "        # Merge con Dolar para pasarlo a USD y rellenar nulos ('S/C' o fines de semana)\n",
    "        df_concat = pd.merge(df_concat, df_dolard, on='Fecha', how='left')\n",
    "        df_concat['Precio'] = df_concat['Precio_Pesos'] / df_concat['Divisa Compra']\n",
    "        df_concat['Precio'] = df_concat['Precio'].ffill().bfill()\n",
    "        \n",
    "        dfs_precios[folder] = df_concat\n",
    "        var_name = \"df_precios_\" + \"\".join(ch if ch.isalnum() else \"_\" for ch in folder)\n",
    "        globals()[var_name] = df_concat\n",
    "        print(f\"{folder} procesado -> USD (NaNs imputados exitosamente)\")"
]

with open(archivo_salida, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"¡Éxito! El archivo {archivo_salida} se creó correctamente.")