# Crop Yield & Price Forecasting — Argentina

A personal forecasting project that predicts **crop yield** (qq/ha) and explores **commodity price** forecasting for Argentina's main crops, at **department-level resolution** (158 real department polygons — not approximate zones).

> Built with the **CRISP-DM** methodology, using public and official data sources: MAGyP, NASA POWER, NASA AppEEARS/MODIS and georef-ar-api.

---

## Table of Contents

- [Motivation](#motivation)
- [Demo](#demo)
- [Crops and geographic resolution](#crops-and-geographic-resolution)
- [Data sources](#data-sources)
- [Methodology](#methodology)
- [Final model and metrics](#final-model-and-metrics)
- [Field-level validation (limitations)](#field-level-validation-limitations)
- [Repository structure](#repository-structure)
- [Installation and usage](#installation-and-usage)
- [Predicting a single field](#predicting-a-single-field)
- [Next steps](#next-steps)
- [Author](#author)

---

## Motivation

Argentina is one of the world's largest agricultural producers and exporters, yet many sowing, commercialization and risk-hedging decisions are still made with information fragmented at the province or broad-zone level. This project estimates expected crop yield **at the department level**, combining historical production data with climate and satellite vegetation variables, aiming to build a decision-support tool for both producers and sector analysts.

## Demo

The interactive app (Streamlit) allows you to:
- Select a department, crop and campaign on a map with all 158 real department polygons.
- Compare the model's historical prediction against the real reported yield.
- Understand **which variables drove each individual prediction** (SHAP).
- Compare a naive price model against a more sophisticated one.
- Review metrics and **model limitations transparently**, including a call-to-action for producers to contribute real field-level data.

```bash
streamlit run app.py
```

## Crops and geographic resolution

| Crop | Treatment |
|---|---|
| Soybean 1st (Soja 1ra) | Treated as its own category (not merged with 2nd-season soybean) |
| Soybean 2nd (Soja 2da) | Treated as its own category |
| Corn (Maíz) | — |
| Wheat (Trigo) | — |
| Sunflower (Girasol) | — |
| Peanut (Maní) | Added as a 5th crop, with its own sub-zoning (West / Core / East) within Córdoba's peanut belt |

The model's geographic unit is **158 real departments** (official polygons, not bounding boxes), selected by cumulative sown-area coverage (~95% of national sown area, 2015–2024, across all six crops).

## Data sources

| Source | Use | Detail |
|---|---|---|
| **MAGyP** — `estimaciones-agricolas` | Target (real yield) | Sown/harvested area, production and yield by department, crop and campaign |
| **NASA POWER** (`point` API) | Climate | Precipitation, temperature (mean/max/min), solar radiation, relative humidity — daily series per department centroid, 2000–present |
| **NASA AppEEARS / MODIS (MOD13Q1.061)** | Vegetation | NDVI and pixel reliability, aggregated by department over key phenological windows |
| **georef-ar-api** (`datos.gob.ar`) | Geometry | Official Argentine department polygons for exact spatial resolution and point-in-polygon lookups |
| **NOAA CPC** — Oceanic Niño Index | Evaluated and discarded | Tested as a climate feature (El Niño/La Niña); added no net value to the final model (see below) |

## Methodology

Built under **CRISP-DM**:

1. **Business understanding**: estimate agricultural yield at department-level resolution to support decisions by producers and analysts.
2. **Data understanding & preparation**: cleaning and unifying the MAGyP dataset (`kg/ha → qq/ha` conversion, lost-area handling, department ID normalization), downloading and validating climate (158 departments × 6 parameters × 26+ years) and NDVI (158 departments × crop-specific phenological windows).
3. **Feature engineering**:
   - Crop- and sub-zone-specific phenological windows (flowering, grain filling) — including a dedicated peanut zoning built with guidance from an agronomist consulted for the project.
   - Thermal and water stress variables calculated **conditionally per crop** (different thresholds for corn, soybean, sunflower and wheat).
   - **Department × crop target encoding** (`depto_target_encoding`): historical average yield, computed strictly on the training fold to avoid leakage.
4. **Modeling**: Random Forest, with `GroupKFold` grouped by campaign (`Campaña`) to prevent information leakage across agricultural years. Hyperparameters tuned via `RandomizedSearchCV`.
5. **Evaluation**: explicitly tested adding an ENSO/ONI climate index (El Niño–La Niña) as a feature. After a 2×2 ablation matrix (with/without target encoding × with/without ONI, all combinations tuned symmetrically), ONI was confirmed to add **no net value** — even combined with target encoding — and was dropped from the production model.
6. **Deployment**: serialized model (`.pkl`) + a point-level (single-field) prediction script, consumed by the Streamlit demo.

## Final model and metrics

**Final configuration**: Random Forest + PAS Zone + Department Target Encoding — **without ONI** — 40 features.

n_estimators=300, max_depth=20, min_samples_leaf=2, min_samples_split=2, max_features=None


| Crop | MAE (qq/ha) | R² | Improvement vs. baseline |
|---|---|---|---|
| Sunflower | 3.33 | 0.366 | +9.7% |
| Corn | 10.43 | 0.601 | +24.6% |
| Peanut | 5.61 | 0.135 | +12.7% |
| Soybean 1st | 4.53 | 0.535 | +21.9% |
| Soybean 2nd | 4.26 | 0.435 | +18.3% |
| Wheat | 5.58 | 0.631 | +14.0% |

## Field-level validation (limitations)

The model was validated against **15 real fields** from a producer (2025/2026 campaign, georeferenced coordinates, crop, sowing date and producer-reported real yield).

**Key finding**: the model predicts the department average well, but **severely compresses variance between fields within the same department** — it cannot distinguish a high-quality field from a low-quality one inside the same zone.

| Metric (10 corn fields) | Value |
|---|---|
| MAE | 43.15 qq/ha |
| Real std. dev. | 24.5 |
| Predicted std. dev. | 6.0 |
| Real vs. predicted correlation | 0.46 |

Target encoding was ruled out as the cause (a model without that feature did not improve the result). The root cause is structural: **the model never saw individual field-level training examples**, only department averages. Closing this gap would require combine-harvester yield-monitor data at the field level as a training target — not simply more NDVI or climate data, which is already available at that resolution.

This limitation is shown transparently in the demo app, along with a call-to-action for producers to contribute real field-level data.

## Repository structure

crops-yield-price-forecast/  
├── data/  
│ ├── raw/  
│ │ ├── clima_departamentos/ # clima_{depto_id}.csv (NASA POWER)  
│ │ ├── ndvi_departamentos/ # NDVI + pixel_reliability (AppEEARS)  
│ │ ├── estimaciones-agricolas-.csv # MAGyP dataset  
│ │ ├── departamentos_completo.geojson  
│ │ ├── departamentos_boxes.geojson # 158 filtered departments  
│ │ ├── departamentos_filtrados.csv  
│ │ ├── ventanas_fenologicas.csv  
│ │ ├── mapa_departamento_zona.csv  
│ │ ├── zona_mani_por_departamento.csv  
│ │ └── oni.ascii.txt # NOAA CPC (evaluated, not used in prod.)  
│ └── processed/  
│ └── dataset_modelo_rendimiento.csv  
├── models/  
│ ├── modelo_rendimiento_rf.pkl  
│ ├── features_modelo_rendimiento.json  
│ └── promedio_rendimiento_departamento_cultivo.csv  
├── notebooks/  
│ ├── 02_.ipynb # data cleaning and unification  
│ ├── 03_.ipynb # feature engineering (climate, NDVI, phenological windows)  
│ └── 05_.ipynb # modeling, tuning and evaluation  
├── scripts/  
│ ├── obtener_poligonos_departamentos.py  
│ ├── data_collection.py # NASA POWER climate download  
│ ├── validar_clima_departamentos.py  
│ ├── enviar_requests_appeears.py # NDVI download (AppEEARS)  
│ ├── revisar_y_descargar_ndvi.py  
│ ├── mapa_departamento_zona.py  
│ ├── zona_mani_por_departamento.py  
│ ├── features_comunes.py # shared notebook/script features  
│ └── prediccion_campo.py # field-level (lat/lon) prediction  
├── app.py # Streamlit demo  
├── requirements.txt  
├── .gitignore  
└── README.md  

> Adjust notebook names if they don't exactly match yours.

## Installation and usage

```bash
git clone https://github.com/ML13-DEV/crops-yield-price-forecast.git
cd crops-yield-price-forecast

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

streamlit run app.py
```

Raw data (climate and NDVI per department) **is not versioned in git** due to its size — it can be downloaded with `scripts/data_collection.py` (NASA POWER) and `scripts/enviar_requests_appeears.py` + `scripts/revisar_y_descargar_ndvi.py` (NDVI).

## Predicting a single field

```python
from scripts.prediccion_campo import calcular_prediccion_campo

result = calcular_prediccion_campo(
    lat=-33.87, lon=-63.10,
    cultivo="maíz",
    campania="2025/2026",
    fecha_siembra="2025-10-15",
    ...  # see the script's docstring for the remaining parameters
)
```

The script automatically resolves the department via point-in-polygon lookup, the matching PAS/peanut zone, and the target encoding — from just latitude/longitude.

## Next steps

- Incorporate combine-harvester yield-monitor data at the field level to reduce the variance compression found in validation.
- Add historical peanut prices to the price module.
- Evaluate Sentinel-2 as a higher-resolution predictor for campaigns from 2015 onward.

## Author

**Manuel Lombardi** — Data Analyst / Data Scientist in training
[GitHub](https://github.com/ML13-DEV) · [LinkedIn](https://linkedin.com/in/manuel-lombardi-572685341)
