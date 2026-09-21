import pandas as pd
import os
for depto_id in ['06476', '22105', '82021']:
    os.remove(f'../data/raw/clima_departamentos/clima_{depto_id}.csv')