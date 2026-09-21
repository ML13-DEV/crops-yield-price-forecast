import pandas as pd

df = pd.read_excel(r'data\raw\producciones_camapanias.xlsx')

print(df[df['Campaña']=='2025/2026'])