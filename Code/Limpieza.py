import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 1. Cargar datasets
soccer_df = pd.read_csv("Data/SoccerLeagues.csv")
country_facts_df = pd.read_csv("Data/Country_facts.csv")

# 2. Eliminar columnas irrelevantes
if 'Unnamed: 15' in soccer_df.columns:
    soccer_df.drop(columns=['Unnamed: 15'], inplace=True)

# 3. Corregir nombres de países
country_name_corrections = {
    'Czech': 'Czech Republic',
    'United': 'United States',
    'New': 'New Zealand',
    'El': 'El Salvador',
    'Puerto': 'Puerto Rico',
    'Saudi': 'Saudi Arabia',
    'South': 'South Korea',
    'Trinidad': 'Trinidad and Tobago'
}
soccer_df['Country'] = soccer_df['Country'].replace(country_name_corrections)

# 4. Corregir valores negativos
soccer_df.loc[soccer_df['HomeRatio'] < 0, 'HomeRatio'] = 0
soccer_df.loc[soccer_df['AwayGoalsDiff'] < 0, 'AwayGoalsDiff'] = 0

# 5. Crear nuevas features de rendimiento
soccer_df['Games'] = soccer_df[['HomeWins', 'HomeDraw', 'HomeLoss']].sum(axis=1)

# Evitar división por cero
soccer_df['Games'] = soccer_df['Games'].replace(0, 1)

soccer_df['WinRate_Local'] = soccer_df['HomeWins'] / soccer_df['Games']
soccer_df['DrawRate_Local'] = soccer_df['HomeDraw'] / soccer_df['Games']
soccer_df['LossRate_Local'] = soccer_df['HomeLoss'] / soccer_df['Games']
soccer_df['WinRate_Visitante'] = soccer_df['AwayWins'] / soccer_df['Games']
soccer_df['LossRate_Visitante'] = soccer_df['AwayLoss'] / soccer_df['Games']
soccer_df['GoalDiff_Visitante_Prom'] = soccer_df['AwayGoalsDiff'] / soccer_df['Games']

# 6. Seleccionar columnas esenciales
soccer_df = soccer_df[[
    'Country', 'League', 'Team', 'Year',
    'HomeRatio', 'AwayGoalsDiff',
    'WinRate_Local', 'DrawRate_Local', 'LossRate_Local',
    'WinRate_Visitante', 'LossRate_Visitante', 'GoalDiff_Visitante_Prom'
]]

# 7. Preparar datos de países
if 'FIFA_Rank' in country_facts_df.columns:
    relevant_country_columns = ['Country', 'FIFA_Rank']
else:
    relevant_country_columns = ['Country', 'PopDensity', 'GDP', 'Attendance']

country_facts_df = country_facts_df[relevant_country_columns]

# 8. Normalizar FIFA_Rank
if 'FIFA_Rank' in country_facts_df.columns:
    country_facts_df['FIFA_Rank'] = pd.to_numeric(country_facts_df['FIFA_Rank'], errors='coerce')
    country_facts_df['FIFA_Rank'] = country_facts_df['FIFA_Rank'].fillna(country_facts_df['FIFA_Rank'].median())
    scaler = MinMaxScaler()
    country_facts_df[['FIFA_Rank']] = scaler.fit_transform(country_facts_df[['FIFA_Rank']])
else:
    numeric_cols = ['PopDensity', 'GDP', 'Attendance']
    for col in numeric_cols:
        country_facts_df[col] = pd.to_numeric(country_facts_df[col], errors='coerce')
        country_facts_df[col] = country_facts_df[col].fillna(country_facts_df[col].median())
    scaler = MinMaxScaler()
    country_facts_df[numeric_cols] = scaler.fit_transform(country_facts_df[numeric_cols])

# 9. Unir datasets
merged_df = soccer_df.merge(country_facts_df, on='Country', how='left')

# 10. Rellenar nulos
for col in merged_df.select_dtypes(include=['object']).columns:
    merged_df[col] = merged_df[col].fillna('Desconocido')
for col in merged_df.select_dtypes(include=['int64', 'float64']).columns:
    merged_df[col] = merged_df[col].fillna(merged_df[col].median())

# 11. Exportar
merged_df.to_csv("Data/Processed_HomeAdvantage_FIFA_Data.csv", index=False)
print("✅ Dataset procesado con nuevas features futboleras guardado con éxito.")
