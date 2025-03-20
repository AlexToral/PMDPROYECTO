import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 1. Cargar los datasets
soccer_df = pd.read_csv("Data/SoccerLeagues.csv")
country_facts_df = pd.read_csv("Data/Country_facts.csv")

# 2. Eliminar columnas irrelevantes en soccer_df
if 'Unnamed: 15' in soccer_df.columns:
    soccer_df.drop(columns=['Unnamed: 15'], inplace=True)

# 3. Corregir nombres de países para coincidencia
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

# 4. Manejo de valores negativos en HomeRatio y AwayGoalsDiff
soccer_df.loc[soccer_df['HomeRatio'] < 0, 'HomeRatio'] = 0
soccer_df.loc[soccer_df['AwayGoalsDiff'] < 0, 'AwayGoalsDiff'] = 0

# 5. Seleccionar solo las columnas esenciales para el análisis
soccer_df = soccer_df[['Country', 'League', 'Team', 'Year', 'HomeRatio', 'AwayGoalsDiff']]

# 6. Seleccionar columnas relevantes del dataset de países
# Usaremos FIFA_Rank si está disponible, de lo contrario se usan factores económicos (como respaldo)
if 'FIFA_Rank' in country_facts_df.columns:
    relevant_country_columns = ['Country', 'FIFA_Rank']
else:
    relevant_country_columns = ['Country', 'PopDensity', 'GDP', 'Attendance']

country_facts_df = country_facts_df[relevant_country_columns]

# 7. Procesar la variable FIFA_Rank (si existe)
if 'FIFA_Rank' in country_facts_df.columns:
    country_facts_df['FIFA_Rank'] = pd.to_numeric(country_facts_df['FIFA_Rank'], errors='coerce')
    country_facts_df['FIFA_Rank'] = country_facts_df['FIFA_Rank'].fillna(country_facts_df['FIFA_Rank'].median())
    # Normalizar la variable (esto la convierte en un valor entre 0 y 1)
    scaler = MinMaxScaler()
    country_facts_df[['FIFA_Rank']] = scaler.fit_transform(country_facts_df[['FIFA_Rank']])
else:
    # En caso de no tener FIFA_Rank, se procesan las variables económicas (como en versiones anteriores)
    numeric_cols = ['PopDensity', 'GDP', 'Attendance']
    for col in numeric_cols:
        country_facts_df[col] = pd.to_numeric(country_facts_df[col], errors='coerce')
        country_facts_df[col] = country_facts_df[col].fillna(country_facts_df[col].median())
    scaler = MinMaxScaler()
    country_facts_df[numeric_cols] = scaler.fit_transform(country_facts_df[numeric_cols])

# 8. Integrar ambos datasets a través de la columna 'Country'
merged_df = soccer_df.merge(country_facts_df, on='Country', how='left')

# 9. Rellenar valores faltantes en columnas categóricas y numéricas
for col in merged_df.select_dtypes(include=['object']).columns:
    merged_df[col] = merged_df[col].fillna('Desconocido')
for col in merged_df.select_dtypes(include=['int64', 'float64']).columns:
    merged_df[col] = merged_df[col].fillna(merged_df[col].median())

# 10. Guardar el dataset procesado
merged_df.to_csv("Data/Processed_HomeAdvantage_FIFA_Data.csv", index=False)
print("¡Limpieza de datos para el análisis del Home Advantage con FIFA Rank completada con éxito!")
