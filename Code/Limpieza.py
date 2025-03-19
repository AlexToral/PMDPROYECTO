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

# 5. Seleccionar solo las columnas esenciales para el análisis de Home Advantage
soccer_df = soccer_df[['Country', 'League', 'Team', 'Year', 'HomeRatio', 'AwayGoalsDiff']]

# 6. En country_facts_df, conservar solo las columnas relevantes
relevant_country_columns = ['Country', 'PopDensity', 'GDP', 'Attendance']
country_facts_df = country_facts_df[relevant_country_columns]

# 7. Convertir columnas numéricas y rellenar valores faltantes con la mediana
numeric_cols_country = ['PopDensity', 'GDP', 'Attendance']
for col in numeric_cols_country:
    country_facts_df[col] = pd.to_numeric(country_facts_df[col], errors='coerce')
    country_facts_df[col] = country_facts_df[col].fillna(country_facts_df[col].median())

# 8. Normalizar las variables numéricas del dataset de países
scaler = MinMaxScaler()
country_facts_df[numeric_cols_country] = scaler.fit_transform(country_facts_df[numeric_cols_country])

# 9. Integrar ambos datasets a través de la columna 'Country'
merged_df = soccer_df.merge(country_facts_df, on='Country', how='left')

# 10. Rellenar valores faltantes en columnas categóricas y numéricas
cat_cols_merged = merged_df.select_dtypes(include=['object']).columns
for col in cat_cols_merged:
    merged_df[col] = merged_df[col].fillna('Desconocido')

num_cols_merged = merged_df.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols_merged:
    merged_df[col] = merged_df[col].fillna(merged_df[col].median())

# 11. Guardar el dataset procesado con la información necesaria para el análisis
merged_df.to_csv("Data/Processed_HomeAdvantage_Data.csv", index=False)
print("¡Limpieza de datos para el análisis del Home Advantage completada con éxito!")
