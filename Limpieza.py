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

# 5. Imputar valores faltantes en columnas NUMÉRICAS de country_facts_df
numeric_columns = [
    'PopDensity', 'Coastline', 'Net migration', 'Infant_mortality',
    'GDP', 'Literacy', 'Phones', 'Arable', 'Crops', 'Other', 'Climate',
    'Birthrate', 'Deathrate', 'Agriculture', 'Industry', 'Service',
    'Attendance'
]

# Para evitar warnings, se recomienda reasignar la serie:
for col in numeric_columns:
    # Si la columna existe en el DataFrame
    if col in country_facts_df.columns:
        # Convertir a numérico (por si se coló algún texto)
        country_facts_df[col] = pd.to_numeric(country_facts_df[col], errors='coerce')
        # Rellenar con la mediana
        country_facts_df[col] = country_facts_df[col].fillna(country_facts_df[col].median())

# 6. Normalización de variables numéricas con MinMaxScaler
scaler = MinMaxScaler()
country_facts_df[numeric_columns] = scaler.fit_transform(country_facts_df[numeric_columns])

# 7. Discretización de HomeRatio
soccer_df['HomeRatio_Category'] = pd.qcut(
    soccer_df['HomeRatio'], q=3, labels=['Baja', 'Media', 'Alta']
)

# 8. Integración de los datasets
merged_df = soccer_df.merge(country_facts_df, on='Country', how='left')

# 9. Identificar qué columnas son categóricas y cuáles son numéricas en merged_df
num_cols_merged = merged_df.select_dtypes(include=['int64','float64']).columns
cat_cols_merged = merged_df.select_dtypes(include=['object']).columns

# Si en el merge te salieron NaN en columnas categóricas, rellénalas con 'Desconocido'
# (pero no toques las numéricas con 'Desconocido')
for col in cat_cols_merged:
    merged_df[col] = merged_df[col].fillna('Desconocido')

# 10. Si quedó alguna columna numérica con NaN, rellénala con la mediana
for col in num_cols_merged:
    merged_df[col] = merged_df[col].fillna(merged_df[col].median())

# 11. Guardar el dataset procesado
merged_df.to_csv("Data/Processed_Sports_Data.csv", index=False)

print("¡Limpieza completada con éxito!")
