import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset procesado
data_path = "Data/Processed_Sports_Data.csv"
merged_df = pd.read_csv(data_path)

# Título de la aplicación
st.title("Análisis de la Ventaja de Jugar en Casa en Deportes")

# Sección 1: Vista previa del dataset
st.subheader("Vista previa del dataset procesado")
st.dataframe(merged_df.head())

# Sección 2: Distribución de HomeRatio antes y después de discretización
st.subheader("Distribución de HomeRatio")
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Antes de la discretización
ax[0].hist(merged_df['HomeRatio'], bins=20, color='blue', alpha=0.7)
ax[0].set_title('Distribución de HomeRatio antes de Discretización')
ax[0].set_xlabel('HomeRatio')
ax[0].set_ylabel('Frecuencia')

# Después de la discretización
sns.countplot(data=merged_df, x='HomeRatio_Category', palette='viridis', ax=ax[1])
ax[1].set_title('Categorías de HomeRatio después de Discretización')
ax[1].set_xlabel('Categoría')
ax[1].set_ylabel('Frecuencia')

st.pyplot(fig)

# Sección 3: Relación entre GDP y HomeRatio
st.subheader("Relación entre GDP y HomeRatio")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=merged_df, x='GDP', y='HomeRatio', hue='HomeRatio_Category', alpha=0.7)
ax.set_title('Relación entre GDP y HomeRatio')
ax.set_xlabel('GDP (Normalizado)')
ax.set_ylabel('HomeRatio')
st.pyplot(fig)

# Sección 4: Relación entre Asistencia y HomeRatio
st.subheader("Relación entre Asistencia y HomeRatio")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=merged_df, x='Attendance', y='HomeRatio', hue='HomeRatio_Category', alpha=0.7)
ax.set_title('Relación entre Asistencia y HomeRatio')
ax.set_xlabel('Asistencia (Normalizada)')
ax.set_ylabel('HomeRatio')
st.pyplot(fig)

# Sección 5: Análisis por país
st.subheader("Análisis de HomeRatio por País")
selected_country = st.selectbox("Selecciona un país", merged_df['Country'].unique())
country_data = merged_df[merged_df['Country'] == selected_country]
st.write(country_data[['League', 'Team', 'Year', 'HomeRatio', 'AwayGoalsDiff']].sort_values(by='Year', ascending=False))
