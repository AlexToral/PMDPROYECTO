import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset procesado
data_path = "Data/Processed_HomeAdvantage_Data.csv"
df = pd.read_csv(data_path)

# Título de la aplicación
st.title("Impacto de la Ventaja de Jugar en Casa en el Desempeño de Equipos de Fútbol")

# Sección 1: Vista previa de los datos
st.subheader("Vista previa de los datos")
st.dataframe(df.head())

# Sección 2: Tendencia del Home Advantage a lo largo del tiempo
st.subheader("Tendencia del Home Advantage a lo largo del tiempo")
trend_df = df.groupby('Year', as_index=False)['HomeRatio'].mean()
fig1, ax1 = plt.subplots()
sns.lineplot(data=trend_df, x='Year', y='HomeRatio', marker='o', ax=ax1)
ax1.set_title('Promedio de HomeRatio por Año')
ax1.set_xlabel('Año')
ax1.set_ylabel('HomeRatio Promedio')
st.pyplot(fig1)

# Sección 3: Comparación de Home Advantage por país
st.subheader("Comparación de Home Advantage por País")
country_avg = df.groupby('Country', as_index=False)['HomeRatio'].mean().sort_values(by='HomeRatio', ascending=False)
fig2, ax2 = plt.subplots(figsize=(10,6))
sns.barplot(data=country_avg, x='HomeRatio', y='Country', palette='viridis', ax=ax2)
ax2.set_title('Promedio de HomeRatio por País')
ax2.set_xlabel('HomeRatio Promedio')
ax2.set_ylabel('País')
st.pyplot(fig2)

# Sección 4: Relación entre factores económicos y Home Advantage

st.subheader("Relación entre Factores Económicos y Home Advantage")

# Relación entre GDP y HomeRatio
fig3, ax3 = plt.subplots(figsize=(10,6))
sns.scatterplot(data=df, x='GDP', y='HomeRatio', hue='Country', alpha=0.7, ax=ax3)
ax3.set_title('Relación entre GDP (Normalizado) y HomeRatio')
ax3.set_xlabel('GDP (Normalizado)')
ax3.set_ylabel('HomeRatio')
st.pyplot(fig3)

# Relación entre Attendance y HomeRatio
fig4, ax4 = plt.subplots(figsize=(10,6))
sns.scatterplot(data=df, x='Attendance', y='HomeRatio', hue='Country', alpha=0.7, ax=ax4)
ax4.set_title('Relación entre Attendance (Normalizado) y HomeRatio')
ax4.set_xlabel('Attendance (Normalizado)')
ax4.set_ylabel('HomeRatio')
st.pyplot(fig4)

# Relación entre PopDensity y HomeRatio
fig5, ax5 = plt.subplots(figsize=(10,6))
sns.scatterplot(data=df, x='PopDensity', y='HomeRatio', hue='Country', alpha=0.7, ax=ax5)
ax5.set_title('Relación entre PopDensity (Normalizado) y HomeRatio')
ax5.set_xlabel('PopDensity (Normalizado)')
ax5.set_ylabel('HomeRatio')
st.pyplot(fig5)

# Sección 5: Análisis Detallado por País y Liga
st.subheader("Análisis Detallado por País y Liga")
selected_country = st.selectbox("Selecciona un país", df['Country'].unique())
selected_league = st.selectbox("Selecciona una liga", df[df['Country'] == selected_country]['League'].unique())
filtered_df = df[(df['Country'] == selected_country) & (df['League'] == selected_league)]
st.write(filtered_df.sort_values(by='Year', ascending=False)[['Year', 'Team', 'HomeRatio', 'AwayGoalsDiff']])
