import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Cargar el dataset procesado que ahora incluye FIFA_Rank
data_path = "Data/Processed_HomeAdvantage_FIFA_Data.csv"
soccer_path = "Data/SoccerLEagues.csv"
country_path = "Data/Country_Facts.csv"

df = pd.read_csv(data_path)
df1 = pd.read_csv(soccer_path)
df2 = pd.read_csv(country_path)

st.title("Impacto de la Ventaja de Jugar en Casa y su Relación con FIFA Rank")

st.title("Datos antes de la limpieza")
st.dataframe(df1)
st.dataframe(df2)

# Sección 1: Vista previa de los datos
st.subheader("Vista previa de los datos (ya clean)")
st.dataframe(df.head())

# Sección 2: Tendencia del Home Advantage a lo largo del tiempo
st.subheader("Tendencia del Home Advantage a lo largo del tiempo")
trend_df = df.groupby('Year', as_index=False)['HomeRatio'].mean()
fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.lineplot(data=trend_df, x='Year', y='HomeRatio', marker='o', ax=ax1)
ax1.set_title('Promedio de HomeRatio por Año')
ax1.set_xlabel('Año')
ax1.set_ylabel('HomeRatio Promedio')
ax1.grid(True)
plt.tight_layout()
st.pyplot(fig1)

# Sección 3: Comparación de Home Advantage por País
st.subheader("Comparación de Home Advantage por País")
country_avg = df.groupby('Country', as_index=False)['HomeRatio'].mean().sort_values(by='HomeRatio', ascending=False)
fig_height = max(6, len(country_avg) * 0.3)
fig2, ax2 = plt.subplots(figsize=(10, fig_height))
sns.barplot(data=country_avg, x='HomeRatio', y='Country', palette='viridis', ax=ax2)
ax2.set_title('Promedio de HomeRatio por País')
ax2.set_xlabel('HomeRatio Promedio')
ax2.set_ylabel('País')
plt.tight_layout()
st.pyplot(fig2)
st.dataframe(country_avg.reset_index(drop=True))

if 'FIFA_Rank' in df.columns:
    st.subheader("Relación entre FIFA Rank y Home Advantage (Interactividad)")
    # Crear gráfico interactivo: usamos 'Team', 'League' y 'Year' como información adicional al pasar el mouse.
    fig = px.scatter(
        df,
        x='FIFA_Rank',
        y='HomeRatio',
        color='Country',
        size='HomeRatio',  # Opcional: se ajusta el tamaño de los puntos según HomeRatio
        hover_data=['Team', 'League', 'Year'],
        title='Relación entre FIFA Rank (Normalizado) y HomeRatio',
        labels={
            'FIFA_Rank': 'FIFA Rank (Normalizado)',
            'HomeRatio': 'HomeRatio'
        },
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("El dataset no contiene información de FIFA Rank.")

# Sección 5: Análisis Detallado por País y Liga (ordenado de mayor a menor según HomeRatio)
st.subheader("Análisis Detallado por País y Liga")
selected_country = st.selectbox("Selecciona un país", df['Country'].unique())
selected_league = st.selectbox("Selecciona una liga", df[df['Country'] == selected_country]['League'].unique())
filtered_df = df[(df['Country'] == selected_country) & (df['League'] == selected_league)]
filtered_df = filtered_df.sort_values(by='HomeRatio', ascending=False)
st.write(filtered_df[['Year', 'Team', 'HomeRatio', 'AwayGoalsDiff']])
