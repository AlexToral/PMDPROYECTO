import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Cargar datasets
clean_data_path = "./Data/Processed_HomeAdvantage_FIFA_Data.csv"
raw_soccer_path = "./Data/SoccerLEagues.csv"
raw_country_path = "./Data/Country_Facts.csv"

# Cargar datos
clean_df = pd.read_csv(clean_data_path)
raw_df1 = pd.read_csv(raw_soccer_path)
raw_df2 = pd.read_csv(raw_country_path)

st.title("Impacto de la Ventaja de Jugar en Casa: Antes vs. Después de la Limpieza")

# Crear pestañas para comparar antes y después
tab1, tab2 = st.tabs(["Datos Antes de la Limpieza", "Datos Después de la Limpieza"])

# Pestaña 1: Datos antes de la limpieza
with tab1:
    st.subheader("Vista de Datos Crudos")
    st.dataframe(raw_df1.head())
    st.dataframe(raw_df2.head())
    
    st.subheader("Comparación de Home Advantage antes de la limpieza")
    if 'HomeRatio' in raw_df1.columns:
        trend_raw = raw_df1.groupby('Year', as_index=False)['HomeRatio'].mean()
        fig_raw, ax_raw = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=trend_raw, x='Year', y='HomeRatio', marker='o', ax=ax_raw)
        ax_raw.set_title('Promedio de HomeRatio por Año (Datos Crudos)')
        ax_raw.set_xlabel('Año')
        ax_raw.set_ylabel('HomeRatio Promedio')
        ax_raw.grid(True)
        plt.tight_layout()
        st.pyplot(fig_raw)
    else:
        st.warning("La columna 'HomeRatio' no está presente en los datos crudos.")
    
    st.subheader("Distribución de Home Advantage por País (Antes de la Limpieza)")
    if 'Country' in raw_df1.columns and 'HomeRatio' in raw_df1.columns:
        country_avg_raw = raw_df1.groupby('Country', as_index=False)['HomeRatio'].mean().sort_values(by='HomeRatio', ascending=False)
        fig_raw2, ax_raw2 = plt.subplots(figsize=(10, max(6, len(country_avg_raw) * 0.3)))
        sns.barplot(data=country_avg_raw, x='HomeRatio', y='Country', palette='coolwarm', ax=ax_raw2)
        ax_raw2.set_title('HomeRatio por País (Antes de la Limpieza)')
        ax_raw2.set_xlabel('HomeRatio Promedio')
        ax_raw2.set_ylabel('País')
        plt.tight_layout()
        st.pyplot(fig_raw2)
        st.dataframe(country_avg_raw.reset_index(drop=True))
    else:
        st.warning("No se pueden generar gráficos debido a datos faltantes en los datos crudos.")

# Pestaña 2: Datos después de la limpieza
with tab2:
    st.subheader("Vista de Datos Limpios")
    st.dataframe(clean_df.head())
    
    st.subheader("Tendencia del Home Advantage después de la limpieza")
    trend_clean = clean_df.groupby('Year', as_index=False)['HomeRatio'].mean()
    fig_clean, ax_clean = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=trend_clean, x='Year', y='HomeRatio', marker='o', ax=ax_clean)
    ax_clean.set_title('Promedio de HomeRatio por Año (Datos Limpios)')
    ax_clean.set_xlabel('Año')
    ax_clean.set_ylabel('HomeRatio Promedio')
    ax_clean.grid(True)
    plt.tight_layout()
    st.pyplot(fig_clean)
    
    st.subheader("Distribución de Home Advantage por País (Después de la Limpieza)")
    country_avg_clean = clean_df.groupby('Country', as_index=False)['HomeRatio'].mean().sort_values(by='HomeRatio', ascending=False)
    fig_clean2, ax_clean2 = plt.subplots(figsize=(10, max(6, len(country_avg_clean) * 0.3)))
    sns.barplot(data=country_avg_clean, x='HomeRatio', y='Country', palette='viridis', ax=ax_clean2)
    ax_clean2.set_title('HomeRatio por País (Después de la Limpieza)')
    ax_clean2.set_xlabel('HomeRatio Promedio')
    ax_clean2.set_ylabel('País')
    plt.tight_layout()
    st.pyplot(fig_clean2)
    st.dataframe(country_avg_clean.reset_index(drop=True))

    # Análisis de FIFA Rank solo si está disponible en los datos limpios
    if 'FIFA_Rank' in clean_df.columns:
        st.subheader("Relación entre FIFA Rank y Home Advantage")
        fig = px.scatter(
            clean_df,
            x='FIFA_Rank',
            y='HomeRatio',
            color='Country',
            size='HomeRatio',
            hover_data=['Team', 'League', 'Year'],
            title='FIFA Rank vs HomeRatio (Datos Limpios)',
            labels={'FIFA_Rank': 'FIFA Rank (Normalizado)', 'HomeRatio': 'HomeRatio'},
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("El dataset limpio no contiene información de FIFA Rank.")

# Sección final: Comparación detallada
st.subheader("Comparación detallada de un país y una liga")
selected_country = st.selectbox("Selecciona un país", clean_df['Country'].unique())
selected_league = st.selectbox("Selecciona una liga", clean_df[clean_df['Country'] == selected_country]['League'].unique())

filtered_clean_df = clean_df[(clean_df['Country'] == selected_country) & (clean_df['League'] == selected_league)].sort_values(by='HomeRatio', ascending=False)
st.write("**Datos después de la limpieza:**")
st.dataframe(filtered_clean_df[['Year', 'Team', 'HomeRatio', 'AwayGoalsDiff']])
