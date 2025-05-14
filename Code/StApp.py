
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
import numpy as np

# Cargar datasets
clean_data_path = "./Data/Processed_HomeAdvantage_FIFA_Data.csv"
raw_soccer_path = "./Data/SoccerLEagues.csv"
raw_country_path = "Data/Country_Facts.csv"

# Cargar datos
clean_df = pd.read_csv(clean_data_path)
raw_df1 = pd.read_csv(raw_soccer_path)
raw_df2 = pd.read_csv(raw_country_path)

st.title("Impacto de la Ventaja de Jugar en Casa: Antes vs. Después de la Limpieza")

# Crear pestañas para comparar antes y después
tab1, tab2, tab3 = st.tabs(["Datos Antes de la Limpieza", "Datos Después de la Limpieza", "Modelos de Predicción"])

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

    st.subheader("Comparación detallada de un país y una liga")
    selected_country = st.selectbox("Selecciona un país", clean_df['Country'].unique())
    selected_league = st.selectbox("Selecciona una liga", clean_df[clean_df['Country'] == selected_country]['League'].unique())
    filtered_clean_df = clean_df[(clean_df['Country'] == selected_country) & (clean_df['League'] == selected_league)].sort_values(by='HomeRatio', ascending=False)
    st.write("**Datos después de la limpieza:**")
    st.dataframe(filtered_clean_df[['Year', 'Team', 'HomeRatio', 'AwayGoalsDiff']])

# Pestaña 3: Modelos de Predicción
with tab3:

    st.subheader("Preparación de Datos")
    df = clean_df.copy()
    df['HomeAdvantage'] = (df['HomeRatio'] >= 0.5).astype(int)
    df.drop(columns=['HomeRatio'], inplace=True)
    for col in ['League', 'Country']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    features = ['FIFA_Rank', 'AwayGoalsDiff',
                'WinRate_Local', 'DrawRate_Local', 'LossRate_Local',
                'WinRate_Visitante', 'LossRate_Visitante', 'GoalDiff_Visitante_Prom',
                'Year', 'League', 'Country']
    
    st.subheader("Análisis de Correlación entre Variables")

    st.markdown("""
    Utilizando la matriz de correlación y un gráfico de calor (heatmap), 
    podemos observar qué tan relacionadas están las variables predictoras entre sí. 
    Una correlación alta (positiva o negativa) puede indicar redundancia o dependencia útil 
    para los modelos. Idealmente, buscamos variables que tengan una buena correlación con la variable objetivo (HomeAdvantage) 
    pero baja entre ellas para evitar multicolinealidad.
    """)

    corr_matrix = df[features + ['HomeAdvantage']].corr()

    fig_corr_total, ax_corr_total = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax_corr_total)
    ax_corr_total.set_title("Matriz de Correlación (incluyendo HomeAdvantage)")
    st.pyplot(fig_corr_total)

    st.markdown("### Correlación con la Variable Objetivo (`HomeAdvantage`)")
    corr_target = corr_matrix['HomeAdvantage'].drop('HomeAdvantage').sort_values(ascending=False)
    st.bar_chart(corr_target)

    st.header("Modelos de Predicción: Árboles de Decisión y Redes Neuronales")

    st.subheader("Preparación de Datos")
    df = clean_df.copy()
    df['HomeAdvantage'] = (df['HomeRatio'] >= 0.5).astype(int)
    df.drop(columns=['HomeRatio'], inplace=True)
    for col in ['League', 'Country']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    features = ['FIFA_Rank', 'AwayGoalsDiff',
                'WinRate_Local', 'DrawRate_Local', 'LossRate_Local',
                'WinRate_Visitante', 'LossRate_Visitante', 'GoalDiff_Visitante_Prom',
                'Year', 'League', 'Country']
    X = df[features].values
    y = df['HomeAdvantage'].values

    # Árbol
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=6, min_samples_split=30, min_samples_leaf=10, class_weight={0:1,1:3.6}, random_state=42)
    clf.fit(X, y)
    y_pred_arbol = clf.predict(X)
    acc_arbol = accuracy_score(y, y_pred_arbol)
    st.subheader("Árbol de Decisión")
    st.write("**Precisión del árbol en entrenamiento:**", round(acc_arbol*100,2), "%")
    fig_cm_arbol, ax_cm_arbol = plt.subplots()
    sns.heatmap(confusion_matrix(y, y_pred_arbol), annot=True, fmt='d', cmap='Greens', ax=ax_cm_arbol)
    ax_cm_arbol.set_title("Matriz de Confusión - Árbol Compacto")
    st.pyplot(fig_cm_arbol)

    # Red neuronal
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=0)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(X.shape[1],)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, verbose=0)
    y_pred_nn = (model.predict(X_test) > 0.5).astype(int)
    acc_nn = accuracy_score(y_test, y_pred_nn)
    st.subheader("Red Neuronal")
    st.write("**Precisión del modelo en prueba:**", round(acc_nn*100,2), "%")
    fig_cm_nn, ax_cm_nn = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred_nn), annot=True, fmt='d', cmap='Blues', ax=ax_cm_nn)
    ax_cm_nn.set_title("Matriz de Confusión - Red Neuronal")
    st.pyplot(fig_cm_nn)

    st.subheader("Curvas de entrenamiento de la red neuronal")
    fig_loss, ax_loss = plt.subplots()
    ax_loss.plot(history.history['loss'], label='Entrenamiento')
    ax_loss.plot(history.history['val_loss'], label='Validación')
    ax_loss.set_title('Curva de Pérdida')
    ax_loss.set_xlabel('Épocas')
    ax_loss.set_ylabel('Pérdida')
    ax_loss.legend()
    st.pyplot(fig_loss)

    fig_acc, ax_acc = plt.subplots()
    ax_acc.plot(history.history['accuracy'], label='Entrenamiento')
    ax_acc.plot(history.history['val_accuracy'], label='Validación')
    ax_acc.set_title('Curva de Precisión')
    ax_acc.set_xlabel('Épocas')
    ax_acc.set_ylabel('Precisión')
    ax_acc.legend()
    st.pyplot(fig_acc)

    st.subheader("Comparación de Modelos: Árbol vs Red Neuronal")
    st.markdown("""
    A continuación se presenta una comparación entre el árbol de decisión y la red neuronal en cuanto a precisión en los datos de prueba, junto con sus ventajas:
    - **Árboles de Decisión** son interpretables y permiten entender reglas claras.
    - **Redes Neuronales** capturan patrones complejos no lineales, mejorando la predicción en datos más ruidosos.
    """)
    st.markdown(f"- Precisión del Árbol: **{round(acc_arbol*100, 2)}%** (entrenamiento)")
    st.markdown(f"- Precisión de la Red Neuronal: **{round(acc_nn*100, 2)}%** (prueba)")

    st.subheader("Predicción personalizada con ambos modelos")
    st.markdown("Introduce un vector de características para predecir si un equipo tiene ventaja como local:")

    fifa_rank = st.slider("FIFA Rank", 0.0, 1.0, 0.45)
    away_goals_diff = st.slider("Diferencia de goles como visitante", -5.0, 5.0, 1.5)
    win_local = st.slider("Win Rate Local", 0.0, 1.0, 0.6)
    draw_local = st.slider("Draw Rate Local", 0.0, 1.0, 0.2)
    loss_local = st.slider("Loss Rate Local", 0.0, 1.0, 0.2)
    win_visit = st.slider("Win Rate Visitante", 0.0, 1.0, 0.4)
    loss_visit = st.slider("Loss Rate Visitante", 0.0, 1.0, 0.5)
    goal_diff_visit = st.slider("GoalDiff Visitante Prom", -5.0, 5.0, 0.1)
    year = st.slider("Año", 2000, 2025, 2019)
    league = st.slider("ID Liga (codificado)", 0, 50, 5)
    country = st.slider("ID País (codificado)", 0, 50, 3)

    entrada = np.array([[fifa_rank, away_goals_diff, win_local, draw_local, loss_local, win_visit, loss_visit, goal_diff_visit, year, league, country]])

    st.markdown("### Resultado del Árbol de Decisión")
    pred_arbol = clf.predict(entrada)[0]
    proba_arbol = clf.predict_proba(entrada)[0][pred_arbol]
    st.write("¿Ventaja de local?:", "✅ SÍ" if pred_arbol == 1 else "❌ NO")
    st.write("Probabilidad:", round(proba_arbol*100, 2), "%")

    st.markdown("### Resultado de la Red Neuronal")
    entrada_nn = scaler.transform(entrada)
    pred_nn_prob = model.predict(entrada_nn)[0][0]
    pred_nn = int(pred_nn_prob > 0.5)
    st.write("¿Ventaja de local?:", "✅ SÍ" if pred_nn == 1 else "❌ NO")
    st.write("Probabilidad:", round(pred_nn_prob*100, 2), "%")

    st.info("💡 Tip: Puedes experimentar con los sliders para observar cómo afectan las predicciones de ambos modelos.")

