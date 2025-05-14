import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf

# 1. Leer el dataset limpio
df = pd.read_csv("Data/Processed_HomeAdvantage_FIFA_Data.csv")

# 2. Crear la variable objetivo binaria (sin usar HomeRatio como input)
df['HomeAdvantage'] = (df['HomeRatio'] >= 0.5).astype(int)
df.drop(columns=['HomeRatio'], inplace=True)

# 3. Codificar League y Country
for col in ['League', 'Country']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# 4. Definir X y y
features = [
    'FIFA_Rank',
    'AwayGoalsDiff',
    'WinRate_Local', 'DrawRate_Local', 'LossRate_Local',
    'WinRate_Visitante', 'LossRate_Visitante', 'GoalDiff_Visitante_Prom',
    'Year', 'League', 'Country'
]
X = df[features].values
y = df['HomeAdvantage'].values

print("Tamaño de entrada (X):", X.shape)
print("Tamaño de salida (y):", y.shape)

# 5. Normalización de entrada
scaler = StandardScaler()
X = scaler.fit_transform(X)

print("Primer vector normalizado:\n", X[0])

# 6. División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

print("X_train:", X_train.shape, "X_test:", X_test.shape)
print("y_train:", y_train.shape, "y_test:", y_test.shape)

# 7. Modelo secuencial binario
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(X.shape[1],)))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 8. Entrenamiento
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=100,
                    batch_size=32,
                    verbose=1)

# 9. Predicciones
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

print("\nPrimeras predicciones (probabilidad):\n", y_pred_prob[:10].flatten())
print("Primeras predicciones (clasificadas):\n", y_pred[:10].flatten())
print("Reales:\n", y_test[:10])

# 10. Métricas
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

# 11. Curva de accuracy
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión del Modelo')
plt.ylabel('Precisión')
plt.xlabel('Épocas')
plt.legend(loc='best')
plt.show()

# 12. Curva de pérdida
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida del Modelo')
plt.ylabel('Pérdida')
plt.xlabel('Épocas')
plt.legend(loc='best')
plt.show()

# 13. Predicción con nuevo vector personalizado (¡ajusta los valores!)
nuevo_equipo = np.array([[0.45, 1.5, 0.6, 0.2, 0.2, 0.4, 0.5, 0.1, 2019, 5, 3]])
nuevo_equipo_scaled = scaler.transform(nuevo_equipo)

nuevo_pred = model.predict(nuevo_equipo_scaled)
resultado = int(nuevo_pred[0][0] > 0.5)

print("\n🔮 Resultado de predicción personalizada:")
print("Probabilidad:", nuevo_pred[0][0])
print("¿Ventaja de local? 👉", "SÍ" if resultado else "NO")
