import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar datos
df = pd.read_csv("Data/Processed_HomeAdvantage_FIFA_Data.csv")
df['HomeAdvantage'] = (df['HomeRatio'] >= 0.5).astype(int)
df = df.drop(columns=['HomeRatio'])

# 2. Codificar categóricas
for col in ['League', 'Country']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# 3. Seleccionar features
features = [
    'FIFA_Rank',
    'AwayGoalsDiff',
    'WinRate_Local', 'DrawRate_Local', 'LossRate_Local',
    'WinRate_Visitante', 'LossRate_Visitante',
    'GoalDiff_Visitante_Prom',
    'Year',
    'League',
    'Country'
]

X = df[features]
y = df['HomeAdvantage']

# 4. Escalamiento
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. División
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 6. Red neuronal
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, input_dim=X.shape[1], activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# 7. Entrenamiento
history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, verbose=0)

# 8. Evaluación
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

# 9. Visualización
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='PuBuGn')
plt.title("Matriz de Confusión - Red Neuronal")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.show()

# 10. Curvas de entrenamiento
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title("Curva de Accuracy - Red Neuronal")
plt.xlabel("Época")
plt.ylabel("Precisión")
plt.tight_layout()
plt.show()
