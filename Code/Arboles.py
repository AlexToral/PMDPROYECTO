import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar el dataset limpio
df = pd.read_csv("Data/Processed_HomeAdvantage_FIFA_Data.csv")

# 2. Crear la variable objetivo sin hacer trampa
df['HomeAdvantage'] = (df['HomeRatio'] >= 0.5).astype(int)
df = df.drop(columns=['HomeRatio'])  # 🔥 ¡IMPORTANTE! Eliminamos la variable que da la respuesta

# 3. Codificar variables categóricas
for col in ['League', 'Country']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# 4. Selección de features potentes
features = [
    'FIFA_Rank',
    'AwayGoalsDiff',
    'WinRate_Local',
    'DrawRate_Local',
    'LossRate_Local',
    'WinRate_Visitante',
    'LossRate_Visitante',
    'GoalDiff_Visitante_Prom',
    'Year',
    'League',
    'Country'
]

X = df[features]
y = df['HomeAdvantage']

# 5. División entrenamiento/prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Entrenar el árbol
clf = DecisionTreeClassifier(
    random_state=42,
    class_weight='balanced',
    max_depth=6,
    min_samples_split=20,
    min_samples_leaf=10
)
clf.fit(X_train, y_train)

# 7. Evaluación
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

# 8. Visualización
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='YlGnBu')
plt.title("Matriz de Confusión - Árbol con Features Futboleras")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.show()
