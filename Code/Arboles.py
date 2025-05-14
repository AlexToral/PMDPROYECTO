import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from subprocess import check_call

# 1. Cargar datos
df = pd.read_csv("Data/Processed_HomeAdvantage_FIFA_Data.csv")

# 2. Variable objetivo
df['HomeAdvantage'] = (df['HomeRatio'] >= 0.5).astype(int)
df.drop(columns=['HomeRatio'], inplace=True)

# 3. Codificación de variables categóricas
le_country = LabelEncoder()
le_league = LabelEncoder()
df['Country'] = le_country.fit_transform(df['Country'])
df['League'] = le_league.fit_transform(df['League'])

# 4. Features y target
X = df.drop(columns=['Team', 'HomeAdvantage'])
y = df['HomeAdvantage']

# 5. Árbol más compacto
clf = tree.DecisionTreeClassifier(
    criterion='entropy',
    max_depth=6,                # 🌿 Reducir profundidad
    min_samples_split=30,      # 🌿 Reglas más generales
    min_samples_leaf=10,
    class_weight={0: 1, 1: 3.6},
    random_state=42
)
clf.fit(X, y)

# 6. Precisión del modelo
accuracy = round(clf.score(X, y) * 100, 2)
print(f"La precisión del Árbol compacto es: {accuracy}%")

# 7. Exportar árbol
os.makedirs('./Guardados', exist_ok=True)
with open("./Guardados/Arbol_Compacto.dot", 'w') as f:
    tree.export_graphviz(
        clf,
        out_file=f,
        max_depth=6,
        feature_names=X.columns,
        class_names=['No Advantage', 'Advantage'],
        rounded=True,
        filled=True
    )

check_call(['dot', '-Tpng', './Guardados/Arbol_Compacto.dot', '-o', './Guardados/Arbol_Compacto.png'])
print("✅ Árbol compacto exportado a ./Guardados/Arbol_Compacto.png")

# 8. Predicción personalizada
features = list(X.columns)
valores_equipo = [0.45, 1.5, 0.6, 0.2, 0.2, 0.4, 0.5, 0.1, 2019, 5, 3]
equipo_hypo = pd.DataFrame([valores_equipo], columns=features)

pred = clf.predict(equipo_hypo)[0]
proba = clf.predict_proba(equipo_hypo)[0][pred]

print("\n🔮 Predicción personalizada:")
print("¿Ventaja de local?", "SÍ ✅" if pred == 1 else "NO ❌")
print(f"Probabilidad: {round(proba*100, 2)}%")

# 9. Visualización matriz de confusión
y_pred = clf.predict(X)
sns.heatmap(confusion_matrix(y, y_pred), annot=True, fmt='d', cmap='BuGn')
plt.title("Matriz de Confusión - Árbol Compacto")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.show()
