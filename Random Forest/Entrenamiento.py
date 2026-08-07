import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 1. Configurar ruta de trabajo local
script_dir = r'C:\Users\Esteb\OneDrive - unimilitar.edu.co\Universidad\9o Semestre\Inteligencia Artificial\Laboratorio\Guía #1\Códigos'

if os.path.exists(script_dir):
    os.chdir(script_dir)
    print(f'Directorio activo:\n{os.getcwd()}\n')

# 2. Cargar el dataset original
df = pd.read_csv('desempeno_estudiantes.csv')

# 3. Separar X e y, y aplicar One-Hot Encoding a 'Carrera'
X = df.drop(columns=['Desempeno_Academico'])
if 'Codigo_Carrera' in X.columns:
    X = X.drop(columns=['Codigo_Carrera'])  # Nos aseguramos de usar solo 'Carrera'

X_encoded = pd.get_dummies(X, columns=['Carrera'], drop_first=False, dtype=int)
y = df['Desempeno_Academico']

# 4. Partición del dataset (80% Entrenamiento, 20% Prueba)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.20, random_state=42, stratify=y
)

# 5. Evaluación progresiva con diferente número de árboles
trees_list = [1, 5, 10, 20, 50, 100]
train_accuracies = []
test_accuracies = []

print('=== EVALUACIÓN PROGRESIVA DE RANDOM FOREST ===\n')
print(
    f'{"N° Árboles":^10} | {"Exactitud Train":^17} | {"Exactitud Test":^15} | {"Diferencia":^12}'
)
print('-' * 62)

for n_trees in trees_list:
    # Crear y entrenar el modelo
    rf = RandomForestClassifier(n_estimators=n_trees, max_features='sqrt', max_depth=10, min_samples_split=7, min_samples_leaf=3, random_state=42)   
    rf.fit(X_train, y_train)

    # Predecir en entrenamiento y prueba
    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)

    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)

    train_accuracies.append(acc_train)
    test_accuracies.append(acc_test)

    diff = acc_train - acc_test
    print(
        f'{n_trees:^10d} | {acc_train:^17.4f} | {acc_test:^15.4f} | {diff:^12.4f}'
    )

# 6. Graficar el comportamiento de la curva de aprendizaje
plt.figure(figsize=(9, 5))
plt.plot(
    trees_list,
    [acc * 100 for acc in train_accuracies],
    'o--',
    linewidth=2,
    label='Entrenamiento (Train)',
)
plt.plot(
    trees_list,
    [acc * 100 for acc in test_accuracies],
    's-',
    linewidth=2,
    label='Prueba (Test)',
)

plt.title('Efecto de la Cantidad de Árboles en la Exactitud del Modelo', fontsize=12)
plt.xlabel('Número de Árboles (n_estimators)', fontsize=10)
plt.ylabel('Exactitud (%)', fontsize=10)
plt.xticks(trees_list)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)
plt.tight_layout()

# Guardar y mostrar gráfica en la carpeta de la guía
plt.savefig('progreso_arboles_random_forest.png', dpi=300)
plt.show()