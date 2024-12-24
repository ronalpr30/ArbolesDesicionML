import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn import tree

"""Storytelling: Predicción de Problemas Cardiacos con Árbol de Decisión
Imagina que eres un investigador en una clínica especializada en salud cardiovascular.
Quieres desarrollar un modelo que ayude a predecir si un paciente tiene o no un problema cardíaco.
Usaremos datos simulados de pacientes que incluyen edad y niveles de colesterol para entrenar un modelo
de árbol de decisión, que permitirá evaluar el riesgo de enfermedades cardíacas.

Contexto: Generación de Datos Simulados
Los datos reflejan una población ficticia de pacientes con edades y niveles de colesterol variados.
El objetivo es utilizar estas variables para clasificar si un paciente tiene o no un problema cardíaco."""

# Generación de datos para su posterior prueba (Rangos de Edad y Colesterol)
np.random.seed(42)

edad = np.random.randint(30, 80, 100)
colesterol = np.random.randint(150, 300, 100)
problema_cardiaco = np.random.choice([0, 1], 100, p=[0.7, 0.3])  # 70% saludables, 30% con problemas

# Crear DataFrame Simulado
pacientes = pd.DataFrame({
    "edad": edad,
    "colesterol": colesterol,
    "problema_cardiaco": problema_cardiaco
})

# Visualización de Datos (Distribución de Pacientes Saludables y Cardíacos)
saludables = pacientes[pacientes["problema_cardiaco"] == 0]
cardiacos = pacientes[pacientes["problema_cardiaco"] == 1]

plt.figure(figsize=(6, 6))
plt.xlabel('Edad', fontsize=20.0)
plt.ylabel('Colesterol', fontsize=20.0)
plt.scatter(saludables["edad"], saludables["colesterol"], 
            label="Saludable (Clase: 0)", marker="*", c="skyblue", s=200)
plt.scatter(cardiacos["edad"], cardiacos["colesterol"],
            label="Cardíaco (Clase: 1)", marker="*", c="lightcoral", s=200)
plt.legend(bbox_to_anchor=(1, 0.15))
plt.show()

# Cálculo de Entropía (Distribución de Edad y Colesterol)
edades = pd.Series(edad)
colesterol_series = pd.Series(colesterol)

print("Distribución de Edades:")
print(edades.value_counts() / edades.size)
print("Distribución de Colesterol:")
print(colesterol_series.value_counts() / colesterol_series.size)

print("Entropía de Edad:", entropy(edades.value_counts() / edades.size, base=2))
print("Entropía de Colesterol:", entropy(colesterol_series.value_counts() / colesterol_series.size, base=2))

# Entrenamiento de Árbol de Decisión
# Dividir datos en entrenamiento y prueba
datos_entrena, datos_prueba, clase_entrena, clase_prueba = train_test_split(
    pacientes[["edad", "colesterol"]],
    pacientes["problema_cardiaco"],
    test_size=0.30,
    random_state=42
)

# Inicializar y entrenar el modelo de Árbol de Decisión
arbol_decision = tree.DecisionTreeClassifier(criterion="entropy")
arbol = arbol_decision.fit(datos_entrena, clase_entrena)

# Calcular la precisión del modelo
accuracy = arbol_decision.score(datos_prueba, clase_prueba)
print("Precisión del modelo:", accuracy)

# Visualización del Árbol de Decisión
print(tree.export_text(arbol, feature_names=["Edad", "Colesterol"]))

plt.figure(figsize=(12, 6))
tree.plot_tree(arbol, feature_names=["Edad", "Colesterol"], filled=True)
plt.show()