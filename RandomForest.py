import pandas as pd
import numpy as np
from random import sample
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn import tree

"""Storytelling: Predicción de Ingresos con Random Forest
Imagina que eres un analista de recursos humanos en una empresa que busca entender mejor 
qué factores influyen en el nivel de ingresos de sus empleados. Quieres desarrollar 
un modelo de machine learning que permita predecir si un trabajador tendrá un ingreso alto o bajo 
en función de su edad, experiencia, nivel educativo, certificaciones y horas de trabajo. 
Para ello utilizaremos el modelo de machine learning random forest

Contexto: Generación de Datos Simulados
Como primer paso, se simulan datos para representar una muestra ficticia de empleados. 
Cada empleado tiene características como edad, experiencia, educación, certificaciones y horas trabajadas. 
El objetivo es predecir si su ingreso será alto (1) o bajo (0)."""

# Crearemos el DataFrame para realizar el modelo
np.random.seed(42) 

# Generación de datos simulados
# Se crean 100 registros con valores aleatorios dentro de rangos establecidos.
data = {
    "edad": np.random.randint(20, 60, 100),
    "experiencia": np.random.randint(1, 30, 100),
    "educacion": np.random.randint(0, 2, 100),  # 0 = No tiene educación superior, 1 = Sí tiene
    "certificaciones": np.random.randint(0, 2, 100),
    "horas_trabajo": np.random.randint(20, 50, 100),
    "ingreso": np.random.randint(0, 2, 100)  # 0 = Bajo, 1 = Alto
}

personas = pd.DataFrame(data)

# Visualización inicial del DataFrame
print("Vista previa de los datos simulados:")
print(personas.head())

# Muestras Bootstrap (con reemplazo)
# Se generan tres muestras aleatorias del conjunto de datos para reflejar diferentes subconjuntos
print("\nMuestras Bootstrap:")
print(personas.sample(frac=2/3, replace=True))
print(personas.sample(frac=2/3, replace=True))
print(personas.sample(frac=2/3, replace=True))

# Selección de características aleatorias
# Se seleccionan aleatoriamente 3 características (sin incluir la columna de ingresos)
print("\nColumnas disponibles para entrenamiento (sin incluir el objetivo):")
print(personas.columns[:-1], "\n")
print("Características seleccionadas aleatoriamente para el modelo:")
print(sample(list(personas.columns[:-1]), 3))  # Corrección: convertir set a lista

# Entrenamiento del modelo Random Forest
# Se crea y entrena un modelo Random Forest con 100 árboles (estimadores) y usando 'gini' como criterio de división
bosque = RandomForestClassifier(n_estimators=100,
                               criterion="gini",
                               max_features="sqrt",
                               bootstrap=True,
                               max_samples=2/3,
                               oob_score=True)  # Calcular el OOB Score (validación interna)

bosque.fit(personas[personas.columns[:-1]].values, personas["ingreso"].values)

# Predicción y evaluación
# Se realiza una predicción para un empleado ficticio con características específicas
prediccion = bosque.predict([[50, 16, 1, 1, 40]])  
print(f"\nPredicción para el nuevo empleado: {prediccion}")

# Evaluación del modelo
precision = bosque.score(personas[personas.columns[:-1]].values, personas["ingreso"].values)
print(f"Precisión en el conjunto de entrenamiento: {precision}")
print(f"Puntaje OOB (Out-of-Bag): {bosque.oob_score_}") 

# Visualización de algunos árboles individuales del bosque
# Se visualizan 3 de los 100 árboles generados por el modelo
for arbol in bosque.estimators_[:3]:  
    plt.figure(figsize=(10, 5))
    tree.plot_tree(arbol, feature_names=personas.columns[:-1], filled=True)
    plt.show()
