# Importar las librerías necesarias
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandasgui import show
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report

"""
Storytelling:
Imagina que trabajas como analista de riesgo en una institución financiera. Tu tarea es construir un 
modelo predictivo que evalúe la probabilidad de impago de los clientes basándote en su ingreso mensual, 
nivel de deuda y comportamiento de pago. Este análisis es clave para determinar qué clientes representan 
un mayor riesgo para la institución y ajustar las estrategias de otorgamiento de créditos.

Con este modelo de regresión logística, puedes:
- Identificar patrones entre ingresos, deuda y comportamiento crediticio.
- Predecir si un cliente caerá en mora (impago).
- Evaluar la efectividad del modelo con métricas clave como la precisión, AUC-ROC y la matriz de confusión.
"""

# Crear datos ficticios
n = 100  # Número de muestras

# Generar valores para las variables independientes
np.random.seed(42)  # Para reproducibilidad
ingreso = np.random.uniform(1000, 10000, n)  # Ingreso mensual
deuda = np.random.uniform(500, 20000, n)  # Deuda total
historial_crediticio = np.random.randint(0, 2, n)  # Historial de pagos (0 = mora, 1 = puntual)

# Crear un modelo simple para la probabilidad de impago
probabilidad_impago = 1 / (1 + np.exp(-(0.01 * ingreso - 0.0001 * deuda - 0.5 * historial_crediticio)))

# Modificar la probabilidad de impago para asegurar que haya ejemplos de ambas clases
impago = (probabilidad_impago > 0.5).astype(int)  # 1 si la probabilidad es mayor a 0.5, 0 si no

# Verificación de la distribución de clases
print(f"Distribución de las clases de impago: {np.bincount(impago)}")

# Crear el DataFrame
df = pd.DataFrame({
    'Ingreso (soles)': ingreso,
    'Deuda (soles)': deuda,
    'Historial Crediticio (0=mora, 1=puntual)': historial_crediticio,
    'Probabilidad de Impago': probabilidad_impago,
    'Impago (0=No, 1=Sí)': impago
})

# Definir las variables independientes (X) y dependientes (y)
X = df[['Ingreso (soles)', 'Deuda (soles)', 'Historial Crediticio (0=mora, 1=puntual)']].values
y = df['Impago (0=No, 1=Sí)'].values

# Separar los datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Verificar la distribución en el conjunto de entrenamiento
print(f"Distribución de las clases en el entrenamiento: {np.bincount(y_train)}")

# Definir y ajustar el modelo de regresión logística
model = LogisticRegression()
model.fit(X_train, y_train)

# Hacer la predicción
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Probabilidades de la clase positiva (impago)

# Calcular las métricas de evaluación
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
class_report = classification_report(y_test, y_pred)

print(f"""
      Precisión: {accuracy}
      AUC-ROC: {roc_auc}
      Matriz de confusión:
      {conf_matrix}
      Informe de clasificación:
      {class_report}
      """)

# Visualización de la probabilidad de impago en función de las variables
plt.figure(figsize=(10, 6))

# Scatter plot de los datos de prueba
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_prob, cmap='coolwarm', label="Probabilidad de Impago")
plt.colorbar(label="Probabilidad de Impago")
plt.title("Evaluación de Riesgo de Créditos: Ingreso vs Deuda")
plt.xlabel("Ingreso (soles)")
plt.ylabel("Deuda (soles)")
plt.legend()
plt.show()

# Mostrar el DataFrame en una interfaz gráfica
show(df)
