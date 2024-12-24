# Importar las librerías necesarias
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandasgui import show
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

"""
Storytelling:
Imagina que trabajas como analista en una agencia de marketing y un cliente ha solicitado entender 
cómo las inversiones publicitarias impactan en los ingresos de su negocio. Tu misión es usar datos 
históricos para construir un modelo que explique esta relación y proporcione predicciones confiables.

Para realizar esta tarea, decides crear un análisis basado en un modelo de regresión lineal simple.
Generas un conjunto de datos ficticio que simula campañas de marketing reales, donde se observa la 
cantidad invertida en publicidad y los ingresos obtenidos. Con este análisis, podrás responder preguntas 
como:
- ¿Cuánto generan los ingresos por cada sol invertido en publicidad?
- ¿Qué tan confiable es el modelo para predecir futuros ingresos basados en una inversión específica?

Además, visualizas los datos para presentar al cliente un informe claro y fácil de entender.
"""

# Crear datos ficticios
n = 100  # Número de muestras

# Generar valores para la cantidad de inversión invertida en publicidad
np.random.seed(42)  # Para reproducibilidad
inversionsoles = np.random.uniform(1000, 10000, n)

# Definir una relación lineal con ruido aleatorio para el ingreso obtenido (Y)
# Suponemos que por cada sol invertido en publicidad, obtendremos 3.5 soles en ingresos
# Agregamos un poco de ruido normal para simular la variedad natural
ingresos = 3.5 * inversionsoles + np.random.normal(0, 200, n) + 3000  # Intercepto en 3000 con el eje Y

# Crear el DataFrame
df = pd.DataFrame({
    'Inversion (soles)': inversionsoles,
    'Ingreso (soles)': ingresos
})

# Definir las variables independientes (X) y dependientes (y)
X = df[['Inversion (soles)']].values
y = df['Ingreso (soles)'].values

# Separar los datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Definir y ajustar el modelo de regresión lineal
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Hacer la predicción
y_pred = regressor.predict(X_test)

# Calcular los coeficientes b1 (pendiente) y b0 (intercepto)
b1 = regressor.coef_[0]
b0 = regressor.intercept_
print(f"""
      b1: {b1}
      b0: {b0}
      """)

# Calcular las métricas de evaluación: MSE, RMSE y R^2
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f"""
      mse: {mse}
      rmse: {rmse}
      r2: {r2}
      """)

# Visualización de los resultados
plt.figure(figsize=(10, 6))

# Scatter plot del set de entrenamiento
plt.scatter(X_train, y_train, color="blue", label="Datos de entrenamiento")

# Scatter plot para el set de prueba
plt.scatter(X_test, y_test, color="green", label="Datos de prueba")

# Graficar la línea de regresión
plt.plot(X_train, regressor.predict(X_train), color="red", label="Línea de regresión")

# Agregar título y etiquetas
plt.title("Regresión lineal: Inversión en publicidad vs Ingresos")
plt.xlabel("Inversión (soles)")
plt.ylabel("Ingresos (soles)")

# Agregar leyenda
plt.legend()

# Mostrar el gráfico
plt.show()

# Mostrar el DataFrame en una interfaz gráfica
show(df)
