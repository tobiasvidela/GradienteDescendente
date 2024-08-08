import numpy as np  # Importa la biblioteca para realizar cálculos numéricos, como operaciones con vectores y matrices.

import matplotlib.pyplot as plt  # Importa la biblioteca para la visualización gráfica de datos.

from sklearn.datasets import fetch_california_housing  # Importa el conjunto de datos de viviendas de California.

# Cargamos el conjunto de datos de viviendas de California.
housing = fetch_california_housing()

# Imprime la descripción del conjunto de datos para obtener información sobre las características y el objetivo.
#print(housing.keys())  # Descomentar para ver las claves del diccionario que contiene los datos.
print(housing.get('DESCR'))  # Descomentar para imprimir la descripción del dataset.

# A partir de los datos de ingresos medianos y valor medio de las casas (target).
# 'MedInc': ingreso medio en el grupo de bloques.
X = np.array(housing.get('data')[:, 0])  # Asigna la columna 'MedInc' a X.

# Otras opciones de características para X (descomentar si es necesario):
#X = np.array(housing.get('data')[:, 2]) # 'AveRooms': número promedio de habitaciones por hogar.
#X = np.array(housing.get('data')[:, 3]) # 'AveBedrms': número promedio de dormitorios por hogar.
#X = np.array(housing.get('data')[:, 4]) # 'Population': población del grupo de bloques.

# Asigna el valor medio de las casas (target) a Y.
Y = np.array(housing.get('target'))  # target: valor medio de la vivienda (en $100,000).

# Datos de ejemplo alternativos (descomentar si se desea usar estos valores en lugar del dataset):
#X = np.array([2.86232, 3.8232, 5.862232, 1.2286, 7.3286, 3.124, 2.795, 4.538491])
#Y = np.array([1.53442, 2.8232, 4.26432, 1.1226, 4.2199, 3.3516, 3.2391, 3.8467])

# Visualiza los datos como un gráfico de dispersión.
plt.scatter(X, Y, alpha=0.2)  # Crea un gráfico de dispersión con un nivel de transparencia (alpha) del 20%.

# Añadir una columna de 1 para incluir el término independiente en la matriz X.
total_filas = 20640  # Define el total de filas (instancias) en el conjunto de datos.
#total_filas = 8  # Alternativa: define el total de filas para datos de ejemplo (descomentar si es necesario).
X = np.array([np.ones(total_filas), X]).T  # Transforma X en una matriz de diseño, incluyendo la columna de 1s.

# Cálculo del Error Cuadrático Medio (vectorial):
# B = (X^t * X)^-1 * X^t * Y
# Donde:
#     X -> matriz X (diseño)
#     Y -> vector Y (valores objetivo)
#     * -> producto matricial

B = np.linalg.inv(X.T @ X) @ X.T @ Y  # Calcula el vector de coeficientes B utilizando la fórmula de mínimos cuadrados.

print(B)  # Imprime los coeficientes del modelo lineal.

# Dibuja la línea de regresión obtenida.
plt.plot([0, 12], [B[0] + B[1] * 0, B[0] + B[1] * 12], c='red')  # Grafica la línea de regresión en rojo.

# Muestra el gráfico de dispersión y la línea de regresión.
plt.show()  # Muestra el gráfico generado.
