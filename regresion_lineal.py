import numpy as np # cálculo numérico: funciones para trabajar con vectores, matrices, etc.

import matplotlib.pyplot as plt # visualización gráfica de datos

from sklearn.datasets import fetch_california_housing # Dataset 1
# Cargamos librería
housing = fetch_california_housing()

"""from sklearn.datasets import fetch_openml # Dataset 2
housing = fetch_openml(name="house_prices", as_frame=True)"""

#print(housing.keys())
#print(housing.get('DESCR'))

# A partir de los datos de habitaciones promedio y valor promedio de las casas (target)

X = np.array(housing.get('data')[:,2]) # 'AveRooms': average number of rooms per household
#X = np.array(housing.get('data')[:,3]) # 'AveBedrms': average number of bedrooms per household
Y = np.array(housing.get('target')) # target: median house value ($100,000)

plt.scatter(X, Y, alpha=0.2)

# añadir columna de 1, para poder obtener el término independiente
total_filas = 20640 # total de filas (instancias) en el conjunto de datos
X = np.array([np.ones(total_filas), X]).T

# Error Cuadrático Medio (vectorial)
# B = (X^t x X)^-1 x X^t x Y
# X -> matriz X
# Y -> matriz Y
# x -> producto vectorial

B = np.linalg.inv(X.T @ X) @ X.T @ Y

plt.plot([0,5], [B[0] + B[1] * 0, B[0] + B[1] * 5], c='red')

plt.show()

