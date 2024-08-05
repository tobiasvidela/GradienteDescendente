import numpy as np # cálculo numérico: funciones para trabajar con vectores, matrices, etc.

import matplotlib.pyplot as plt # visualización gráfica de datos

from sklearn.datasets import fetch_california_housing # Dataset 1
# Cargamos librería
housing = fetch_california_housing()

#print(housing.keys())
print(housing.get('DESCR'))

# A partir de los datos de habitaciones promedio y valor promedio de las casas (target)

X = np.array(housing.get('data')[:,0]) # 'MedInc': median income in block group
#X = np.array(housing.get('data')[:,2]) # 'AveRooms': average number of rooms per household
#X = np.array(housing.get('data')[:,3]) # 'AveBedrms': average number of bedrooms per household
#X = np.array(housing.get('data')[:,4]) # 'Population': block group population
Y = np.array(housing.get('target')) # target: median house value ($100,000)

#X = np.array([2.86232, 3.8232, 5.862232, 1.2286, 7.3286, 3.124, 2.795, 4.538491])
#Y = np.array([4.53442, 5.8232, 5.26432, 4.1226, 5.2199, 4.3516, 3.2391, 4.6467])

plt.scatter(X, Y)

# añadir columna de 1, para término independiente
total_filas = 20640 # total de filas (instancias) en el conjunto de datos
X = np.array([np.ones(total_filas), X]).T

# Error Cuadrático Medio (vectorial):
# B = (X^t x X)^-1 x X^t x Y
# Donde
#     X -> matriz X
#     Y -> matriz Y
#     x -> producto vectorial

B = np.linalg.inv(X.T @ X) @ X.T @ Y

print(B)

plt.plot([0,12], [B[0] + B[1] * 0, B[0] + B[1] * 12], c='red')

plt.show()
