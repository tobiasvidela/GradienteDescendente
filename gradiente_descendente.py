import matplotlib.pyplot as plt
import numpy as np

# Establecer puntos de datos a partir de un ejemplo
x = [2,5,3]
y = [2,4,5]

# Trazar puntos de datos y límites x e y
plt.scatter(x, y, c='#4ad66d')
plt.xlim((0, 6))
plt.ylim((0, 6))

# Establecer valores iniciales del bias, eta e iteraciones
bias = 0 # valor inicial de la intersección
eta = 0.1
n_iterations = 10

# Computar gradiente descendente
for iteration in range(n_iterations):
  # calcular gradientes
  gradientes = (1/3) * ((-2 * (2 - (bias + (0.4 * 2)))) + (-2 * (5 - (bias + (0.4 * 3)))) + (-2 * (4 - (bias + (0.4 * 5)))))
  bias = bias - (eta * gradientes) # actualizar el valor de la intersección

  # trazar cambio de bias
  y_predict = []
  for n in x:
    y_predict.append(bias + (0.4 * n))
  plt.plot(x, y_predict, color='#00a8e8')

  # calcular ECM en cada iteración
  errores = []
  for i in range(len(x)):
    errores.append((y[i] - (bias + (0.4 * (x[i])))) ** 2)
  # error cuadrático medio (ECM)
  ecm = (1/3) * np.sum(errores)
  print(f'ECM: {ecm} and Bias: {bias}')

plt.show()
