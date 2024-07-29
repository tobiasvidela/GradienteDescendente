# Librerías para trabajar IAs
import numpy as np # cálculo numérico, álgebra
import scipy as sc # extensión de numpy
import matplotlib.pyplot as plt # visualización de datos

# Función a OPTIMIZAR (La función debe ser derivable, naturalmente)
func = lambda th: np.sin(1/2 * th[0] ** 2 - 1/4 * th[1] ** 2 + 3) * np.cos(2 * th[0] + 1 - np.e ** th[1]) # th[0] = x, th[1] = 1

resolucion = 100

_X = np.linspace(-2, 2, resolucion) # vector de 100 valores
_Y = np.linspace(-2, 2, resolucion) # vector de 100 valores

_Z = np.zeros((resolucion, resolucion)) # matriz de 100*100 con valores nulos (vacía)

# llenar la matriz _Z
for index_of_x, x_value in enumerate(_X): #enumerate() devuelve el índice del valor y el valor del vector parámetro
  for index_if_y, y_value in enumerate(_Y):
    _Z[index_if_y, index_of_x] = func([x_value, y_value])

# visualizar datos
#plt.contour(_X, _Y, _Z, 100) # vista 'de líneas'
plt.contourf(_X, _Y, _Z, 100) # vista sólida
plt.colorbar()

# Theta
Theta = np.random.rand(2) * 4 - 2 # Genera dos números [x, y] en el rango -2 a 2

plt.plot(Theta[0], Theta[1], "o", c="black") # generar un punto grueso y negro aleatorio

# usar derivadas parciales para obtener la pendiente en el punto dado
# y saber hacia dónde "descender" hasta el mínimo

_T = np.copy(Theta) # copia para salvar datos
h = 0.001 # cantidad de varianza
gradiente = np.zeros(2) # vector con los mismos componentes que Theta
tasa_aprendizaje = 0.01

repetir = 100

for _ in range(repetir):  
  for index_t, th in enumerate(Theta):
    _T = np.copy(Theta)

    _T[index_t] = _T[index_t] + h
    
    deriv = (func(_T) - func(Theta)) / h

    gradiente[index_t] = deriv

  Theta = Theta - tasa_aprendizaje * gradiente

  print(func(Theta))

  if (_ % 10 == 0):
    plt.plot(Theta[0], Theta[1], ".", c="red")

plt.plot(Theta[0], Theta[1], "o", c="green")
plt.show()

"""
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
"""