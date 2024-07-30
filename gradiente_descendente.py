# Librerías para trabajar IAs
import numpy as np # cálculo numérico, álgebra
import scipy as sc # extensión de numpy
import matplotlib.pyplot as plt # visualización de datos

# Función a OPTIMIZAR (La función debe ser derivable, naturalmente)
# th[0] = x, th[1] = 1
func1 = lambda th: np.sin(1/2 * th[0] ** 2 - 1/4 * th[1] ** 2 + 3) * np.cos(2 * th[0] + 1 - np.e ** th[1])


resolucion = 200

_X = np.linspace(-2, 2, resolucion) # vector de 100 valores
_Y = np.linspace(-2, 2, resolucion) # vector de 100 valores

_Z = np.zeros((resolucion, resolucion)) # matriz de 100*100 con valores nulos (vacía)

# llenar la matriz _Z
for index_of_x, x_value in enumerate(_X): #enumerate() devuelve el índice del valor y el valor del vector parámetro
  for index_if_y, y_value in enumerate(_Y):
    _Z[index_if_y, index_of_x] = func1([x_value, y_value])

# visualizar datos
#plt.contour(_X, _Y, _Z, 100) # vista 'de líneas'
plt.contourf(_X, _Y, _Z, 100) # vista sólida
plt.colorbar()

# Theta
Theta = np.random.rand(2) * 4 - 2 # Genera dos números en el rango [-2, 2]
print(f'Theta: {Theta}')

plt.plot(Theta[0], Theta[1], "o", c="red") # marcar Theta

# usar derivadas parciales para obtener la pendiente en el punto dado
# y saber hacia dónde "descender" hasta el mínimo

h = 0.001 # incremento
gradiente = np.zeros(2)
tasa_aprendizaje = 0.01 # hiperparámetro que modifica el valor del módulo del vector gradiente (altera el "tamaño de los pasos")

repetir = 1000

for _ in range(repetir):  
  for it, th in enumerate(Theta):
    _T = np.copy(Theta)
    _T[it] = _T[it] + h
    deriv = (func1(_T) - func1(Theta)) / h # derivada por definición
    gradiente[it] = deriv

  Theta = Theta - tasa_aprendizaje * gradiente

  print(f'Theta: {func1(Theta)}')

  if (_ % 10 == 0):
    plt.plot(Theta[0], Theta[1], ".", c="blue")

plt.plot(Theta[0], Theta[1], "o", c="green")
plt.show()
