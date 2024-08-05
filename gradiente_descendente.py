# Librerías para trabajar IAs
import numpy as np # cálculo numérico, álgebra
import scipy as sc # extensión de numpy
import matplotlib.pyplot as plt # visualización de datos

# Función a OPTIMIZAR (La función debe ser derivable, naturalmente)
# th[0] = x, th[1] = 1
func1 = lambda th: (th[0] ** 2 + th[1] - 11) ** 2 + (th[0] + th[1] ** 2 - 7) ** 2
func2 = lambda th: np.sin(1/2 * th[0] ** 2 - 1/4 * th[1] ** 2 + 3) * np.cos(2 * th[0] + 1 - np.e ** th[1])
# func3 = lambda th: (1 - th[0]) ** 2 + 100 * (th[1] - th[0] ** 2) ** 2 # XXXXXXXX

resolucion = 250

_X = np.linspace(-5, 5, resolucion) # vector de 250 valores
_Y = np.linspace(-5, 5, resolucion) # vector de 250 valores

_Z = np.zeros((resolucion, resolucion)) # matriz de 250*250 con valores nulos (vacía)

# llenar la matriz _Z
for index_of_x, x_value in enumerate(_X): #enumerate() devuelve el índice del valor y el valor del vector parámetro
  for index_if_y, y_value in enumerate(_Y):
    _Z[index_if_y, index_of_x] = func1([x_value, y_value])

# visualizar datos
#plt.contour(_X, _Y, _Z, resolucion) # vista 'de líneas'
plt.contourf(_X, _Y, _Z, resolucion) # vista sólida
plt.colorbar()

# Tita
Tita = np.random.rand(2) * 10 - 5 # Genera dos números en el rango [-5, 5]
print(f'Tita: {Tita}')

plt.plot(Tita[0], Tita[1], "o", c="red") # marcar Tita

# usar derivadas parciales para obtener la pendiente en el punto dado
# y saber hacia dónde "descender" hasta el mínimo

h = 0.001 # incremento
gradiente = np.zeros(2)
tasa_aprendizaje = 0.001 # hiperparámetro que modifica el valor del módulo del vector gradiente (altera el "tamaño de los pasos")

repetir = 1000

for _ in range(repetir):  
  for it, th in enumerate(Tita):
    _T = np.copy(Tita)
    _T[it] = _T[it] + h
    deriv = (func1(_T) - func1(Tita)) / h # derivada por definición
    gradiente[it] = deriv

  Tita = Tita - tasa_aprendizaje * gradiente


  if (_ % 10 == 0):
    print(f'Tita: {func1(Tita)}')
    plt.plot(Tita[0], Tita[1], ".", c="blue")

plt.plot(Tita[0], Tita[1], "o", c="green")
plt.show()
