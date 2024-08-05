# Librerías: sklearn, tensorflow, pytorch, etc.
# Desde cero

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import time

from sklearn.datasets import make_circles
from IPython.display import clear_output

# CREAR EL DATASET
n = 500 # nro de registros en el conjunto de datos
p = 2 # cantidad de características de cada registro

X, Y = make_circles(n_samples = n, factor = 0.5, noise = 0.05)
Y = Y[:, np.newaxis]
print(X)
print(Y)

plt.scatter(X[Y[:,0] == 0 , 0], X[Y[:,0] == 0 , 1], c='blue')
plt.scatter(X[Y[:,0] == 1 , 0], X[Y[:,0] == 1 , 1], c='orange')
plt.axis('equal')
#plt.show()
# Red neuronal: separar en dos clases diferentes la nube de puntos

# CAPA DE LA RED
class neuralLayer():
  # Estructura de datos para crear capas en nuestra Red Neuronal
  def __init__(self, n_conexiones, n_neuronas, f_activacion):
    self.f_activacion = f_activacion
    self.b = np.random.rand(1, n_neuronas) * 2 - 1 # Vector columna con n_neuronas paramámetros - [-1 , 1]
    self.W = np.random.rand(n_conexiones, n_neuronas) * 2 - 1 # Matriz - [-1 , 1]

# FUNCIONES DE ACTIVACIÓN
# función sigmoide (Distorsión del x en un rango de 0 a 1)
sigmoide = (lambda x: 1 / (1 + np.e ** (-x)), lambda x: x * (1 - x))

#relu = lambda x: np.maximum(0, x)

#_x = np.linspace(-5, 5, 100) # vector de 100 valores en el rango [-5, 5]
#plt.plot(_x, sigmoide[0](_x))

# Crear red capa por capa
# l0 = neuralLayer(p, 4, sigmoide)
# l1 = neuralLayer(4, 8, sigmoide)
# ...

def crear_RedNeuronal(topologia_red,f_activacion):
  Red_Neuronal = []
  for index_capa, capa in enumerate(topologia_red[:-1]):
    Red_Neuronal.append(neuralLayer(topologia_red[index_capa], topologia_red[index_capa + 1], f_activacion))
  return Red_Neuronal

topologia_red = [p, 4, 8, 1] # sencilla
RN = crear_RedNeuronal(topologia_red, sigmoide)

# ENTRENAMIENTO
# funcion de coste
f_coste = (lambda Yp, Yr: np.mean((Yp - Yr) ** 2), lambda Yp, Yr: (Yp - Yr)) # error cuadrático medio

def entrenar(red_neuronal, X, Y, f_coste, tasa_aprendizaje = 0.01, entrenando=True):
  out = [(None, X)]
  # Forward pass
  for c, capa in enumerate(red_neuronal):
    suma_ponderada = out[-1][1] @ red_neuronal[c].W + red_neuronal[c].b
    activacion = red_neuronal[c].f_activacion[0](suma_ponderada)
    out.append((suma_ponderada, activacion))
  print(f_coste[0](out[-1][1], Y))

  # Backward pass
  if entrenando:
    # Backpropagation
    deltas = []

    for l in reversed(range(0, len(red_neuronal))):
      z = out[l+1][0] # suma ponderada
      a = out[l+1][1] # activación
      if l == len(red_neuronal) - 1:
        # calcular delta última capa
        deltas.insert(0, f_coste[1](a,Y) * red_neuronal[l].f_activacion[1](a))
      else:
        # calcular delta respecto a capa previa
        deltas.insert(0, deltas[0] @ _W.T * red_neuronal[l].f_activacion[1](a))

      _W = red_neuronal[l].W # Vector de parámetros W que conecta la capa actual con la capa anterior

      # Descendo del Gradiente
      red_neuronal[l].b = red_neuronal[l].b - np.mean(deltas[0], axis=0, keepdims=True) * tasa_aprendizaje
      red_neuronal[l].W = red_neuronal[l].W - out[l][1].T @ deltas[0] * tasa_aprendizaje
  
  return out[-1][1]

# Test

errores = []
for i in range(2500):
  # Entrenar la red
  Yp = entrenar(RN, X, Y, f_coste, tasa_aprendizaje=0.05, entrenando=True)
  if i % 25 == 0:
    errores.append(f_coste[0](Yp, Y))

    res = 50

    _x0 = np.linspace(-1.5, 1.5, res)
    _x1 = np.linspace(-1.5, 1.5, res)

    _Y = np.zeros((res, res))

    for i0, x0 in enumerate(_x0):
      for i1, x1 in enumerate(_x1):
        _Y[i0,i1] = entrenar(RN, np.array([[x0, x1]]), Y, f_coste, entrenando=False)[0][0]
    
    plt.pcolormesh(_x0, _x1, _Y, cmap="coolwarm")
    plt.axis("equal")

    plt.scatter(X[Y[:,0] == 0, 0], X[Y[:,0] == 0, 1], c="skyblue")
    plt.scatter(X[Y[:,0] == 1, 0], X[Y[:,0] == 1, 1], c="salmon")

    clear_output(wait=True)
    plt.show()
    plt.plot(range(len(errores)), errores)
    plt.show()
    time.sleep(0.5)