# Librerías necesarias: sklearn, numpy, scipy, matplotlib, etc.
# El siguiente código implementa una red neuronal desde cero, utilizando únicamente las librerías básicas.

import numpy as np  # Librería para realizar cálculos numéricos, especialmente con matrices y vectores.
import scipy as sc  # Extensión de numpy para cálculos científicos avanzados.
import matplotlib.pyplot as plt  # Librería para la visualización gráfica de datos.
import time  # Librería para manejar funciones relacionadas con el tiempo (por ejemplo, pausas).

from sklearn.datasets import make_circles  # Función para generar un conjunto de datos en forma de círculos.
from IPython.display import clear_output  # Función para limpiar la salida de la consola, útil en notebooks.

# CREAR EL DATASET
n = 500  # Número de registros en el conjunto de datos.
p = 2  # Cantidad de características (dimensiones) de cada registro.

# Genera un conjunto de datos con 500 muestras, distribuidas en dos círculos concéntricos.
X, Y = make_circles(n_samples=n, factor=0.5, noise=0.05)  # 'factor' y 'noise' controlan la separación y el ruido entre los círculos.
Y = Y[:, np.newaxis]  # Convierte el vector Y en una matriz columna para compatibilidad con la red neuronal.

# Visualiza los datos generados, separando los dos grupos de datos con diferentes colores.
plt.scatter(X[Y[:, 0] == 0, 0], X[Y[:, 0] == 0, 1], c='blue')  # Puntos del primer círculo en azul.
plt.scatter(X[Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1], c='orange')  # Puntos del segundo círculo en naranja.
plt.axis('equal')  # Asegura que los ejes X e Y tengan la misma escala.
# plt.show()  # Muestra el gráfico. (Comentar si no se quiere mostrar aquí).

# Implementación de una Red Neuronal para separar en dos clases la nube de puntos generada.

# CAPA DE LA RED
class neuralLayer():
  # Clase que representa una capa en la red neuronal, incluyendo pesos, sesgos y función de activación.
  def __init__(self, n_conexiones, n_neuronas, f_activacion):
    # Inicializa la capa con un número de conexiones, neuronas y una función de activación.
    self.f_activacion = f_activacion  # Almacena la función de activación (y su derivada).
    self.b = np.random.rand(1, n_neuronas) * 2 - 1  # Inicializa los sesgos con valores aleatorios entre [-1, 1].
    self.W = np.random.rand(n_conexiones, n_neuronas) * 2 - 1  # Inicializa los pesos con valores aleatorios entre [-1, 1].

# FUNCIONES DE ACTIVACIÓN
# Función sigmoide, que distorsiona los valores de entrada en un rango de [0, 1].
# La función se define como una tupla que contiene la función de activación y su derivada.
sigmoide = (lambda x: 1 / (1 + np.e ** (-x)), lambda x: x * (1 - x))

# Definición de la función para crear la red capa por capa.
def crear_RedNeuronal(topologia_red, f_activacion):
  # Función que construye una red neuronal dada una topología (lista de capas) y una función de activación.
  Red_Neuronal = []
  for index_capa, capa in enumerate(topologia_red[:-1]):
    # Para cada capa en la topología, crea una capa de la red neuronal y la añade a la lista.
    Red_Neuronal.append(neuralLayer(topologia_red[index_capa], topologia_red[index_capa + 1], f_activacion))
  return Red_Neuronal

# Ejemplo de topología de la red neuronal (3 capas: entrada, 2 ocultas, salida).
topologia_red = [p, 4, 8, 1]  # Red neuronal sencilla con 4 neuronas en la primera capa oculta y 8 en la segunda.
RN = crear_RedNeuronal(topologia_red, sigmoide)  # Crea la red neuronal utilizando la topología y la función sigmoide.

# ENTRENAMIENTO
# Función de coste que mide la diferencia entre la salida predicha y la real.
# La función también devuelve su derivada (necesaria para backpropagation).
f_coste = (lambda Yp, Yr: np.mean((Yp - Yr) ** 2), lambda Yp, Yr: (Yp - Yr))  # Error cuadrático medio (MSE).

def entrenar(red_neuronal, X, Y, f_coste, tasa_aprendizaje=0.01, entrenando=True):
  # Función para entrenar la red neuronal mediante descenso del gradiente.
  # 'X' es la entrada, 'Y' es la salida esperada, 'f_coste' es la función de coste.
  # 'tasa_aprendizaje' controla el tamaño de los ajustes en cada iteración.
  out = [(None, X)]  # Inicializa la lista de salidas, comenzando con la entrada X.
  
  # Forward pass: propaga las entradas hacia adelante a través de la red.
  for c, capa in enumerate(red_neuronal):
    suma_ponderada = out[-1][1] @ red_neuronal[c].W + red_neuronal[c].b  # Calcula la suma ponderada (z = W·X + b).
    activacion = red_neuronal[c].f_activacion[0](suma_ponderada)  # Aplica la función de activación.
    out.append((suma_ponderada, activacion))  # Almacena la suma ponderada y la activación en la lista de salidas.

  print(f_coste[0](out[-1][1], Y))  # Calcula y muestra el coste (error) después del forward pass.

  # Backward pass: ajusta los pesos y los sesgos mediante backpropagation.
  if entrenando:
    deltas = []  # Lista para almacenar los errores (deltas) en cada capa.

    # Backpropagation: calcular los deltas comenzando desde la última capa.
    for l in reversed(range(0, len(red_neuronal))):
      z = out[l + 1][0]  # Suma ponderada (z) en la capa l+1.
      a = out[l + 1][1]  # Activación (a) en la capa l+1.

      if l == len(red_neuronal) - 1:
        # Para la última capa, el delta se calcula usando la derivada de la función de coste y de activación.
        deltas.insert(0, f_coste[1](a, Y) * red_neuronal[l].f_activacion[1](a))
      else:
        # Para capas anteriores, el delta se propaga hacia atrás multiplicando por los pesos de la siguiente capa.
        deltas.insert(0, deltas[0] @ _W.T * red_neuronal[l].f_activacion[1](a))
          
      _W = red_neuronal[l].W  # Guarda los pesos actuales para usar en la siguiente iteración.

      # Actualización de los pesos y sesgos utilizando descenso del gradiente.
      red_neuronal[l].b = red_neuronal[l].b - np.mean(deltas[0], axis=0, keepdims=True) * tasa_aprendizaje  # Actualiza los sesgos.
      red_neuronal[l].W = red_neuronal[l].W - out[l][1].T @ deltas[0] * tasa_aprendizaje  # Actualiza los pesos.

  return out[-1][1]  # Devuelve la última activación (salida de la red neuronal).

# Test

errores = []  # Lista para almacenar el error en cada iteración.

for i in range(2500):  # Número de iteraciones para entrenar la red.
  # Entrena la red neuronal.
  Yp = entrenar(RN, X, Y, f_coste, tasa_aprendizaje=0.05, entrenando=True)
  
  if i % 100 == 0:
    errores.append(f_coste[0](Yp, Y))  # Almacena el error actual cada 100 iteraciones.

    res = 50  # Resolución para visualizar la malla.

    _x0 = np.linspace(-1.5, 1.5, res)  # Rango de valores para el eje x0.
    _x1 = np.linspace(-1.5, 1.5, res)  # Rango de valores para el eje x1.

    _Y = np.zeros((res, res))  # Inicializa una matriz para almacenar las predicciones.

    for i0, x0 in enumerate(_x0):
      for i1, x1 in enumerate(_x1):
        _Y[i0, i1] = entrenar(RN, np.array([[x0, x1]]), Y, f_coste, entrenando=False)[0][0]
        # Predice el valor para cada punto de la malla (sin entrenar).

    # Visualiza el resultado de la clasificación.
    plt.pcolormesh(_x0, _x1, _Y, cmap="coolwarm")  # Muestra la malla con las predicciones.
    plt.axis("equal")  # Asegura que los ejes tengan la misma escala.

    plt.scatter(X[Y[:, 0] == 0, 0], X[Y[:, 0] == 0, 1], c="skyblue")  # Puntos originales del primer círculo.
    plt.scatter(X[Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1], c="salmon")  # Puntos originales del segundo círculo.

    clear_output(wait=True)  # Limpia la salida anterior antes de mostrar la nueva.
    plt.show()  # Muestra el gráfico actualizado.

    plt.plot(range(len(errores)), errores)  # Grafica el error a lo largo del tiempo.
    plt.show()  # Muestra el gráfico del error.

    time.sleep(0.5)  # Pausa para que la visualización sea más clara.
