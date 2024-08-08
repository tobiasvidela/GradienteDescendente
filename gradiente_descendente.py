# Librerías para trabajar con Inteligencia Artificial y matemáticas
import numpy as np  # Importa numpy para realizar cálculos numéricos y álgebra.
import scipy as sc  # Importa scipy como extensión de numpy para cálculos científicos.
import matplotlib.pyplot as plt  # Importa matplotlib para la visualización de datos.

# Funciones objetivo a OPTIMIZAR (Las funciones deben ser derivables).
# Función 1: Una función derivable para optimizar. 
# th[0] = x, th[1] = y (representan las variables independientes).
func1 = lambda th: (th[0] ** 2 + th[1] - 11) ** 2 + (th[0] + th[1] ** 2 - 7) ** 2

# Función 2: Otra función derivable que combina senos y cosenos para la optimización.
func2 = lambda th: np.sin(1/2 * th[0] ** 2 - 1/4 * th[1] ** 2 + 3) * np.cos(2 * th[0] + 1 - np.e ** th[1])

# Resolución para la visualización de la función en una cuadrícula.
resolucion = 250

# Creación de vectores de valores para las variables x e y en el rango de [-5, 5].
_X = np.linspace(-5, 5, resolucion)  # Vector de 250 valores igualmente espaciados entre -5 y 5 para la variable X.
_Y = np.linspace(-5, 5, resolucion)  # Vector de 250 valores igualmente espaciados entre -5 y 5 para la variable Y.

# Inicialización de una matriz 250x250 con valores nulos para almacenar los resultados de la función.
_Z = np.zeros((resolucion, resolucion))  # Matriz 2D para almacenar los valores calculados de la función.

# Llenado de la matriz _Z con los valores de la función en cada punto de la cuadrícula.
for index_of_x, x_value in enumerate(_X):  # Recorre todos los valores de X (con su índice).
  for index_if_y, y_value in enumerate(_Y):  # Recorre todos los valores de Y (con su índice).
    _Z[index_if_y, index_of_x] = func1([x_value, y_value])  # Calcula el valor de la función en cada punto (x, y).

# Visualización de los datos generados por la función en la cuadrícula.
#plt.contour(_X, _Y, _Z, resolucion)  # Vista en 'líneas de contorno'.
plt.contourf(_X, _Y, _Z, resolucion)  # Vista 'sólida' de las líneas de contorno.
plt.colorbar()  # Añade una barra de color para indicar la escala de valores.

# Inicialización de los parámetros Tita (vector de parámetros) en un rango aleatorio [-5, 5].
Tita = np.random.rand(2) * 10 - 5  # Genera dos números aleatorios en el rango [-5, 5].
print(f'Tita: {Tita}')  # Imprime los valores iniciales de Tita.

# Marcar el punto inicial Tita en el gráfico.
plt.plot(Tita[0], Tita[1], "o", c="red")  # Marca el punto inicial con un círculo rojo.

# Usar derivadas parciales para obtener la pendiente en el punto dado
# y determinar la dirección en la que se debe "descender" hasta el mínimo.

h = 0.001  # Incremento para calcular la derivada numérica (diferencias finitas).
gradiente = np.zeros(2)  # Inicializa un vector de gradiente con ceros.
tasa_aprendizaje = 0.001  # Hiperparámetro que ajusta el tamaño del paso en cada iteración.

repetir = 1000  # Número de iteraciones del algoritmo de optimización.

# Bucle de optimización usando el método de descenso de gradiente.
for _ in range(repetir):  
  for it, th in enumerate(Tita):  # Itera sobre cada parámetro de Tita.
    _T = np.copy(Tita)  # Crea una copia de Tita para calcular la derivada.
    _T[it] += h  # Incrementa el parámetro actual por h.
    deriv = (func1(_T) - func1(Tita)) / h  # Calcula la derivada utilizando la definición de diferencias finitas.
    gradiente[it] = deriv  # Almacena la derivada en el vector de gradiente.

  # Actualiza los parámetros Tita en la dirección opuesta al gradiente (descenso de gradiente).
  Tita -= tasa_aprendizaje * gradiente

  # Cada 10 iteraciones, imprime el valor de la función objetivo y grafica el progreso.
  if (_ % 10 == 0):
    print(f'Tita: {func1(Tita)}')  # Imprime el valor de la función objetivo en la iteración actual.
    plt.plot(Tita[0], Tita[1], ".", c="blue")  # Marca el progreso de Tita con puntos azules.

# Marca el punto final Tita (mínimo encontrado) en el gráfico.
plt.plot(Tita[0], Tita[1], "o", c="green")  # Marca el punto final con un círculo verde.

# Muestra el gráfico con las líneas de contorno y el trayecto de descenso del gradiente.
plt.show()  # Muestra el gráfico generado.
