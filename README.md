# Aplicación del Cálculo al Entrenamiento de Redes Neuronales
## Algoritmo de Regresión Lineal Simple
*Funciones Convexas*


Consiste en definir una función lineal que represente de la mejor manera posible una realidad planteada a partir de los datos porporcionados. Esta recta funciona como modelo para describir y predecir datos.
Cuando este modelo depende de una sola variable, estamos frente a un *Modelo de Regresión Lineal Simple*. Cuando hay más de una variable en cuestión, nos enfrentamos a un *Modelo de Regresión Lineal Múltiple*, donde ya no es una recta en un plano sino un plano en el espacio (dos variables), o un hiperplano en un espacio multidimensional (3 o más variables).

Utilizando el método de **Mínimos Cuadrados Ordinarios** (MCO), obtenemos (aplicando la ecuación) los valores de los parámetros para los cuales se miniza el error de nuestra función. Encontrando así, un modelo óptimo para los datos.

Debido a que implica el cálculos complejos y costosos (traspuesta, inversa, multiplicación vectorial, etc.), entre otras cosas, encontramos como alternativa al siguiente método o algoritmo.
## Algoritmo del Descenso del Gradiente
*Funciones No-Convexas*


También llamado Gradiente Descendente, es uno de los algoritmos más utilizados para entrenar modelos de inteligencia artificial, incluyendo redes neuronales. En pocas palabras, consiste en aplicar el gradiente descendente a una función de costo definida para encontrar los valores óptimos (mínimos locales) para las variables dadas. Es decir, es utilizado para reducir lo más posible el error en un modelo de datos.
Es una técnica de optimización.
### Conceptos clave
- **Función de Coste/Costo**: *función (derivable) que sirve para evaluar el desempeño de un modelo de datos: "mide qué tan bien hace su trabajo la IA en cuestión".*
- **Ratio de aprendizaje**:
*Un valor que multiplica al gradiente, que equivale al "tamaño de los pasos" que se da en cada iteración.*
### Videos de consulta
1. [¿Qué es el Descenso del Gradiente? Algoritmo de Inteligencia Artificial | DotCSV](https://www.youtube.com/watch?v=A6FiCDoz8_4)
1. [Regresión Lineal y Mínimos Cuadrados Ordinarios | DotCSV](https://www.youtube.com/watch?v=k964_uNn3l0)
1. [Regresión Líneal Simple 📈 En Python 🐍](https://www.youtube.com/watch?v=b7gOUbSmGIY)
1. [Gradiente Descendente Paso a Paso con Python: Un Algoritmo de Optimización para Machine Learning](https://www.youtube.com/watch?v=FNWbigoQNOk)
1. [Descenso de Gradiente. Cómo Aprenden las Redes Neuronales | Aprendizaje Profundo. Capítulo 2](https://www.youtube.com/watch?v=mwHiaTrQOiI)
1. [Implementación del Gradiente Descendiente en Python](https://www.youtube.com/watch?v=GaoUAlDHjOg)
1. [GRADIENTE DESCENDENTE (PYTHON)](https://www.youtube.com/watch?v=jk53nZxh4mI)
- - -
*Videla Guliotti, Tobías Uriel*