#   Codigo que implementa el metodo de gradiente 
#   para minimizar una funcion de dos variables
# 
#           Autor:
#   Dr. Ivan de Jesus May-Cen
#   imaycen@hotmail.com
#   Version 1.0 : 16/10/2024
#

import numpy as np
import matplotlib.pyplot as plt

# Función objetivo f(x, y) = (x - 2)^2 + (y - 3)^2
def objective_function(x, y):
    return (x - 2)**2 + (y - 3)**2

# Gradiente de la función objetivo: ∇f(x, y) = [2*(x-2), 2*(y-3)]
def gradient(x, y):
    grad_x = 2 * (x - 2)
    grad_y = 2 * (y - 3)
    return np.array([grad_x, grad_y])

# Parámetros de la optimización
learning_rate = 0.1  # Tasa de aprendizaje
max_iters = 100  # Número máximo de iteraciones
tolerance = 1e-6  # Tolerancia para detener el algoritmo

# Inicialización
x = np.random.randn()  # Punto inicial aleatorio para x
y = np.random.randn()  # Punto inicial aleatorio para y
history = []  # Para guardar el valor de la función en cada iteración
xy_values = []  # Para guardar los valores de (x, y) en cada iteración

# Método del gradiente descendente
for i in range(max_iters):
    # Guardamos el valor de la función objetivo y los valores de (x, y)
    history.append(objective_function(x, y))
    xy_values.append([x, y])
    
    # Calculamos el gradiente
    grad = gradient(x, y)
    
    # Actualizamos los valores de x e y
    x_new = x - learning_rate * grad[0]
    y_new = y - learning_rate * grad[1]
    
    # Si la diferencia es menor que la tolerancia, terminamos
    if np.sqrt((x_new - x)**2 + (y_new - y)**2) < tolerance:
        print(f"Convergencia alcanzada en {i+1} iteraciones.")
        break
    
    x, y = x_new, y_new

# Si no alcanzamos convergencia
if i == max_iters - 1:
    print("Se alcanzó el número máximo de iteraciones.")

# Mostrar resultados finales
print(f"Valor mínimo encontrado: x = {x:.4f}, y = {y:.4f}")
print(f"Valor de la función objetivo en el mínimo: f(x, y) = {objective_function(x, y):.4f}")

# Graficar la convergencia
plt.figure(figsize=(12, 6))

# Subplot 1: Convergencia de la función objetivo
plt.subplot(1, 2, 1)
plt.plot(history, marker='o')
plt.title("Convergencia del valor de la función objetivo")
plt.xlabel("Iteraciones")
plt.ylabel("f(x, y)")
plt.grid(True)

# Subplot 2: Camino de optimización en el espacio (x, y)
plt.subplot(1, 2, 2)

# Generamos los valores de la función en una malla de (x, y) para graficar los contornos
x_range = np.linspace(-1, 5, 100)
y_range = np.linspace(0, 6, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = objective_function(X, Y)

# Graficamos los contornos de la función objetivo
plt.contour(X, Y, Z, levels=30, cmap='viridis')
xy_values = np.array(xy_values)

# Graficamos el camino de las iteraciones
plt.plot(xy_values[:, 0], xy_values[:, 1], 'r.-', label="Camino de optimización")

# Marcamos la solución final
plt.scatter(x, y, color='green', zorder=5, label=f"Solución: (x = {x:.4f}, y = {y:.4f})")
plt.title("Camino de optimización en el espacio (x, y)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)

# Mostrar gráficas
plt.tight_layout()
plt.savefig("convergencia-y-solucion-2-variables.eps")
plt.show()

