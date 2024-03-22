import numpy as np
import matplotlib.pyplot as plt

# Definición de la función
def func(a, b):
    return 10 - np.exp(-(a**2 + 3*b**2))

# Gradiente de la función
def gradient(a, b):
    da = 2 * a * np.exp(-(a**2 + 3*b**2))
    db = 6 * b * np.exp(-(a**2 + 3*b**2))
    return np.array([da, db])

# Descenso del gradiente
def gradient_descent(lr, iterations):
    # Inicialización aleatoria de los parámetros
    a = np.random.uniform(-1, 1)
    b = np.random.uniform(-1, 1)
    
    # Almacenamiento de la convergencia
    convergence = []
    
    for _ in range(iterations):
        # Calcular el gradiente
        grad = gradient(a, b)
        
        # Actualizar parámetros
        a -= lr * grad[0]
        b -= lr * grad[1]
        
        # Calcular el valor de la función y almacenar para convergencia
        loss = func(a, b)
        convergence.append(loss)
    
    return a, b, convergence

# Parámetros
lr = 0.01  # Learning rate
iterations = 200  # Número de iteraciones

# Ejecutar descenso del gradiente
opt_a, opt_b, convergence = gradient_descent(lr, iterations)

# Graficar la convergencia
plt.plot(convergence)
plt.title('Convergencia del Error')
plt.xlabel('Iteraciones')
plt.ylabel('Valor de la función')
plt.show()

print("Valor óptimo de a:", opt_a)
print("Valor óptimo de b:", opt_b)
print("Valor óptimo de la función:", func(opt_a, opt_b))
