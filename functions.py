import numpy as np
import input_parser

# Función 1D con múltiples mínimos locales por la oscilación del coseno;
# el optimizador puede quedar atrapado dependiendo del punto inicial
def cosine(point):
    return np.cos(5 * point[0]) + 2

# Función 1D cuyo único mínimo está en el borde derecho del dominio;
# sirve para verificar que el algoritmo sigue la pendiente constante hasta el límite
def linear(point):
    return -2 * point[0] + 5

# Función 2D donde los términos mixtos (0.5x + 0.5y) crean un valle curvo hacia el origen;
# las potencias ² y ⁴ del término mixto hacen que la superficie sea más empinada lejos del centro
def zakharov(point):
    mixed_term = 0.5 * point[0] + 0.5 * point[1]
    return point[0]**2 + point[1]**2 + mixed_term**2 + mixed_term**4

# Benchmark N-dimensional con muchos mínimos locales para exigir al optimizador;
# primero transforma las coordenadas con w = 1 + (x-1)/4, luego suma un término seno inicial,
# una sumatoria de penalizaciones entre componentes consecutivas y un término final con seno doble
def levy(point):
    dimension = len(point)
    transformed = 1 + (point - 1) / 4
    first_term = np.sin(np.pi * transformed[0]) ** 2

    summation = 0
    for index in range(dimension - 1):
        current = transformed[index]
        following = transformed[index + 1]
        summation += (current - 1) ** 2 * (1 + 10 * np.sin(np.pi * following) ** 2)

    last = transformed[dimension - 1]
    last_term = (last - 1) ** 2 * (1 + np.sin(2 * np.pi * last) ** 2)

    return first_term + summation + last_term

AVAILABLE = [
    ("Coseno: f(x) = cos(5x) + 2, x ∈ [-3, 3]", 1, cosine, (-3.0, 3.0), (0.07, 100, 1e-6), 1.0),
    ("Lineal: f(x) = -2x + 5, x ∈ [0, 3]", 1, linear, (0.0, 3.0), (1.5, 10, 1e-6), -1.0),
    ("Zakharov: f(x,y) = x² + y² + (0.5x+0.5y)² + (0.5x+0.5y)⁴, x,y ∈ [-5, 10]", 2, zakharov, (-5.0, 10.0), (0.035, 400, 1e-6), 0.0),
    ("Levy (multivariable), x ∈ [-10, 10]", None, levy, (-10.0, 10.0), (0.5, 500, 1e-8), 0.0),
]

# Muestra un menú numerado con las funciones disponibles y espera la elección del usuario;
# válida la entrada con parse_selection y devuelve el nombre, dimensión, función, límites,
# valores por defecto y mínimo global de la función elegida
def select():
    print("\nFunciones disponibles:")
    for number, (name, dimension, _, _bounds, _defaults, _minimum) in enumerate(AVAILABLE, start=1):
        hint = f" (dimensión: {dimension})" if dimension else ""
        print(f" {number}. {name}{hint}")

    while True:
        try:
            selection = input_parser.parse_selection(input("\nSelecciona una función (número): "), len(AVAILABLE))
            name, dimension, function, bounds, defaults, global_minimum = AVAILABLE[selection]
            return name, dimension, function, bounds, defaults, global_minimum
        except ValueError as error:
            print(f" {error}")