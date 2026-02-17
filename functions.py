import numpy as np
import input_parser

def cosine(point):
    return np.cos(5 * point[0]) + 2

def linear(point):
    return -2 * point[0] + 5

def polynomial(point):
    mixed_term = 0.5 * point[0] + 0.5 * point[1]
    return point[0]**2 + point[1]**2 + mixed_term**2 + mixed_term**4

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
    ("Coseno: f(x) = cos(5x) + 2, x ∈ [-3, 3]", 1, cosine, (-3.0, 3.0), (0.07, 100, 1e-6)),
    ("Lineal: f(x) = -2x + 5, x ∈ [0, 3]", 1, linear, (0.0, 3.0), (1.5, 10, 1e-6)),
    ("Polinomial: f(x,y) = x² + y² + (0.5x+0.5y)² + (0.5x+0.5y)⁴, x,y ∈ [-5, 10]", 2, polynomial, (-5.0, 10.0), (0.035, 400, 1e-6)),
    ("Levy (multivariable), x ∈ [-10, 10]", None, levy, (-10.0, 10.0), (0.5, 500, 1e-8)),
]

def select():
    print("\nFunciones disponibles:")
    for number, (name, dimension, _, _bounds, _defaults) in enumerate(AVAILABLE, start=1):
        hint = f" (dimensión: {dimension})" if dimension else ""
        print(f" {number}. {name}{hint}")

    while True:
        try:
            selection = input_parser.parse_selection(input("\nSelecciona una función (número): "), len(AVAILABLE))
            name, dimension, function, bounds, defaults = AVAILABLE[selection]
            return name, dimension, function, bounds, defaults
        except ValueError as error:
            print(f" {error}")