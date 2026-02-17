import numpy as np
from collections import namedtuple

IterationRecord = namedtuple(
    'IterationRecord',
    ['iteration', 'point', 'value', 'gradient_norm']
)

def differentiate(objective, point, step_size=1e-8):
    gradient = np.zeros_like(point)
    for index in range(len(point)):
        forward_point = np.copy(point)
        backward_point = np.copy(point)
        forward_point[index] += step_size
        backward_point[index] -= step_size
        gradient[index] = (objective(forward_point) - objective(backward_point)) / (2 * step_size)
    return gradient

def constrain(point, bounds):
    return np.clip(point, bounds[0], bounds[1])


def sample_point(bounds, dimension):
    return np.random.uniform(bounds[0], bounds[1], dimension).tolist()

def descend(objective, initial_point, learning_rate, maximum_iterations, convergence_threshold, bounds=None, silent=False):
    current_point = np.array(initial_point, dtype=float)
    history = []

    for iteration in range(maximum_iterations):
        gradient = differentiate(objective, current_point)
        gradient_norm = np.linalg.norm(gradient)
        current_value = objective(current_point)

        history.append(IterationRecord(iteration, current_point.copy(), current_value, gradient_norm))

        if not silent:
            print(f"Iteración {iteration}: f(x) = {current_value:.6f}, ||g|| = {gradient_norm:.6f}")

        if gradient_norm < convergence_threshold:
            if not silent:
                print("\n Criterio de paro alcanzado (||∇f(x)|| < ε)")
            break

        next_point = current_point - learning_rate * gradient

        position_change = np.linalg.norm(next_point - current_point)
        if position_change < convergence_threshold:
            if not silent:
                print("\n Posición estable (el punto ya no se mueve)")
            break

        if bounds is not None:
            next_point = constrain(next_point, bounds)

        constrained_change = np.linalg.norm(next_point - current_point)
        if constrained_change < convergence_threshold:
            if not silent:
                print("\n Punto atrapado en el borde del dominio")
            break

        current_point = next_point

    final_value = objective(current_point)
    return current_point, final_value, history