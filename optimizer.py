import numpy as np
from collections import namedtuple

IterationRecord = namedtuple(
    'IterationRecord',
    ['iteration', 'point', 'value', 'gradient_norm']
)

def differentiate(objective, point, step_size=1e-8):
    gradient = np.zeros_like(point)
    for index in range(len(point)):
        h = step_size * max(1.0, abs(point[index]))
        forward_point = np.copy(point)
        backward_point = np.copy(point)
        forward_point[index] += h
        backward_point[index] -= h
        gradient[index] = (objective(forward_point) - objective(backward_point)) / (2 * h)
    return gradient

def constrain(point, bounds):
    if bounds is None:
        return point
    return np.clip(point, bounds[0], bounds[1])

def sample_point(bounds, dimension):
    return np.random.uniform(bounds[0], bounds[1], dimension).tolist()

def report(message, silent):
    if not silent:
        print(message)

def backtrack(objective, current_point, gradient, learning_rate, current_value, bounds, threshold):
    effective_rate = learning_rate
    while True:
        next_point = constrain(current_point - effective_rate * gradient, bounds)
        if objective(next_point) < current_value:
            return next_point
        effective_rate *= 0.5
        if effective_rate < threshold:
            return next_point

def descend(objective, initial_point, learning_rate, maximum_iterations, convergence_threshold, bounds=None, silent=False):
    current_point = np.array(initial_point, dtype=float)
    history = []

    for iteration in range(maximum_iterations):
        gradient = differentiate(objective, current_point)
        gradient_norm = np.linalg.norm(gradient)
        current_value = objective(current_point)
        history.append(IterationRecord(iteration, current_point.copy(), current_value, gradient_norm))
        report(f"Iteración {iteration}: f(x) = {current_value:.6f}, ||g|| = {gradient_norm:.6f}", silent)

        if gradient_norm < convergence_threshold:
            report("\n Criterio de paro alcanzado (||∇f(x)|| < ε)", silent)
            break

        next_point = backtrack(objective, current_point, gradient, learning_rate, current_value, bounds, convergence_threshold)
        position_change = np.linalg.norm(next_point - current_point)
        if position_change < convergence_threshold:
            report("\n Posición estable (el punto ya no se mueve)", silent)
            break

        current_point = next_point

    return current_point, objective(current_point), history