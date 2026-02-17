import numpy as np
import matplotlib.pyplot as plt

def plot_convergence(axis, history, value_label="f(x)"):
    iterations = [record.iteration for record in history]
    values = [record.value for record in history]

    axis.plot(iterations, values, 'b-o', linewidth=2, markersize=6)
    axis.set_xlabel('Iteración', fontsize=12)
    axis.set_ylabel(value_label, fontsize=12)
    axis.set_title('Convergencia del Algoritmo', fontsize=13)
    axis.grid(True, alpha=0.3)


def mark_endpoints(axis, horizontal, vertical):
    axis.plot(horizontal[0], vertical[0], 'go', markersize=12, label='Inicio')
    axis.plot(horizontal[-1], vertical[-1], 'r*', markersize=15, label='Final')


def evaluate_grid(objective, horizontal_grid, vertical_grid):
    surface_values = np.zeros_like(horizontal_grid)
    for row in range(horizontal_grid.shape[0]):
        for column in range(horizontal_grid.shape[1]):
            point = np.array([horizontal_grid[row, column], vertical_grid[row, column]])
            surface_values[row, column] = objective(point)
    return surface_values


def plot_one_dimension(history, objective):
    horizontal_trajectory = [record.point[0] for record in history]
    vertical_trajectory = [record.value for record in history]

    margin = 1
    lower_bound = min(horizontal_trajectory) - margin
    upper_bound = max(horizontal_trajectory) + margin

    domain = np.linspace(lower_bound, upper_bound, 500)
    function_values = [objective(np.array([point])) for point in domain]

    figure, (function_axis, convergence_axis) = plt.subplots(1, 2, figsize=(14, 5))

    function_axis.plot(domain, function_values, 'b-', linewidth=2, label='f(x)')
    function_axis.plot(
        horizontal_trajectory, vertical_trajectory,
        'ro-', markersize=8, linewidth=2, label='Trayectoria GD', alpha=0.7
    )
    mark_endpoints(function_axis, horizontal_trajectory, vertical_trajectory)
    function_axis.set_xlabel('x', fontsize=12)
    function_axis.set_ylabel('f(x)', fontsize=12)
    function_axis.set_title('Función y Trayectoria del Gradiente Descendente', fontsize=13)
    function_axis.legend()
    function_axis.grid(True, alpha=0.3)

    plot_convergence(convergence_axis, history)

    plt.tight_layout()
    plt.show()


def plot_two_dimensions(history, objective):
    horizontal_trajectory = [record.point[0] for record in history]
    vertical_trajectory = [record.point[1] for record in history]
    depth_trajectory = [record.value for record in history]

    margin = 1
    lower_horizontal = min(horizontal_trajectory) - margin
    upper_horizontal = max(horizontal_trajectory) + margin
    lower_vertical = min(vertical_trajectory) - margin
    upper_vertical = max(vertical_trajectory) + margin

    horizontal_domain = np.linspace(lower_horizontal, upper_horizontal, 100)
    vertical_domain = np.linspace(lower_vertical, upper_vertical, 100)
    horizontal_grid, vertical_grid = np.meshgrid(horizontal_domain, vertical_domain)
    surface_values = evaluate_grid(objective, horizontal_grid, vertical_grid)

    figure = plt.figure(figsize=(18, 5))

    surface_axis = figure.add_subplot(131, projection='3d')
    surface_axis.plot_surface(
        horizontal_grid, vertical_grid, surface_values,
        cmap='viridis', alpha=0.6, edgecolor='none'
    )
    surface_axis.plot(
        horizontal_trajectory, vertical_trajectory, depth_trajectory,
        'ro-', linewidth=3, markersize=6, label='Trayectoria GD'
    )
    surface_axis.scatter(
        horizontal_trajectory[0], vertical_trajectory[0], depth_trajectory[0],
        color='green', s=150, marker='o', label='Inicio'
    )
    surface_axis.scatter(
        horizontal_trajectory[-1], vertical_trajectory[-1], depth_trajectory[-1],
        color='red', s=200, marker='*', label='Final'
    )
    surface_axis.set_xlabel('x', fontsize=10)
    surface_axis.set_ylabel('y', fontsize=10)
    surface_axis.set_zlabel('f(x,y)', fontsize=10)
    surface_axis.set_title('Vista 3D', fontsize=12)
    surface_axis.legend()

    contour_axis = figure.add_subplot(132)
    contour = contour_axis.contour(
        horizontal_grid, vertical_grid, surface_values,
        levels=20, cmap='viridis', alpha=0.6
    )
    contour_axis.clabel(contour, inline=True, fontsize=8)
    contour_axis.plot(
        horizontal_trajectory, vertical_trajectory,
        'ro-', linewidth=2, markersize=8, label='Trayectoria GD'
    )
    mark_endpoints(contour_axis, horizontal_trajectory, vertical_trajectory)
    contour_axis.set_xlabel('x', fontsize=10)
    contour_axis.set_ylabel('y', fontsize=10)
    contour_axis.set_title('Curvas de Nivel y Trayectoria', fontsize=12)
    contour_axis.legend()
    contour_axis.grid(True, alpha=0.3)

    convergence_axis = figure.add_subplot(133)
    plot_convergence(convergence_axis, history, value_label="f(x,y)")

    plt.tight_layout()
    plt.show()


def plot_high_dimension(history):
    print("\n⚠️ Visualización completa no disponible para más de 2 dimensiones.")
    print("Mostrando solo gráfica de convergencia...\n")

    gradient_norms = [record.gradient_norm for record in history]
    iterations = [record.iteration for record in history]

    figure, (value_axis, gradient_axis) = plt.subplots(1, 2, figsize=(14, 5))

    plot_convergence(value_axis, history)

    gradient_axis.plot(iterations, gradient_norms, 'r-o', linewidth=2, markersize=6)
    gradient_axis.set_xlabel('Iteración', fontsize=12)
    gradient_axis.set_ylabel('||∇f(x)||', fontsize=12)
    gradient_axis.set_title('Convergencia de la Norma del Gradiente', fontsize=13)
    gradient_axis.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def visualize(history, objective):
    dimension = len(history[0].point)

    if dimension == 1:
        plot_one_dimension(history, objective)
    elif dimension == 2:
        plot_two_dimensions(history, objective)
    else:
        plot_high_dimension(history)