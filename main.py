import functions
import optimizer
import visualization
import input_parser


def is_within_bounds(initial_point, bounds):
    for value in initial_point:
        if value < bounds[0] or value > bounds[1]:
            return False
    return True


def request_float(prompt, default_value, parameter_name):
    while True:
        try:
            text = input(prompt).strip()
            return default_value if not text else input_parser.parse_float(text, parameter_name)
        except ValueError as error:
            print(f" {error}")


def request_integer(prompt, default_value, parameter_name):
    while True:
        try:
            text = input(prompt).strip()
            if not text:
                if default_value is not None:
                    return default_value
                raise ValueError(f"{parameter_name} es obligatorio.")
            return input_parser.parse_positive_integer(text, parameter_name)
        except ValueError as error:
            print(f" {error}")


def configure(defaults):
    default_rate, default_iterations, default_tolerance = defaults
    print("\nConfiguración de la optimización:")

    learning_rate = request_float(f"  Tasa de aprendizaje α (default: {default_rate}): ", default_rate, "tasa de aprendizaje")
    maximum_iterations = request_integer(f"  Máximo de iteraciones N (default: {default_iterations}): ", default_iterations, "máximo de iteraciones")
    convergence_threshold = request_float(f"  Criterio de paro ε (default: {default_tolerance}): ", default_tolerance, "criterio de paro")

    return learning_rate, maximum_iterations, convergence_threshold


def locate(dimension, bounds):
    while True:
        try:
            print(f"\nPunto inicial para {dimension}D (separado por espacios):")
            initial_point = input_parser.parse_vector(input("  "), dimension)
            if is_within_bounds(initial_point, bounds):
                return initial_point
            print(f" Punto fuera del dominio [{bounds[0]}, {bounds[1]}]. Intenta de nuevo.")
        except ValueError as error:
            print(f" {error}")


def collect_restarts():
    while True:
        try:
            restart_text = input("Número de reinicios aleatorios (default: 0): ").strip()
            if not restart_text or restart_text == "0":
                return 0
            return input_parser.parse_positive_integer(restart_text, "reinicios")
        except ValueError as error:
            print(f" {error}")


def restart(objective, bounds, dimension, learning_rate, maximum_iterations, convergence_threshold, restart_count):
    best_point = None
    best_value = float('inf')
    best_history = []

    for attempt in range(restart_count):
        random_start = optimizer.sample_point(bounds, dimension)
        candidate_point, candidate_value, candidate_history = optimizer.descend(
            objective, random_start, learning_rate,
            maximum_iterations, convergence_threshold,
            bounds=bounds, silent=True
        )

        improved = candidate_value < best_value
        marker = " ← mejor hasta ahora" if improved else ""
        print(f"  Reinicio {attempt + 1}: f(x) = {candidate_value:.6f}{marker}")

        if improved:
            best_point = candidate_point
            best_value = candidate_value
            best_history = candidate_history

    return best_point, best_value, best_history


def format_record(record):
    return (
        f"{record.iteration}: x = {record.point}, "
        f"f(x) = {record.value:.6f}, "
        f"||g|| = {record.gradient_norm:.6f}"
    )


def display_results(final_point, final_value, history):
    print("\n" + "=" * 60)
    print("   RESULTADO FINAL")
    print("=" * 60)
    print("x* =", final_point)
    print("f(x*) =", final_value)

    print("\n" + "=" * 60)
    print("   HISTORIAL DE ITERACIONES")
    print("=" * 60)

    for record in history:
        print(format_record(record))


def save(filepath, final_point, final_value, history):
    try:
        with open(filepath, 'w') as file:
            file.write(f"x* = {final_point}\n")
            file.write(f"f(x*) = {final_value}\n\n")
            for record in history:
                file.write(format_record(record) + "\n")
        print(f" Resultados guardados en '{filepath}'")
    except IOError:
        print(f" No se pudo guardar en '{filepath}'")


def main():
    name, suggested_dimension, objective, bounds, defaults = functions.select()

    print("\n" + "=" * 60)
    print(f"  GRADIENTE DESCENDENTE PARA: {name}")
    print("=" * 60)

    if suggested_dimension is not None:
        dimension = suggested_dimension
    else:
        dimension = request_integer("\nIngresa la dimensión d: ", None, "dimensión")

    learning_rate, maximum_iterations, convergence_threshold = configure(defaults)
    initial_point = locate(dimension, bounds)

    print()
    details_text = input("¿Mostrar detalle de iteraciones? (s/n, default: n): ").strip().lower()
    show_details = details_text == 's'
    restart_count = collect_restarts()

    print(f"\n" + "=" * 60)
    print("  EJECUTANDO OPTIMIZACIÓN...")
    print("=" * 60 + "\n")

    final_point, final_value, history = optimizer.descend(
        objective, initial_point, learning_rate,
        maximum_iterations, convergence_threshold,
        bounds=bounds, silent=not show_details
    )

    if restart_count > 0:
        print(f"\n--- EJECUTANDO {restart_count} REINICIOS ALEATORIOS ---\n")
        print(f"  Ejecución inicial: f(x) = {final_value:.6f}")

        best_point, best_value, best_history = restart(
            objective, bounds, dimension, learning_rate,
            maximum_iterations, convergence_threshold, restart_count
        )

        if best_value < final_value:
            print(f"\n Reinicio encontró mejor resultado: f(x) = {best_value:.6f} < {final_value:.6f}")
            final_point, final_value, history = best_point, best_value, best_history
        else:
            print(f"\n La ejecución inicial fue la mejor: f(x) = {final_value:.6f}")

    display_results(final_point, final_value, history)

    save_text = input("\n¿Guardar resultados en archivo? (s/n): ").strip().lower()
    if save_text == 's':
        filepath = input("Nombre del archivo (default: resultados.txt): ").strip()
        filepath = filepath if filepath else "resultados.txt"
        save(filepath, final_point, final_value, history)

    response = input("\n¿Ver visualización gráfica? (s/n): ").strip().lower()
    if response == 's':
        visualization.visualize(history, objective)

if __name__ == "__main__":
    main()