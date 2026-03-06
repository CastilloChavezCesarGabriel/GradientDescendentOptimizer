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

    learning_rate = request_float(f" Tasa de aprendizaje α (default: {default_rate}): ", default_rate, "tasa de aprendizaje")
    maximum_iterations = request_integer(f" Máximo de iteraciones N (default: {default_iterations}): ", default_iterations, "máximo de iteraciones")
    convergence_threshold = request_float(f" Criterio de paro ε (default: {default_tolerance}): ", default_tolerance, "criterio de paro")

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
        candidate_point, candidate_value, candidate_history = optimizer.descend(
            objective, optimizer.sample_point(bounds, dimension), learning_rate,
            maximum_iterations, convergence_threshold,
            bounds=bounds, silent=True
        )

        improved = candidate_value < best_value
        marker = " ← mejor hasta ahora" if improved else ""
        print(f" Reinicio {attempt + 1}: f(x) = {candidate_value:.6f}{marker}")

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

def announce(final_value, global_minimum):
    tolerance = 0.05
    reached = abs(final_value - global_minimum) <= tolerance
    if reached:
        print("\n[MINIMO GLOBAL] El algoritmo alcanzo el minimo global conocido.")
        print(f" f* = {global_minimum}, resultado: f(x) = {final_value:.6f}")
    else:
        print("\n[MINIMO LOCAL] El algoritmo convergió a un minimo local.")
        print(f" Minimo global conocido: f* = {global_minimum}")
        print(f" Resultado obtenido:     f(x) = {final_value:.6f}")
        print(f" Diferencia:             {abs(final_value - global_minimum):.6f}")

def display_results(final_point, final_value, history):
    print("\n" + "=" * 60)
    print(" RESULTADO FINAL")
    print("=" * 60)
    print("x* =", final_point)
    print("f(x*) =", final_value)

    print("\n" + "=" * 60)
    print(" HISTORIAL DE ITERACIONES")
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

def print_header(name):
    print("\n" + "=" * 60)
    print(f" GRADIENTE DESCENDENTE PARA: {name}")
    print("=" * 60)

def resolve_dimension(suggested_dimension):
    if suggested_dimension is not None:
        return suggested_dimension
    return request_integer("\nIngresa la dimensión d: ", None, "dimensión")

def request_details():
    print()
    details_text = input("¿Mostrar detalle de iteraciones? (s/n, default: n): ").strip().lower()
    return details_text == 's'

def print_banner():
    print(f"\n" + "=" * 60)
    print(" EJECUTANDO OPTIMIZACIÓN...")
    print("=" * 60 + "\n")

def improve(objective, bounds, dimension, learning_rate, maximum_iterations, convergence_threshold, restart_count, initial_value):
    print(f"\n--- EJECUTANDO {restart_count} REINICIOS ALEATORIOS ---\n")
    print(f"  Ejecución inicial: f(x) = {initial_value:.6f}")

    best_point, best_value, best_history = restart(
        objective, bounds, dimension, learning_rate,
        maximum_iterations, convergence_threshold, restart_count
    )

    if best_value < initial_value:
        print(f"\n Reinicio encontró mejor resultado: f(x) = {best_value:.6f} < {initial_value:.6f}")
        return best_point, best_value, best_history

    print(f"\n La ejecución inicial fue la mejor: f(x) = {initial_value:.6f}")
    return None, None, None

def offer_save(final_point, final_value, history):
    save_text = input("\n¿Guardar resultados en archivo? (s/n): ").strip().lower()
    if save_text != 's':
        return
    filepath = input("Nombre del archivo (default: resultados.txt): ").strip()
    save(filepath if filepath else "resultados.txt", final_point, final_value, history)

def offer_visualization(history, objective):
    response = input("\n¿Ver visualización gráfica? (s/n): ").strip().lower()
    if response == 's':
        visualization.visualize(history, objective)

def present(final_point, final_value, history, global_minimum, objective):
    display_results(final_point, final_value, history)
    announce(final_value, global_minimum)
    offer_save(final_point, final_value, history)
    offer_visualization(history, objective)

def main():
    name, suggested_dimension, objective, bounds, defaults, global_minimum = functions.select()
    print_header(name)
    dimension = resolve_dimension(suggested_dimension)
    learning_rate, maximum_iterations, convergence_threshold = configure(defaults)
    initial_point = locate(dimension, bounds)
    show_details = request_details()
    restart_count = collect_restarts()

    print_banner()
    final_point, final_value, history = optimizer.descend(
        objective, initial_point, learning_rate,
        maximum_iterations, convergence_threshold,
        bounds=bounds, silent=not show_details
    )

    if restart_count > 0:
        result = improve(
            objective, bounds, dimension, learning_rate,
            maximum_iterations, convergence_threshold,
            restart_count, final_value
        )
        if result[0] is not None:
            final_point, final_value, history = result

    present(final_point, final_value, history, global_minimum, objective)

if __name__ == "__main__":
    main()
