def parse_float(text, parameter_name):
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"'{text}' no es un valor numérico válido para {parameter_name}.")

def parse_positive_integer(text, parameter_name):
    try:
        value = int(text)
        if value <= 0:
            raise ValueError(f"{parameter_name} debe ser un entero positivo.")
        return value
    except ValueError as error:
        if "entero positivo" in str(error):
            raise
        raise ValueError(f"'{text}' no es un entero válido para {parameter_name}.")

def parse_vector(text, expected_dimension):
    components = text.strip().split()

    if len(components) != expected_dimension:
        raise ValueError(f"Se esperaban {expected_dimension} valores, se recibieron {len(components)}.")

    try:
        return [float(component) for component in components]
    except ValueError:
        raise ValueError(f"Entrada inválida. Ingresa {expected_dimension} valores numéricos separados por espacios.")

def parse_selection(text, maximum):
    try:
        value = int(text) - 1
        if value < 0 or value >= maximum:
            raise ValueError(f"Selección inválida. Elige un número entre 1 y {maximum}.")
        return value
    except ValueError as error:
        if "Selección inválida" in str(error):
            raise
        raise ValueError(f"'{text}' no es un número válido.")