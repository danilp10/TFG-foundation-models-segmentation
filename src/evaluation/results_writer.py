import os
import pandas as pd


def save_results(results_dict, output_path):
    """Guarda un diccionario de resultados como fila en un CSV. Si el archivo
    ya existe, añade la fila al final, si no existe, lo crea con cabecera.
    Además, crea la carpeta de destino si no existe."""
    df = pd.DataFrame(results_dict)
    if os.path.exists(output_path):
        df.to_csv(output_path, mode="a", header=False, index=False)
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
