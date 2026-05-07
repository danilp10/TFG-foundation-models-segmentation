import os

import pandas as pd


def save_results(results_dict, output_path):
    df = pd.DataFrame(results_dict)
    if os.path.exists(output_path):
        df.to_csv(output_path, mode="a", header=False, index=False)
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
