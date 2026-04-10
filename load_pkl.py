from pathlib import Path
import pickle
from typing import Any, Union


def load_pkl(file_path: Union[str, Path]) -> Any:
    """Load .pkl file and return the deserialized object."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() != ".pkl":
        raise ValueError(f"File suffix is not .pkl: {path}")

    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load pkl: {path}, error: {e}") from e

data = load_pkl(r"D:\Research\Life_Long_Learning\diffusion_proejct\diffusion\diffusion_behavior\experiment_results\results\kitchen__refinement_estimation__42________None.pkl")
if not isinstance(data, dict):
    raise TypeError(f"Loaded object is not a dict, got: {type(data)}")

if "results" not in data:
    raise KeyError("Key 'results' not found in loaded data.")

results = data["results"]

# Print each option's total_time_sec metric.
option_time_items = [
    (k, v) for k, v in results.items()
    if "_option_" in k and k.endswith("_total_time_sec")
]

if not option_time_items:
    print("No option total_time_sec metrics found in results.")
else:
    for key, value in sorted(option_time_items):
        print(f"{key}: {value}")