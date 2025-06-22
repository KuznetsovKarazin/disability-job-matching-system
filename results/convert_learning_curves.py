import pickle
import numpy as np

def describe_structure(obj, indent=0):
    """Print structure recursively for debug"""
    spacing = "  " * indent
    if isinstance(obj, dict):
        for k, v in obj.items():
            print(f"{spacing}- {k}: {type(v).__name__}")
            describe_structure(v, indent + 1)
    elif isinstance(obj, list):
        print(f"{spacing}List of {len(obj)} items")
    else:
        print(f"{spacing}{type(obj).__name__}")

# Путь к старому файлу
input_path = "results/learning_curves.pkl"
output_path = "results/learning_curves_compatible.pkl"

# Загрузка
with open(input_path, "rb") as f:
    data = pickle.load(f)

print("\n📋 Loaded structure:")
describe_structure(data)

# Преобразование
converted_data = {}
for model_name, metrics in data.items():
    new_metrics = {}
    for key, value in metrics.items():
        try:
            new_metrics[key] = value.tolist() if isinstance(value, np.ndarray) else value
        except Exception as e:
            print(f"⚠️ Could not convert {model_name} -> {key}: {e}")
    converted_data[model_name] = new_metrics

# Сохранение
with open(output_path, "wb") as f:
    pickle.dump(converted_data, f)

print(f"\n✅ Saved compatible version to: {output_path}")
