# 📄 results/plot_learning_curves.py

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

# === Paths ===
input_path = "results/learning_curves_compatible.pkl"
output_dir = "results/learning_curves"
os.makedirs(output_dir, exist_ok=True)

# === Load Curves ===
with open(input_path, "rb") as f:
    learning_curves = pickle.load(f)

# === Individual Plots ===
for model_name, data in learning_curves.items():
    train_sizes = data["train_sizes"]
    train_scores = data["train_scores"]
    test_scores = data["test_scores"]

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color="orange", label="Training score")
    plt.plot(train_sizes, test_mean, 'o-', color="orangered", label="Validation score")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color="orange")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2, color="orangered")

    plt.title(f"📈 Learning Curve: {model_name}")
    plt.xlabel("Training Set Size")
    plt.ylabel("F1 Score")
    plt.legend(loc="best")
    plt.grid(True)

    output_file = os.path.join(output_dir, f"learning_curve_{model_name}.png")
    plt.savefig(output_file)
    plt.close()

print("✅ Individual plots saved.")

# === Combined Comparison Plot (Validation Scores) ===
plt.figure(figsize=(12, 7))

for model_name, data in learning_curves.items():
    test_mean = np.mean(data["test_scores"], axis=1)
    plt.plot(data["train_sizes"], test_mean, marker='o', label=model_name)

plt.title("📊 Validation F1 Scores Comparison")
plt.xlabel("Training Set Size")
plt.ylabel("Validation F1 Score")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()

combined_file = os.path.join(output_dir, "combined_validation_scores.png")
plt.savefig(combined_file)
plt.close()

print(f"✅ Combined comparison plot saved: {combined_file}")
