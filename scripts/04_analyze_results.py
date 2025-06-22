#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Configuration ===
metrics_path = "results/metrics_summary.csv"
complexity_path = "results/model_complexity.csv"
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# === Load and preprocess data ===
print("📊 Loading data...")

# Load and clean metrics
metrics_df = pd.read_csv(metrics_path)
metrics_df.rename(columns={metrics_df.columns[0]: "model"}, inplace=True)

# Load complexity data
complexity_df = pd.read_csv(complexity_path)
complexity_df.rename(columns={"Model": "model",
                              "TrainingTime_sec": "training_time_s",
                              "PredictionTime_sec": "prediction_time_s",
                              "ModelSize_KB": "model_size_kb"}, inplace=True)

# Merge
df = pd.merge(metrics_df, complexity_df, on="model", how="outer")
print(f"✅ Loaded and merged {df.shape[0]} models")

# === Setup visualization style ===
sns.set(style="whitegrid", font_scale=1.1)

# === Save summary table ===
df.to_csv(f"{output_dir}/merged_model_summary.csv", index=False)

# === Main metrics barplots ===
metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
for metric in metrics:
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="model", y=metric, hue="model", palette="Blues_d", legend=False)
    plt.title(f"{metric.upper()} by Model")
    plt.ylabel(metric.upper())
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{metric}_barplot.png")
    plt.close()

# === F1-score ranking ===
df_sorted = df.sort_values(by="f1_score", ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x="f1_score", y="model", data=df_sorted, palette="crest")
plt.title("🔝 F1-score Ranking")
plt.xlabel("F1 Score")
plt.ylabel("Model")
plt.tight_layout()
plt.savefig(f"{output_dir}/f1_ranking.png")
plt.close()

# === Heatmap of f1_score values ===
plt.figure(figsize=(10, 4))
heat_data = df.pivot_table(index="model", values="f1_score")
sns.heatmap(heat_data, annot=True, cmap="YlGnBu", fmt=".3f")
plt.title("F1 Score Heatmap")
plt.tight_layout()
plt.savefig(f"{output_dir}/f1_score_heatmap.png")
plt.close()

# === Metric correlation heatmap ===
plt.figure(figsize=(12, 8))
sns.heatmap(df.select_dtypes("number").corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Metric Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{output_dir}/metric_correlation_heatmap.png")
plt.close()

# === Metric distribution plots ===
melted = df.melt(id_vars="model", value_vars=metrics, var_name="Metric", value_name="Value")
plt.figure(figsize=(10, 6))
sns.boxplot(data=melted, x="Metric", y="Value")
plt.title("Metric Distribution Across Models")
plt.tight_layout()
plt.savefig(f"{output_dir}/metric_distribution.png")
plt.close()

# === Training Time Only ===
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x="model", y="training_time_s", palette="rocket")
plt.title("⏱ Training Time per Model (seconds)")
plt.ylabel("Training Time (s)")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(f"{output_dir}/training_time_separate_barplot.png")
plt.close()

# === Prediction Time Only ===
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x="model", y="prediction_time_s", palette="mako")
plt.title("⚡ Prediction Time per Model (seconds)")
plt.ylabel("Prediction Time (s)")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(f"{output_dir}/prediction_time_separate_barplot.png")
plt.close()

# === Composite Performance Summary ===
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("📈 Model Performance Summary", fontsize=16)

# F1
sns.barplot(data=df, x="model", y="f1_score", ax=axes[0, 0], palette="cool")
axes[0, 0].set_title("F1 Score")

# ROC AUC
sns.barplot(data=df, x="model", y="roc_auc", ax=axes[0, 1], palette="crest")
axes[0, 1].set_title("ROC AUC")

# Accuracy
sns.barplot(data=df, x="model", y="accuracy", ax=axes[1, 0], palette="viridis")
axes[1, 0].set_title("Accuracy")

# Training Time
sns.barplot(data=df, x="model", y="training_time_s", ax=axes[1, 1], palette="rocket")
axes[1, 1].set_title("Training Time (s)")

for ax in axes.flatten():
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"{output_dir}/composite_model_comparison.png")
plt.close()

# === F1 vs Model Size ===
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="model_size_kb", y="f1_score", hue="model", s=150)
plt.title("📏 F1 Score vs Model Size")
plt.xlabel("Model Size (KB)")
plt.tight_layout()
plt.savefig(f"{output_dir}/f1_vs_model_size.png")
plt.close()

print("✅ All visualizations saved to:", output_dir)

# === Aggregated Comparison Barplot (Grouped) ===
plt.figure(figsize=(14, 6))

# Melt metrics for grouped barplot
melted_metrics = df.melt(id_vars="model", value_vars=metrics, var_name="Metric", value_name="Score")

# Grouped barplot
sns.barplot(data=melted_metrics, x="model", y="Score", hue="Metric", palette="Set2")
plt.title("📊 Model Performance Comparison (Grouped by Metric)")
plt.ylabel("Score")
plt.xlabel("Model")
plt.xticks(rotation=30, ha='right')
plt.ylim(0.0, 1.05)
plt.legend(title="Metric", loc="upper right")
plt.tight_layout()
plt.savefig(f"{output_dir}/model_comparison_grouped_barplot.png")
plt.close()