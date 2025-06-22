
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np

def plot_metrics_comparison(metrics_df, save_dir='results'):
    """Barplot for Accuracy, Precision, Recall, F1, ROC AUC"""
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics_df[metrics].plot(kind='bar', ax=ax)
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'model_comparison_barplot.png'))
    plt.close()

def plot_f1_heatmap(metrics_df, save_dir='results'):
    """Heatmap for F1-score"""
    fig, ax = plt.subplots(figsize=(10, 1.5))
    f1_df = pd.DataFrame(metrics_df['f1_score']).T
    sns.heatmap(f1_df, annot=True, fmt=".3f", cmap="YlGnBu", cbar=False, ax=ax)
    plt.title('F1 Score Heatmap')
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'f1_score_heatmap.png'))
    plt.close()

def plot_model_ranking(metrics_df, save_dir='results'):
    """Ranking models based on F1 score"""
    sorted_df = metrics_df.sort_values(by='f1_score', ascending=False).reset_index()
    plt.figure(figsize=(8, 4))
    sns.barplot(x='f1_score', y='index', data=sorted_df, palette='viridis')
    plt.xlabel('F1 Score')
    plt.ylabel('Model')
    plt.title('Model Ranking by F1 Score')
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'f1_ranking.png'))
    plt.close()

def plot_training_times(metrics_df, results, save_dir='results'):
    """Training time comparison"""
    times = {name: results[name]['training_time'] for name in results if 'training_time' in results[name]}
    time_df = pd.DataFrame.from_dict(times, orient='index', columns=['training_time'])
    time_df = time_df.sort_values(by='training_time', ascending=True)
    plt.figure(figsize=(8, 4))
    sns.barplot(x='training_time', y=time_df.index, data=time_df.reset_index(), palette='mako')
    plt.xlabel('Seconds')
    plt.ylabel('Model')
    plt.title('Model Training Times')
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'training_times.png'))
    plt.close()

def plot_metric_distribution(metrics_df, save_dir='results'):
    """Distribution of all metrics"""
    melted = metrics_df.reset_index().melt(id_vars='index', var_name='metric', value_name='score')
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='metric', y='score', data=melted)
    plt.title('Distribution of Metric Scores Across Models')
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'metric_distribution.png'))
    plt.close()

def plot_correlation_heatmap(metrics_df, save_dir='results'):
    """Heatmap of metric correlations"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(metrics_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation between Evaluation Metrics")
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'metric_correlation_heatmap.png'))
    plt.close()

def plot_metric_trends(metrics_df, save_dir='results'):
    """Lineplot of metric evolution across models"""
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    plt.figure(figsize=(12, 6))
    for metric in metrics:
        plt.plot(metrics_df.index, metrics_df[metric], marker='o', label=metric)
    plt.title("Metric Trends Across Models")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'metric_trends_lineplot.png'))
    plt.close()
