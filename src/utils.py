import torch
from typing import List, Tuple
from pathlib import Path
import re
import shutil
import pandas as pd

def get_best_device(options:List):
    """
    Returns the best available device in this order of preference:
      1. CUDA (NVIDIA GPU) if available
      2. MPS (Apple Silicon GPU) if available
      3. CPU (fallback)
    """
    if torch.cuda.is_available() and 'cuda' in options:
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        
    elif torch.backends.mps.is_available() and 'mps' in options:
        device = torch.device("mps")
        print("Using MPS (Apple Silicon)")
        
    else:
        device = torch.device("cpu")
        print("Using CPU")
        
    return device

def get_next_run_id(
    base_dir: Path,
    pattern: str = r".*?(\d+).*",
    delete_last: bool = True,
    thr: int = 3
) -> Tuple[int, int]:
    """
    Finds the highest existing run_XXXX number in base_dir and returns the next number.
    If delete_last=True and number of existing dirs > thr, deletes the directory
    with the lowest number before returning the next ID.

    Returns:
        Tuple[next_id: int, count_after_operation: int]
        
    Example:
        Existing: run_003, run_007, run_042
        → returns (43, 3)
        
        With delete_last=True, thr=3 and 4 directories exist:
        Deletes the lowest one (e.g. run_003), then returns next id + new count (3)
    """
    if not base_dir.exists():
        base_dir.mkdir(parents=True, exist_ok=True)
        return 1, 0

    existing = []
    
    # Collect all matching directories and their numbers
    for item in base_dir.iterdir():
        if not item.is_dir():
            continue
        match = re.match(pattern, item.name)
        if match:
            try:
                num = int(match.group(1))
                existing.append((num, item))
            except ValueError:
                continue

    if not existing:
        return 1, 0

    # Sort by number (smallest to largest)
    existing.sort(key=lambda x: x[0])
    current_count = len(existing)

    # Delete oldest (lowest number) if needed
    if delete_last and current_count > thr:
        oldest_num, oldest_path = existing[0]
        try:
            shutil.rmtree(oldest_path)
            print(f"Deleted oldest run directory: {oldest_path.name} (#{oldest_num})")
            current_count -= 1
        except Exception as e:
            print(f"Warning: Failed to delete {oldest_path.name}: {e}")

    next_id = max(num for num, _ in existing) + 1
    return next_id, current_count

from src.feature_generators.feature_generator import FeatureGenerator
from src.feature_generators.PCHEM.PCHEM_basic import PCHEMBaseline
from src.feature_generators.PLM.esm2 import EmbedderESM2
from src.feature_generators.PLM.pbert import EmbedderBERT

def get_embedder(emb_key:str) -> FeatureGenerator:

    """
    return corresponding feature generator (embedder)
    """

    if emb_key == 'ESM2':
        return EmbedderESM2
    if emb_key =='PBERT':
        return EmbedderBERT
    if emb_key == "PCHEM":
        return PCHEMBaseline

import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, precision_score, recall_score, confusion_matrix

def calculate_metrics(y_true, y_score, outdir ,thr:float = 0.5, title:str='validation', print_acc:bool=False):

        y_pred = y_score >= thr
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)

        result_metrics = {
            'accuracy': accuracy,
            'f1': f1,
            'mcc': mcc,
            'precision': precision,
            'recall': recall
        }

        c_matrix = confusion_matrix(y_true, y_pred)
        result_metrics['confusion_matrix'] = c_matrix

        if print_acc:
            print(f'\n--- ACHIEVED ACCURACY: {accuracy*100}% ---\n')

        # Set up a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 1.5]})
        fig.suptitle(title,fontsize=16)
        # Bar plot for metrics
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metrics_values = [result_metrics['accuracy'], result_metrics['precision'], 
                          result_metrics['recall'], result_metrics['f1']]
        bars = ax1.bar(metrics_names, metrics_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax1.set_ylim(0, 1)
        ax1.set_title('Classification Metrics', fontsize=14)
        ax1.set_ylabel('Score', fontsize=14)
        ax1.tick_params(axis='x', rotation=45)

        # Add value labels on top of bars
        for bar in bars:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.3f}', 
                     ha='center', va='bottom', fontsize=12)

        # Heatmap for confusion matrix
        sns.heatmap(c_matrix, annot=True, fmt='d', cmap='Blues', ax=ax2, 
                    cbar_kws={'label': 'Count'}, annot_kws={'size': 14})
        ax2.set_title('Confusion Matrix', fontsize=16)
        ax2.set_xlabel('Predicted Label', fontsize=14)
        ax2.set_ylabel('True Label', fontsize=14)

        # Adjust layout and display
        plt.tight_layout()
        plt.savefig(outdir / f"{title}.png")

def random_forest_feature_importance_plot(model, feature_names, outdir:Path, top_n=10, 
                                         title="Top Features",
                                         palette="viridis", figsize=(12, 7)):
    """
    Plots feature importance from a RandomForestClassifier (or DecisionTreeClassifier).
    
    Parameters:
    -----------
    model : fitted RandomForestClassifier or DecisionTreeClassifier
    feature_names : list or array of feature names (same length as n_features)
    top_n : int, default=10 - how many top features to display
    title : str, custom plot title
    palette : str, seaborn color palette name
    figsize : tuple, figure size (width, height)
    """
    # Get importance scores
    importances = model.feature_importances_
    
    # Create sorted DataFrame
    fi_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False).reset_index(drop=True)
    
    # Optional: add rank column
    fi_df['rank'] = fi_df.index + 1
    
    # Plot
    plt.figure(figsize=figsize)
    
    # Horizontal barplot with nice colors
    sns.barplot(
        data=fi_df.head(top_n),
        x='importance',
        y='feature',
        hue='feature',          # this gives nice color variation
        palette=palette,
        legend=False            # usually don't need legend when hue=feature
    )
    
    # Add value labels on bars
    for i, v in enumerate(fi_df['importance'].head(top_n)):
        plt.text(v + 0.005, i, f'{v:.4f}', 
                 va='center', fontsize=11, fontweight='medium')
    
    plt.title(title, fontsize=16, pad=15)
    plt.xlabel('Importance', fontsize=13)
    plt.ylabel('Feature', fontsize=13)
    
    # Make sure y-axis labels are readable
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    
    # Add subtle grid
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    
    plt.tight_layout()

    plt.savefig(outdir / f"feature_importance.png")