"""
Shared Model Evaluation Utilities
Used by all 3 model training scripts (Members 1, 2, 3)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report
)
import os

FIGURES_DIR = "results/figures/"
METRICS_DIR = "results/metrics/"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


def evaluate_model(model, X_test, y_test, model_name, disease_name):
    """
    Run full evaluation on a trained model and print/save results.

    Args:
        model: Trained sklearn-compatible model
        X_test: Test features
        y_test: True test labels
        model_name: e.g. 'Random Forest'
        disease_name: e.g. 'diabetes'
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": model_name,
        "Disease": disease_name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1 Score": round(f1_score(y_test, y_pred), 4),
        "AUC-ROC": round(roc_auc_score(y_test, y_proba), 4),
    }

    print(f"\n{'='*50}")
    print(f"  {model_name} — {disease_name.upper()}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if k not in ["Model", "Disease"]:
            print(f"  {k}: {v}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

    plot_confusion_matrix(y_test, y_pred, model_name, disease_name)
    plot_roc_curve(y_test, y_proba, model_name, disease_name)

    return metrics




def plot_confusion_matrix(y_test, y_pred, model_name, disease_name):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    plt.title(f'Confusion Matrix — {model_name} ({disease_name})')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    path = f"{FIGURES_DIR}cm_{disease_name}_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(path)
    plt.close()
    print(f"  ✅ Confusion matrix saved: {path}")


def plot_roc_curve(y_test, y_proba, model_name, disease_name):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.2f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve — {model_name} ({disease_name})')
    plt.legend()
    plt.tight_layout()
    path = f"{FIGURES_DIR}roc_{disease_name}_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(path)
    plt.close()
    print(f"  ✅ ROC curve saved: {path}")
