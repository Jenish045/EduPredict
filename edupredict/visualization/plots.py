"""
Diagnostic visualization tools for EduPredict.

Uses matplotlib (as allowed by project policy) to generate clean, high-resolution
plots for loss curves, confusion matrices, residuals, feature importances, and ROC curves.
Gracefully degrades to structured text formatting if matplotlib is not available.
"""

import os
from typing import Sequence, List, Optional, Any, Union, Dict, Tuple

Number = Union[int, float]


def plot_loss_curve(
    loss_history: Sequence[Number],
    title: str = "Training Loss Convergence",
    save_path: Optional[str] = None
) -> str:
    """
    Plot training loss over epochs.
    """
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(loss_history) + 1), loss_history, color="#1f77b4", linewidth=2.0)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=200)
            plt.close()
            return f"Loss curve saved to {save_path}"
        else:
            plt.close()
            return "Loss curve rendered."
    except ImportError:
        # Text-based fallback
        lines = [f"=== {title} (First 5 & Last 5) ==="]
        for ep in range(min(5, len(loss_history))):
            lines.append(f"Epoch {ep+1:4d}: Loss = {loss_history[ep]:.6f}")
        if len(loss_history) > 10:
            lines.append("...")
            for ep in range(len(loss_history) - 5, len(loss_history)):
                lines.append(f"Epoch {ep+1:4d}: Loss = {loss_history[ep]:.6f}")
        return "\n".join(lines)


def plot_confusion_matrix_heatmap(
    cm: Sequence[Sequence[int]],
    labels: Sequence[Any],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None
) -> str:
    """
    Render confusion matrix heatmap.
    """
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.matshow(cm, cmap="Blues", alpha=0.85)
        fig.colorbar(cax)

        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels([str(l) for l in labels], fontsize=11)
        ax.set_yticklabels([str(l) for l in labels], fontsize=11)

        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, str(cm[i][j]), ha="center", va="center", color="black", fontsize=13, fontweight="bold")

        plt.title(title, pad=20, fontsize=14, fontweight="bold")
        plt.xlabel("Predicted Label", fontsize=12)
        plt.ylabel("True Label", fontsize=12)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=200)
            plt.close()
            return f"Confusion matrix saved to {save_path}"
        else:
            plt.close()
            return "Confusion matrix rendered."
    except ImportError:
        lines = [f"=== {title} ==="]
        header = "True \\ Pred | " + " | ".join(f"{str(l):>6}" for l in labels)
        lines.append(header)
        lines.append("-" * len(header))
        for i, row in enumerate(cm):
            lines.append(f"{str(labels[i]):>10} | " + " | ".join(f"{v:>6}" for v in row))
        return "\n".join(lines)


def plot_feature_importances(
    feature_names: Sequence[str],
    importances: Sequence[float],
    title: str = "Feature Importances",
    save_path: Optional[str] = None
) -> str:
    """
    Horizontal bar chart of feature importances.
    """
    try:
        import matplotlib.pyplot as plt
        sorted_pairs = sorted(zip(feature_names, importances), key=lambda x: x[1])
        feats = [p[0] for p in sorted_pairs]
        imps = [p[1] for p in sorted_pairs]

        plt.figure(figsize=(8, 5))
        plt.barh(feats, imps, color="#2ca02c", alpha=0.85)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel("Relative Importance", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5, axis="x")
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=200)
            plt.close()
            return f"Feature importances saved to {save_path}"
        else:
            plt.close()
            return "Feature importances rendered."
    except ImportError:
        lines = [f"=== {title} ==="]
        for f, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
            bar = "#" * int(imp * 30)
            lines.append(f"{f:<25} {imp:6.4f} | {bar}")
        return "\n".join(lines)


def plot_actual_vs_predicted(
    y_true: Sequence[Number],
    y_pred: Sequence[Number],
    title: str = "Actual vs. Predicted Final Exam Scores",
    save_path: Optional[str] = None
) -> str:
    """
    Scatter plot comparing ground truth continuous values against predictions.
    """
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 6))
        plt.scatter(y_true, y_pred, alpha=0.6, color="#4b6584", edgecolors="none")
        
        mn = min(min(y_true), min(y_pred))
        mx = max(max(y_true), max(y_pred))
        plt.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Ideal 1:1 Fit")

        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel("Actual Score", fontsize=12)
        plt.ylabel("Predicted Score", fontsize=12)
        plt.legend(loc="upper left")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=200)
            plt.close()
            return f"Actual vs predicted plot saved to {save_path}"
        else:
            plt.close()
            return "Plot rendered."
    except ImportError:
        return "Matplotlib not installed. Skipping scatter plot."


def plot_roc_curves(
    curves: Dict[str, Tuple[List[float], List[float], float]],
    title: str = "ROC Curves Comparison",
    save_path: Optional[str] = None
) -> str:
    """
    Plot ROC Curves for multiple classifiers on the same figure.
    
    Args:
        curves: Mapping of model_name -> (fpr, tpr, auc_score).
        title: Plot title.
        save_path: Optional path to save PNG.
    """
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        for idx, (name, (fpr, tpr, auc_val)) in enumerate(curves.items()):
            color = colors[idx % len(colors)]
            plt.plot(fpr, tpr, color=color, linewidth=2.0, label=f"{name} (AUC = {auc_val:.3f})")

        # Diagonal chance line
        plt.plot([0.0, 1.0], [0.0, 1.0], "k--", alpha=0.7, label="Chance (AUC = 0.500)")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate (FPR)", fontsize=12)
        plt.ylabel("True Positive Rate (TPR / Recall)", fontsize=12)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=200)
            plt.close()
            return f"ROC curves saved to {save_path}"
        else:
            plt.close()
            return "ROC curves rendered."
    except ImportError:
        lines = [f"=== {title} ==="]
        for name, (_, _, auc_val) in curves.items():
            lines.append(f"{name:<25}: AUC = {auc_val:.4f}")
        return "\n".join(lines)
