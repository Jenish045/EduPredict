"""
Diagnostic visualization tools for EduPredict.

Uses matplotlib (as allowed by project policy) to generate clean, high-resolution
plots for loss curves, confusion matrices, residuals, feature importances, and ROC curves.
Gracefully degrades to structured text formatting if matplotlib is not available.
"""

import os
from typing import Sequence, List, Optional, Any, Union, Dict, Tuple
from edupredict.data.uci_metadata import get_column_display_name

os.environ.setdefault("MPLBACKEND", "Agg")
if "MPLCONFIGDIR" not in os.environ:
    _temp_dir = os.environ.get("TEMP", os.environ.get("TMP", "C:\\temp"))
    _cache_dir = os.path.join(_temp_dir, "matplotlib_cache")
    os.makedirs(_cache_dir, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = _cache_dir

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
        feats = [get_column_display_name(p[0]) for p in sorted_pairs]
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


def plot_correlation_heatmap(
    corr_matrix: Sequence[Sequence[float]],
    feature_names: Sequence[str],
    title: str = "Pearson Feature Correlation Matrix",
    save_path: Optional[str] = None
) -> str:
    """
    Render correlation matrix heatmap computed from first principles.
    """
    try:
        import matplotlib.pyplot as plt
        p = len(feature_names)
        display_names = [get_column_display_name(f) for f in feature_names]
        fig, ax = plt.subplots(figsize=(max(6, p * 1.1), max(5, p * 0.9)))
        cax = ax.matshow(corr_matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
        cbar.set_label("Pearson Correlation (r)", fontsize=11)

        ax.set_xticks(range(p))
        ax.set_yticks(range(p))
        ax.set_xticklabels(display_names, rotation=35, ha="left", fontsize=9)
        ax.set_yticklabels(display_names, fontsize=9)

        for i in range(p):
            for j in range(p):
                val = corr_matrix[i][j]
                text_color = "white" if abs(val) > 0.55 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=10, fontweight="bold")

        plt.title(title, pad=30, fontsize=13, fontweight="bold")
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=200)
            plt.close()
            return f"Correlation heatmap saved to {save_path}"
        else:
            plt.close()
            return "Correlation heatmap rendered."
    except ImportError:
        lines = [f"=== {title} ==="]
        header = "Feature | " + " | ".join(f"{f[:8]:>8}" for f in feature_names)
        lines.append(header)
        for i, f in enumerate(feature_names):
            row_str = f"{f[:10]:<10} | " + " | ".join(f"{corr_matrix[i][j]:8.2f}" for j in range(p))
            lines.append(row_str)
        return "\n".join(lines)


def plot_feature_distributions(
    features_dict: Dict[str, Sequence[Number]],
    title: str = "Student Numerical Feature Distributions",
    save_path: Optional[str] = None
) -> str:
    """
    Render multi-panel histograms for numerical input features.
    """
    try:
        import matplotlib.pyplot as plt
        num_features = len(features_dict)
        cols = 2
        rows = (num_features + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(10, rows * 3.5))
        axes = [axes] if num_features == 1 else (axes.flatten() if hasattr(axes, "flatten") else [axes])

        palette = ["#3867d6", "#20bf6b", "#fa8231", "#8854d0", "#eb3b5a"]

        for idx, (fname, vals) in enumerate(features_dict.items()):
            ax = axes[idx]
            clean_vals = [float(v) for v in vals if v is not None]
            color = palette[idx % len(palette)]
            ax.hist(clean_vals, bins=15, color=color, edgecolor="black", alpha=0.75)
            ax.set_title(get_column_display_name(fname), fontsize=11, fontweight="bold")
            ax.set_xlabel("Value", fontsize=10)
            ax.set_ylabel("Frequency", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.4)

        # Hide any unused subplots
        for idx in range(num_features, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            plt.close()
            return f"Feature distributions saved to {save_path}"
        else:
            plt.close()
            return "Feature distributions rendered."
    except ImportError:
        return f"Matplotlib not installed. Skipping feature distributions."


def plot_classification_comparison(
    benchmark_records: List[Dict[str, Any]],
    title: str = "Classification Models Performance Comparison",
    save_path: Optional[str] = None
) -> str:
    """
    Render grouped bar chart comparing performance metrics across all models.
    """
    try:
        import matplotlib.pyplot as plt
        models = [r["Model"] for r in benchmark_records]
        metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
        colors = ["#2bcbba", "#45aaf2", "#4b7bec", "#a55eea", "#fd9644"]

        n_models = len(models)
        n_metrics = len(metrics)
        bar_width = 0.15

        fig, ax = plt.subplots(figsize=(10, 5.5))

        for m_idx, metric in enumerate(metrics):
            values = [r.get(metric, 0.0) for r in benchmark_records]
            positions = [i + m_idx * bar_width for i in range(n_models)]
            ax.bar(positions, values, width=bar_width, label=metric, color=colors[m_idx % len(colors)], alpha=0.85, edgecolor="black", linewidth=0.5)

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel("Score [0.0 - 1.0]", fontsize=11)
        ax.set_ylim([0.0, 1.1])
        ax.set_xticks([i + bar_width * (n_metrics - 1) / 2.0 for i in range(n_models)])
        ax.set_xticklabels(models, fontsize=10, fontweight="bold")
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=200)
            plt.close()
            return f"Classification comparison saved to {save_path}"
        else:
            plt.close()
            return "Classification comparison rendered."
    except ImportError:
        lines = [f"=== {title} ==="]
        for r in benchmark_records:
            lines.append(f"{r['Model']}: Acc={r.get('Accuracy')}, F1={r.get('F1')}, AUC={r.get('ROC-AUC')}")
        return "\n".join(lines)


def plot_model_complexity_curves(
    complexity_dict: Dict[str, Any],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> str:
    """
    Render train vs validation evaluation curves across hyperparameter complexity.
    Highlights optimal trade-off point between underfitting and overfitting.
    """
    model_name = complexity_dict.get("model_name", "Model")
    param_name = complexity_dict.get("param_name", "Complexity Parameter")
    param_values = complexity_dict.get("param_values", [])
    train_scores = complexity_dict.get("train_scores", [])
    val_scores = complexity_dict.get("val_scores", [])
    best_param = complexity_dict.get("best_param")
    best_val = complexity_dict.get("best_val_score")

    plot_title = title or f"{model_name}: Underfitting vs Overfitting Analysis"
    param_labels = {
        "n_neighbors": "Number of Neighbors (k)",
        "max_depth": "Maximum Tree Depth (max_depth)",
        "l2_reg": "L2 Ridge Penalty (lambda)",
    }
    display_param = param_labels.get(param_name, param_name.replace("_", " ").title())
    metric_label = "Holdout Accuracy" if "Classifier" in model_name else "Holdout Determination (R²)"

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8.5, 5))
        ax.plot(
            param_values,
            train_scores,
            "o-",
            color="#2980b9",
            linewidth=2.0,
            label="Training Score",
        )
        ax.plot(
            param_values,
            val_scores,
            "s-",
            color="#e67e22",
            linewidth=2.0,
            label="Validation Score",
        )

        if best_param is not None and best_val is not None:
            ax.axvline(
                x=best_param,
                color="#27ae60",
                linestyle="--",
                alpha=0.8,
                label=f"Optimal {display_param}={best_param} ({metric_label}={best_val:.3f})",
            )
            ax.scatter([best_param], [best_val], color="#27ae60", s=100, zorder=5)

        ax.set_title(plot_title, fontsize=13, fontweight="bold")
        ax.set_xlabel(display_param, fontsize=11)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=200)
            plt.close()
            return f"Complexity curves saved to {save_path}"
        else:
            plt.close()
            return "Complexity curves rendered."
    except ImportError:
        lines = [f"=== {plot_title} ==="]
        for p, tr, va in zip(param_values, train_scores, val_scores):
            lines.append(f"{param_name}={p}: Train={tr:.4f}, Val={va:.4f}")
        return "\n".join(lines)

