"""
Reproducible experiment runner for EduPredict.

Executes regression and classification benchmark pipelines using the unified
leak-free MLPipeline and Scikit-learn estimators, computing evaluation metrics
from scratch and generating diagnostic visualizations.
"""

import os
import time
from typing import Dict, Any, Optional, List

from edupredict.data.generator import generate_student_data
from edupredict.data.loader import load_csv, save_csv
from edupredict.data.dataset import Dataset
from edupredict.data.schema import infer_schema, DatasetSchema
from edupredict.pipeline.pipeline import MLPipeline, ExperimentSuite
from edupredict.math.stats import correlation_matrix
from edupredict.visualization.plots import (
    plot_confusion_matrix_heatmap,
    plot_actual_vs_predicted,
    plot_roc_curves,
    plot_correlation_heatmap,
    plot_feature_distributions,
    plot_classification_comparison,
)
from edupredict.metrics.classification import roc_curve
from edupredict.experiments.benchmark import BenchmarkReport


def run_full_experiment(
    csv_path: Optional[str] = None,
    output_dir: str = "artifacts",
    random_seed: int = 42,
    target_column: Optional[str] = None,
    task_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run complete ML experimental workflow:
    1. Load dataset or generate reproducible synthetic student data.
    2. Infer schema (or use user-specified target/task).
    3. Benchmark candidate models using leak-free MLPipeline.
    4. Compute all metrics using scratch-implemented formulas.
    5. Generate diagnostic visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"=== EduPredict: Starting Classical ML Experiment (Seed={random_seed}) ===")

    # Step 1: Load or generate dataset
    if csv_path and os.path.exists(csv_path):
        print(f"[*] Loading dataset from: {csv_path}")
        headers, rows = load_csv(csv_path)
        dataset = Dataset(headers=headers, data=rows)
    else:
        print("[*] Generating synthetic student academic performance dataset...")
        dataset = generate_student_data(n_samples=600, random_seed=random_seed, missing_rate=0.02)
        default_csv = "data/student_performance.csv"
        os.makedirs("data", exist_ok=True)
        save_csv(default_csv, dataset.headers, dataset.rows)
        print(f"[*] Saved dataset to {default_csv}")

    print(f"[*] Dataset loaded successfully: {dataset.shape[0]} rows, {dataset.shape[1]} columns")
    print(dataset.summary())

    results: Dict[str, Any] = {
        "dataset_shape": dataset.shape,
        "regression": None,
        "classification": None,
    }

    # Detect multi-task educational dataset vs arbitrary tabular dataset
    is_student_benchmark = (
        "final_exam_score" in dataset.headers and "academic_risk" in dataset.headers
        and target_column is None
    )

    if is_student_benchmark:
        # Benchmark both educational tasks
        # 1. Regression Task: final_exam_score
        print("\n" + "=" * 60)
        print(" TASK 1: Final Exam Score Prediction (Regression)")
        print("=" * 60)

        schema_reg = infer_schema(
            dataset.headers, dataset.rows,
            target_column="final_exam_score",
            task_type="regression",
            ignored_columns=["student_id", "academic_risk"],
        )
        print(f"[*] Regression Schema: {len(schema_reg.numerical_features)} numeric, {len(schema_reg.categorical_features)} categorical features")

        raw_X_reg, y_reg = dataset.split_features_target(
            feature_cols=schema_reg.feature_names,
            target_col="final_exam_score",
        )
        y_reg = [float(v) for v in y_reg]

        suite_reg = ExperimentSuite(schema_reg, raw_X_reg, y_reg, random_seed=random_seed)
        report_reg = suite_reg.run_all()
        report_reg.print_table()
        results["regression"] = suite_reg.results

        # Diagnostic plot: Actual vs Predicted for best regression model
        best_reg_pipe = suite_reg.pipelines.get("Linear Regression") or list(suite_reg.pipelines.values())[0]
        p_act_vs_pred = os.path.join(output_dir, "actual_vs_predicted_scores.png")
        r2_val = best_reg_pipe.test_metrics.get("r2", 0.0)
        plot_actual_vs_predicted(
            best_reg_pipe.y_test,
            best_reg_pipe.y_test_pred,
            title=f"Actual vs. Predicted Final Exam Scores (Linear Regression R² = {r2_val:.4f})",
            save_path=p_act_vs_pred,
        )

        # 2. Classification Task: academic_risk
        print("\n" + "=" * 60)
        print(" TASK 2: Early Academic Risk Detection (Classification)")
        print("=" * 60)

        schema_clf = infer_schema(
            dataset.headers, dataset.rows,
            target_column="academic_risk",
            task_type="binary_classification",
            ignored_columns=["student_id", "final_exam_score"],
        )
        print(f"[*] Classification Schema: {len(schema_clf.numerical_features)} numeric, {len(schema_clf.categorical_features)} categorical features")

        raw_X_clf, y_clf = dataset.split_features_target(
            feature_cols=schema_clf.feature_names,
            target_col="academic_risk",
        )
        y_clf = [int(v) for v in y_clf]

        suite_clf = ExperimentSuite(schema_clf, raw_X_clf, y_clf, random_seed=random_seed)
        report_clf = suite_clf.run_all()
        report_clf.print_table()
        results["classification"] = suite_clf.results

        # Diagnostic plots: Confusion Matrix and ROC Curves
        best_clf_pipe = suite_clf.pipelines.get("Logistic Regression") or list(suite_clf.pipelines.values())[0]
        p_cm = os.path.join(output_dir, "confusion_matrix.png")
        plot_confusion_matrix_heatmap(
            best_clf_pipe.test_metrics["confusion_matrix"],
            labels=["Normal (0)", "At-Risk (1)"],
            title="Confusion Matrix - Logistic Regression",
            save_path=p_cm,
        )

        # ROC Curves for models supporting predict_proba
        roc_data = {}
        for m_name, pipe in suite_clf.pipelines.items():
            if pipe.y_test_proba is not None and pipe.test_metrics.get("roc_auc") is not None:
                fpr, tpr, _ = roc_curve(pipe.y_test, pipe.y_test_proba)
                auc_val = pipe.test_metrics["roc_auc"]
                roc_data[m_name] = (fpr, tpr, auc_val)

        if roc_data:
            p_roc = os.path.join(output_dir, "roc_curves.png")
            plot_roc_curves(roc_data, save_path=p_roc)

        # Classification comparison bar chart
        clf_records_for_plot = [
            {
                "Model": m_name,
                "Accuracy": res["metrics"]["accuracy"],
                "Precision": res["metrics"]["precision"],
                "Recall": res["metrics"]["recall"],
                "F1": res["metrics"]["f1"],
                "ROC-AUC": res["metrics"].get("roc_auc") or 0.0,
            }
            for m_name, res in suite_clf.results.items()
        ]
        p_comp = os.path.join(output_dir, "classification_comparison.png")
        plot_classification_comparison(clf_records_for_plot, save_path=p_comp)

        # Dataset feature distributions & correlation heatmap
        num_cols = schema_reg.numerical_features
        if num_cols:
            from edupredict.preprocessing.imputer import SimpleImputer
            raw_num_matrix = [[row[dataset.header_to_idx[c]] for c in num_cols] for row in dataset.rows]
            clean_num_matrix = SimpleImputer(strategy="mean").fit_transform(raw_num_matrix)
            num_data = {c: [clean_num_matrix[r][i] for r in range(len(clean_num_matrix))] for i, c in enumerate(num_cols)}

            p_dist = os.path.join(output_dir, "feature_distributions.png")
            plot_feature_distributions(num_data, save_path=p_dist)

            # Correlation matrix from scratch
            col_matrices = [num_data[c] for c in num_cols]
            corr = correlation_matrix(col_matrices)
            p_corr = os.path.join(output_dir, "correlation_heatmap.png")
            plot_correlation_heatmap(corr, num_cols, save_path=p_corr)

    else:
        # Arbitrary single-target dataset
        schema = infer_schema(
            dataset.headers, dataset.rows,
            target_column=target_column,
            task_type=task_type,
        )
        print(f"[*] Inferred Schema for '{schema.target_column}' ({schema.task_type}):")
        print(f"    Numerical: {schema.numerical_features}")
        print(f"    Categorical: {schema.categorical_features}")
        print(f"    Ignored: {schema.ignored_columns}")

        raw_X, y = dataset.split_features_target(
            feature_cols=schema.feature_names,
            target_col=schema.target_column,
        )
        if "classification" in schema.task_type:
            y = [int(v) if str(v).isdigit() else str(v) for v in y]
        else:
            y = [float(v) for v in y]

        suite = ExperimentSuite(schema, raw_X, y, random_seed=random_seed)
        report = suite.run_all()
        report.print_table()
        task_key = "classification" if "classification" in schema.task_type else "regression"
        results[task_key] = suite.results

    print("\n[*] All experiments and diagnostics completed successfully.")
    return results
