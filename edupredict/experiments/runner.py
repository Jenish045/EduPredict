"""
Reproducible experiment runner for EduPredict.

Executes regression and classification benchmark pipelines using the unified
leak-free MLPipeline and custom from-scratch estimators (Linear Regression,
KNN Classifier, Decision Tree Classifier), computing evaluation metrics
from first principles and generating diagnostic visualizations and overfitting curves.
"""

import os
import time
from typing import Dict, Any, Optional, List

from edupredict.data.generator import generate_student_data
from edupredict.data.loader import load_csv, save_csv
from edupredict.data.dataset import Dataset
from edupredict.data.schema import infer_schema, DatasetSchema
from edupredict.model_selection.split import train_test_split
from edupredict.pipeline.pipeline import MLPipeline, ExperimentSuite
from edupredict.math.stats import correlation_matrix
from edupredict.visualization.plots import (
    plot_loss_curve,
    plot_confusion_matrix_heatmap,
    plot_actual_vs_predicted,
    plot_roc_curves,
    plot_correlation_heatmap,
    plot_feature_distributions,
    plot_classification_comparison,
    plot_model_complexity_curves,
)
from edupredict.metrics.classification import roc_curve
from edupredict.experiments.benchmark import BenchmarkReport
from edupredict.analysis.overfitting import (
    analyze_knn_complexity,
    analyze_decision_tree_complexity,
    analyze_linear_regression_regularization,
)


def run_full_experiment(
    csv_path: Optional[str] = None,
    output_dir: str = "artifacts",
    random_seed: int = 42,
    target_column: Optional[str] = None,
    task_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run complete ML experimental workflow:
    1. Load primary UCI Student Performance dataset (or synthetic fallback).
    2. Infer schema with explicit data-leakage safeguards (excluding intermediate grades).
    3. Benchmark candidate models using leak-free MLPipeline.
    4. Compute all metrics using scratch-implemented formulas.
    5. Perform underfitting and overfitting complexity analysis.
    6. Generate diagnostic visualizations and charts.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"=== EduPredict: Starting Classical ML Experiment (Seed={random_seed}) ===")

    # Step 1: Resolve dataset path
    resolved_path = csv_path or "data/student-mat.csv"

    if os.path.exists(resolved_path):
        print(f"[*] Loading dataset from: {resolved_path}")
        headers, rows = load_csv(resolved_path)
        dataset = Dataset(headers=headers, data=rows)
    else:
        print("[*] Generating synthetic student academic performance dataset in-memory...")
        dataset = generate_student_data(n_samples=600, random_seed=random_seed, missing_rate=0.02)

    print(f"[*] Dataset loaded successfully: {dataset.shape[0]} rows, {dataset.shape[1]} columns")

    results: Dict[str, Any] = {
        "dataset_shape": dataset.shape,
        "regression": None,
        "classification": None,
        "overfitting_analysis": {},
    }

    # Case A: UCI Student Performance Dataset (with G3)
    is_uci_mat = "G3" in dataset.headers and target_column is None

    # Case B: Synthetic Benchmark Dataset (with final_exam_score and academic_risk)
    is_synthetic_benchmark = (
        "final_exam_score" in dataset.headers and "academic_risk" in dataset.headers
        and target_column is None
    )

    if is_uci_mat:
        # Augment with derived academic_risk column if not already present
        # Risk = 1 if G3 < 10 (failing threshold in Portuguese secondary school), else 0
        g3_idx = dataset.header_to_idx["G3"]
        if "academic_risk" not in dataset.headers:
            new_headers = list(dataset.headers) + ["academic_risk"]
            new_rows = []
            for row in dataset.rows:
                g3_val = row[g3_idx]
                risk = 1 if (g3_val is not None and float(g3_val) < 10.0) else 0
                new_rows.append(list(row) + [risk])
            dataset = Dataset(headers=new_headers, data=new_rows)
            print("[*] Created academic_risk target: Risk=1 (G3 < 10), Normal=0 (G3 >= 10)")

        # -------------------------------------------------------------
        # TASK 1: Final Math Grade Prediction (G3) - Regression
        # -------------------------------------------------------------
        print("\n" + "=" * 60)
        print(" TASK 1: Final Mathematics Grade Prediction (G3 Regression)")
        print(" [Leakage Prevention: Excluding G1 and G2 from feature set]")
        print("=" * 60)

        schema_reg = infer_schema(
            dataset.headers, dataset.rows,
            target_column="G3",
            task_type="regression",
            ignored_columns=["G1", "G2", "academic_risk"],
        )
        print(f"[*] Regression Schema: {len(schema_reg.numerical_features)} numeric, {len(schema_reg.categorical_features)} categorical features")

        raw_X_reg, y_reg = dataset.split_features_target(
            feature_cols=schema_reg.feature_names,
            target_col="G3",
        )
        y_reg = [float(v) for v in y_reg]

        suite_reg = ExperimentSuite(
            schema_reg, raw_X_reg, y_reg,
            custom_params={
                "Linear Regression": {
                    "solver": "gradient_descent",
                    "l2_reg": 5.0,
                    "max_epochs": 500,
                    "learning_rate": 0.01,
                    "random_seed": random_seed,
                }
            },
            random_seed=random_seed,
        )
        report_reg = suite_reg.run_all()

        # Add Training-Mean Baseline to benchmark table
        best_reg_pipe = suite_reg.pipelines.get("Linear Regression")
        if best_reg_pipe is not None:
            from edupredict.metrics.regression import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score
            mean_y_tr = sum(best_reg_pipe.y_train) / len(best_reg_pipe.y_train)
            dummy_reg_preds = [mean_y_tr] * len(best_reg_pipe.y_test)
            report_reg.add_result(
                "Training-Mean Baseline",
                {
                    "MSE": mean_squared_error(best_reg_pipe.y_test, dummy_reg_preds),
                    "RMSE": root_mean_squared_error(best_reg_pipe.y_test, dummy_reg_preds),
                    "MAE": mean_absolute_error(best_reg_pipe.y_test, dummy_reg_preds),
                    "R2": r2_score(best_reg_pipe.y_test, dummy_reg_preds),
                    "TRAIN_TIME_MS": 0.0,
                    "INFER_TIME_MS": 0.0,
                }
            )
        report_reg.print_table()
        results["regression"] = suite_reg.results

        # Diagnostic plot: Actual vs Predicted
        best_reg_pipe = suite_reg.pipelines.get("Linear Regression")
        if best_reg_pipe is not None:
            p_act_vs_pred = os.path.join(output_dir, "actual_vs_predicted_scores.png")
            r2_val = best_reg_pipe.test_metrics.get("r2", 0.0)
            plot_actual_vs_predicted(
                best_reg_pipe.y_test,
                best_reg_pipe.y_test_pred,
                title=f"Actual vs. Predicted Final Mathematics Grade (G3) — Linear Regression (R² = {r2_val:.4f})",
                save_path=p_act_vs_pred,
            )

            # Training Loss Curve
            if hasattr(best_reg_pipe.estimator, "loss_history_") and best_reg_pipe.estimator.loss_history_:
                p_loss = os.path.join(output_dir, "linear_regression_loss_curve.png")
                plot_loss_curve(
                    best_reg_pipe.estimator.loss_history_,
                    title="Linear Regression Gradient Descent Loss Convergence",
                    save_path=p_loss,
                )

        # -------------------------------------------------------------
        # TASK 2: Academic Risk Detection (Classification)
        # -------------------------------------------------------------
        print("\n" + "=" * 60)
        print(" TASK 2: Academic Risk Classification (KNN & Decision Tree)")
        print(" [Leakage Prevention: Excluding G1, G2, and G3 from feature set]")
        print("=" * 60)

        schema_clf = infer_schema(
            dataset.headers, dataset.rows,
            target_column="academic_risk",
            task_type="binary_classification",
            ignored_columns=["G1", "G2", "G3"],
        )
        print(f"[*] Classification Schema: {len(schema_clf.numerical_features)} numeric, {len(schema_clf.categorical_features)} categorical features")

        raw_X_clf, y_clf = dataset.split_features_target(
            feature_cols=schema_clf.feature_names,
            target_col="academic_risk",
        )
        y_clf = [int(v) for v in y_clf]

        suite_clf = ExperimentSuite(
            schema_clf, raw_X_clf, y_clf,
            custom_params={
                "Decision Tree Classifier": {
                    "max_depth": 2,
                    "min_samples_split": 2,
                    "random_seed": random_seed,
                },
                "KNN Classifier": {
                    "n_neighbors": 5,
                    "weights": "uniform",
                },
            },
            random_seed=random_seed,
        )
        report_clf = suite_clf.run_all()

        # Add Majority-Class Baseline to benchmark table
        best_clf_name = "Decision Tree Classifier" if "Decision Tree Classifier" in suite_clf.pipelines else list(suite_clf.pipelines.keys())[0]
        ref_clf_pipe = suite_clf.pipelines[best_clf_name]
        dummy_clf_preds = [0] * len(ref_clf_pipe.y_test)
        from edupredict.metrics.classification import accuracy_score, precision_score, recall_score, f1_score
        report_clf.add_result(
            "Majority-Class Baseline",
            {
                "ACCURACY": accuracy_score(ref_clf_pipe.y_test, dummy_clf_preds),
                "PRECISION": precision_score(ref_clf_pipe.y_test, dummy_clf_preds, pos_label=1, zero_division=0.0),
                "RECALL": recall_score(ref_clf_pipe.y_test, dummy_clf_preds, pos_label=1, zero_division=0.0),
                "F1": f1_score(ref_clf_pipe.y_test, dummy_clf_preds, pos_label=1, zero_division=0.0),
                "ROC_AUC": 0.5000,
                "TRAIN_TIME_MS": 0.0,
                "INFER_TIME_MS": 0.0,
            }
        )
        report_clf.print_table()
        results["classification"] = suite_clf.results

        # Confusion Matrix for best classification model (Decision Tree)
        best_clf_pipe = suite_clf.pipelines[best_clf_name]
        p_cm = os.path.join(output_dir, "confusion_matrix.png")
        plot_confusion_matrix_heatmap(
            best_clf_pipe.test_metrics["confusion_matrix"],
            labels=["Normal (0)", "At-Risk (1)"],
            title=f"Confusion Matrix - {best_clf_name}",
            save_path=p_cm,
        )

        # ROC Curves
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

        # -------------------------------------------------------------
        # OVERFITTING & UNDERFITTING COMPLEXITY ANALYSIS
        # -------------------------------------------------------------
        print("\n" + "=" * 60)
        print(" OVERFITTING & UNDERFITTING COMPLEXITY ANALYSIS")
        print(" [Validation Set Protocol: Hyperparameters tuned strictly on Train-Sub / Val]")
        print(" [Holdout Test Set (79 samples) remains completely isolated and untouched]")
        print("=" * 60)

        # To prevent data leakage and preserve test set integrity, split the training pool (316 samples)
        # into Train-Sub (80% = 253 samples) and Validation (20% = 63 samples)
        X_tr_sc = best_clf_pipe.X_train_scaled
        y_tr = best_clf_pipe.y_train

        X_tr_sub, X_val_sc, y_tr_sub, y_val = train_test_split(
            X_tr_sc, y_tr, test_size=0.20, stratify=y_tr, random_seed=random_seed
        )

        # 1. KNN Complexity Curve: Varying k
        knn_analysis = analyze_knn_complexity(X_tr_sub, y_tr_sub, X_val_sc, y_val)
        print(f"[*] {knn_analysis['summary']}")
        p_knn_curve = os.path.join(output_dir, "knn_overfitting_curve.png")
        plot_model_complexity_curves(knn_analysis, save_path=p_knn_curve)
        results["overfitting_analysis"]["knn"] = knn_analysis

        # 2. Decision Tree Complexity Curve: Varying max_depth
        dt_analysis = analyze_decision_tree_complexity(X_tr_sub, y_tr_sub, X_val_sc, y_val)
        print(f"[*] {dt_analysis['summary']}")
        p_dt_curve = os.path.join(output_dir, "tree_overfitting_curve.png")
        plot_model_complexity_curves(dt_analysis, save_path=p_dt_curve)
        results["overfitting_analysis"]["decision_tree"] = dt_analysis

        # 3. Linear Regression Regularization Curve: Varying l2_reg
        if best_reg_pipe is not None:
            reg_tr_sc = best_reg_pipe.X_train_scaled
            reg_y_tr = best_reg_pipe.y_train
            reg_tr_sub, reg_val_sc, reg_y_tr_sub, reg_y_val = train_test_split(
                reg_tr_sc, reg_y_tr, test_size=0.20, random_seed=random_seed
            )
            lr_analysis = analyze_linear_regression_regularization(
                reg_tr_sub, reg_y_tr_sub, reg_val_sc, reg_y_val
            )
            print(f"[*] {lr_analysis['summary']}")
            p_lr_curve = os.path.join(output_dir, "linear_regression_regularization_curve.png")
            plot_model_complexity_curves(lr_analysis, save_path=p_lr_curve)
            results["overfitting_analysis"]["linear_regression"] = lr_analysis

        # Correlation Heatmap for numerical features
        num_cols = schema_reg.numerical_features
        if num_cols:
            from edupredict.preprocessing.imputer import SimpleImputer
            raw_num_matrix = [[row[dataset.header_to_idx[c]] for c in num_cols] for row in dataset.rows]
            clean_num_matrix = SimpleImputer(strategy="mean").fit_transform(raw_num_matrix)
            num_data = {c: [clean_num_matrix[r][i] for r in range(len(clean_num_matrix))] for i, c in enumerate(num_cols)}

            p_dist = os.path.join(output_dir, "feature_distributions.png")
            plot_feature_distributions(num_data, save_path=p_dist)

            col_matrices = [num_data[c] for c in num_cols]
            corr = correlation_matrix(col_matrices)
            p_corr = os.path.join(output_dir, "correlation_heatmap.png")
            plot_correlation_heatmap(corr, num_cols, save_path=p_corr)

    elif is_synthetic_benchmark:
        # Benchmark both educational tasks on synthetic dataset
        print("\n" + "=" * 60)
        print(" TASK 1: Final Exam Score Prediction (Regression)")
        print("=" * 60)

        schema_reg = infer_schema(
            dataset.headers, dataset.rows,
            target_column="final_exam_score",
            task_type="regression",
            ignored_columns=["student_id", "academic_risk"],
        )
        raw_X_reg, y_reg = dataset.split_features_target(
            feature_cols=schema_reg.feature_names,
            target_col="final_exam_score",
        )
        y_reg = [float(v) for v in y_reg]

        suite_reg = ExperimentSuite(
            schema_reg, raw_X_reg, y_reg,
            custom_params={
                "Linear Regression": {
                    "solver": "gradient_descent",
                    "l2_reg": 5.0,
                    "max_epochs": 500,
                    "learning_rate": 0.01,
                    "random_seed": random_seed,
                }
            },
            random_seed=random_seed,
        )
        report_reg = suite_reg.run_all()

        # Add Training-Mean Baseline to benchmark table
        best_reg_pipe = suite_reg.pipelines.get("Linear Regression")
        if best_reg_pipe is not None:
            from edupredict.metrics.regression import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score
            mean_y_tr = sum(best_reg_pipe.y_train) / len(best_reg_pipe.y_train)
            dummy_reg_preds = [mean_y_tr] * len(best_reg_pipe.y_test)
            report_reg.add_result(
                "Training-Mean Baseline",
                {
                    "MSE": mean_squared_error(best_reg_pipe.y_test, dummy_reg_preds),
                    "RMSE": root_mean_squared_error(best_reg_pipe.y_test, dummy_reg_preds),
                    "MAE": mean_absolute_error(best_reg_pipe.y_test, dummy_reg_preds),
                    "R2": r2_score(best_reg_pipe.y_test, dummy_reg_preds),
                    "TRAIN_TIME_MS": 0.0,
                    "INFER_TIME_MS": 0.0,
                }
            )
        report_reg.print_table()
        results["regression"] = suite_reg.results

        best_reg_pipe = suite_reg.pipelines.get("Linear Regression")
        if best_reg_pipe:
            p_act_vs_pred = os.path.join(output_dir, "actual_vs_predicted_scores.png")
            r2_val = best_reg_pipe.test_metrics.get("r2", 0.0)
            plot_actual_vs_predicted(
                best_reg_pipe.y_test,
                best_reg_pipe.y_test_pred,
                title=f"Actual vs. Predicted Final Exam Scores (Linear Regression R² = {r2_val:.4f})",
                save_path=p_act_vs_pred,
            )

        print("\n" + "=" * 60)
        print(" TASK 2: Early Academic Risk Detection (Classification)")
        print("=" * 60)

        schema_clf = infer_schema(
            dataset.headers, dataset.rows,
            target_column="academic_risk",
            task_type="binary_classification",
            ignored_columns=["student_id", "final_exam_score"],
        )
        raw_X_clf, y_clf = dataset.split_features_target(
            feature_cols=schema_clf.feature_names,
            target_col="academic_risk",
        )
        y_clf = [int(v) for v in y_clf]

        suite_clf = ExperimentSuite(
            schema_clf, raw_X_clf, y_clf,
            custom_params={
                "Decision Tree Classifier": {
                    "max_depth": 2,
                    "min_samples_split": 2,
                    "random_seed": random_seed,
                },
                "KNN Classifier": {
                    "n_neighbors": 5,
                    "weights": "uniform",
                },
            },
            random_seed=random_seed,
        )
        report_clf = suite_clf.run_all()
        report_clf.print_table()
        results["classification"] = suite_clf.results

    else:
        # Arbitrary single-target dataset
        schema = infer_schema(
            dataset.headers, dataset.rows,
            target_column=target_column,
            task_type=task_type,
        )
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
