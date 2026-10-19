"""
Reproducible end-to-end experiment runner for EduPredict.

Executes both student performance regression and academic risk classification pipelines,
benchmarking all four implemented classical models from scratch:
1. Linear Regression (Regression)
2. Logistic Regression (Classification)
3. K-Nearest Neighbors (Classification & Regression)
4. Decision Tree (Classification)
and Gaussian Naive Bayes baseline.
"""

import os
import time
from typing import Dict, Any, Optional

from edupredict.data.generator import generate_student_data
from edupredict.data.loader import load_csv, save_csv
from edupredict.data.dataset import Dataset
from edupredict.preprocessing.imputer import SimpleImputer
from edupredict.preprocessing.encoder import OneHotEncoder
from edupredict.preprocessing.scaler import StandardScaler
from edupredict.model_selection.split import train_test_split
from edupredict.models.linear_regression import LinearRegression
from edupredict.models.logistic_regression import LogisticRegression
from edupredict.models.knn import KNNClassifier, KNNRegressor
from edupredict.models.decision_tree import DecisionTreeClassifier
from edupredict.models.naive_bayes import GaussianNaiveBayes
from edupredict.metrics.regression import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score
from edupredict.metrics.classification import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)
from edupredict.math.stats import correlation_matrix
from edupredict.visualization.plots import (
    plot_loss_curve,
    plot_confusion_matrix_heatmap,
    plot_feature_importances,
    plot_actual_vs_predicted,
    plot_roc_curves,
    plot_correlation_heatmap,
    plot_feature_distributions,
    plot_classification_comparison,
)
from edupredict.experiments.benchmark import BenchmarkReport


def run_full_experiment(
    csv_path: Optional[str] = None,
    output_dir: str = "artifacts",
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Run complete ML experimental workflow:
    1. Ingestion / Synthetic Generation
    2. Missing value handling (Imputation)
    3. Categorical encoding
    4. Feature scaling (fit on train only!)
    5. Regression benchmarking (Linear Regression vs KNN Regressor)
    6. Classification benchmarking (Logistic Regression vs KNN vs Decision Tree vs Naive Bayes)
    7. Diagnostic visual reporting (Loss curves, confusion matrix, ROC curves)
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"=== EduPredict: Starting Classical ML Experiment (Seed={random_seed}) ===")

    # Step 1: Load or generate dataset
    if csv_path and os.path.exists(csv_path):
        print(f"[*] Loading dataset from: {csv_path}")
        headers, rows = load_csv(csv_path)
        dataset = Dataset(headers=headers, data=rows, target_col="final_exam_score")
    else:
        print("[*] Generating synthetic student academic performance dataset...")
        dataset = generate_student_data(n_samples=600, random_seed=random_seed, missing_rate=0.02)
        default_csv = "data/student_performance.csv"
        os.makedirs("data", exist_ok=True)
        save_csv(default_csv, dataset.headers, dataset.rows)
        print(f"[*] Saved dataset to {default_csv}")

    print(f"[*] Dataset loaded successfully: {dataset.shape[0]} rows, {dataset.shape[1]} columns")
    print(dataset.summary())

    # Step 2: Separate features and targets (Zero Target Leakage)
    feature_cols = [
        "study_hours",
        "attendance",
        "past_score",
        "assignments_completed",
        "parental_education",
        "internet_access",
        "tutoring",
    ]
    raw_X = [[dataset.get_column(col)[r] for col in feature_cols] for r in range(len(dataset))]
    y_reg = [float(v) for v in dataset.get_column("final_exam_score")]
    y_clf = [int(v) for v in dataset.get_column("academic_risk")]

    # Step 3: Split raw records and targets TOGETHER before ANY preprocessing
    # Uses a single unified split mechanism with shared indices to strictly preserve sample alignment!
    # Stratified by y_clf so training and testing partitions maintain identical class proportions.
    raw_X_train, raw_X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        raw_X, y_reg, y_clf,
        test_size=0.2,
        random_seed=random_seed,
        stratify=y_clf
    )

    num_cols_idx = [0, 1, 2, 3]
    cat_cols_idx = [4, 5, 6]

    num_X_train = [[row[i] for i in num_cols_idx] for row in raw_X_train]
    cat_X_train = [[row[i] for i in cat_cols_idx] for row in raw_X_train]

    num_X_test = [[row[i] for i in num_cols_idx] for row in raw_X_test]
    cat_X_test = [[row[i] for i in cat_cols_idx] for row in raw_X_test]

    # Step 4: Fit numerical and categorical imputers on TRAINING FEATURES ONLY
    num_imputer = SimpleImputer(strategy="mean")
    cat_imputer = SimpleImputer(strategy="mode")

    num_X_train_clean = num_imputer.fit_transform(num_X_train)
    cat_X_train_clean = cat_imputer.fit_transform(cat_X_train)

    # Transform test partition with frozen training imputer parameters (NO LEAKAGE)
    num_X_test_clean = num_imputer.transform(num_X_test)
    cat_X_test_clean = cat_imputer.transform(cat_X_test)

    # Diagnostic EDA Visualizations: Feature distributions & Pearson correlation heatmap
    # Rendered strictly on the training partition to respect the training boundary
    num_feature_names = ["study_hours", "attendance", "past_score", "assignments_completed"]
    features_dict = {
        name: [num_X_train_clean[r][i] for r in range(len(num_X_train_clean))]
        for i, name in enumerate(num_feature_names)
    }
    plot_feature_distributions(
        features_dict,
        title="Student Numerical Feature Distributions (Training Split)",
        save_path=os.path.join(output_dir, "feature_distributions.png")
    )

    corr_cols = [features_dict[name] for name in num_feature_names] + [y_reg_train]
    corr_names = num_feature_names + ["final_score"]
    corr_mat = correlation_matrix(corr_cols)
    plot_correlation_heatmap(
        corr_mat,
        corr_names,
        title="Feature & Target Pearson Correlation Matrix (Training Split)",
        save_path=os.path.join(output_dir, "correlation_heatmap.png")
    )

    # Step 5: Fit OneHotEncoder on TRAINING CATEGORICAL FEATURES ONLY
    ohe = OneHotEncoder(drop_first=False)
    cat_X_train_encoded = ohe.fit_transform(cat_X_train_clean)
    cat_X_test_encoded = ohe.transform(cat_X_test_clean)

    # Combine numeric and encoded categorical features for train and test
    combined_X_train: list[list[float]] = []
    for r in range(len(num_X_train_clean)):
        combined_X_train.append([float(v) for v in num_X_train_clean[r]] + [float(v) for v in cat_X_train_encoded[r]])

    combined_X_test: list[list[float]] = []
    for r in range(len(num_X_test_clean)):
        combined_X_test.append([float(v) for v in num_X_test_clean[r]] + [float(v) for v in cat_X_test_encoded[r]])

    # Build feature column names list from encoder fitted on train
    encoded_feature_names = ["study_hours", "attendance", "past_score", "assignments_completed"]
    for col_idx, cats in enumerate(ohe.categories_):
        cat_name = feature_cols[cat_cols_idx[col_idx]]
        for cat in cats:
            encoded_feature_names.append(f"{cat_name}_{cat}")

    # Step 6: Fit StandardScaler on TRAINING FEATURES ONLY, then transform test
    scaler = StandardScaler()
    X_train = scaler.fit_transform(combined_X_train)
    X_test = scaler.transform(combined_X_test)  # strictly transform test without fitting

    print(f"\n[*] Split completed: {len(X_train)} training samples, {len(X_test)} testing samples.")
    print(f"[*] Verified zero leakage: Imputer, Encoder & Scaler fit strictly on training partition.")


    # ==========================================
    # REGRESSION BENCHMARK: Predicting Final Score
    # ==========================================
    print("\n" + "=" * 50)
    print(">>> 1. REGRESSION: PREDICTING FINAL EXAM SCORE")
    print("=" * 50)
    reg_report = BenchmarkReport(task_name="Student Final Exam Score Prediction (Regression)")

    # 1A. Linear Regression via Gradient Descent
    t0 = time.perf_counter()
    lr_gd = LinearRegression(solver="gd", learning_rate=0.05, max_epochs=600, random_seed=random_seed)
    lr_gd.fit(X_train, y_reg_train)
    t_train_lr = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    preds_lr_gd = lr_gd.predict(X_test)
    t_infer_lr = (time.perf_counter() - t0) * 1000.0

    reg_report.add_result("Linear Regression (GD)", {
        "MSE": mean_squared_error(y_reg_test, preds_lr_gd),
        "RMSE": root_mean_squared_error(y_reg_test, preds_lr_gd),
        "MAE": mean_absolute_error(y_reg_test, preds_lr_gd),
        "R2": r2_score(y_reg_test, preds_lr_gd),
        "Train(ms)": round(t_train_lr, 2),
        "Infer(ms)": round(t_infer_lr, 2),
    })

    # Save GD Loss Curve
    plot_loss_curve(
        lr_gd.loss_history_,
        title="Linear Regression (GD) MSE Loss Curve",
        save_path=os.path.join(output_dir, "linear_regression_loss.png")
    )

    # 1B. Linear Regression via Normal Equation
    t0 = time.perf_counter()
    lr_ne = LinearRegression(solver="normal_equation", l2_penalty=1e-4)
    lr_ne.fit(X_train, y_reg_train)
    t_train_ne = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    preds_lr_ne = lr_ne.predict(X_test)
    t_infer_ne = (time.perf_counter() - t0) * 1000.0

    reg_report.add_result("Linear Regression (Normal Eq)", {
        "MSE": mean_squared_error(y_reg_test, preds_lr_ne),
        "RMSE": root_mean_squared_error(y_reg_test, preds_lr_ne),
        "MAE": mean_absolute_error(y_reg_test, preds_lr_ne),
        "R2": r2_score(y_reg_test, preds_lr_ne),
        "Train(ms)": round(t_train_ne, 2),
        "Infer(ms)": round(t_infer_ne, 2),
    })

    # 1C. KNN Regressor
    t0 = time.perf_counter()
    knn_reg = KNNRegressor(n_neighbors=7, weights="distance")
    knn_reg.fit(X_train, y_reg_train)
    t_train_knn = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    preds_knn = knn_reg.predict(X_test)
    t_infer_knn = (time.perf_counter() - t0) * 1000.0

    reg_report.add_result("KNN Regressor (k=7)", {
        "MSE": mean_squared_error(y_reg_test, preds_knn),
        "RMSE": root_mean_squared_error(y_reg_test, preds_knn),
        "MAE": mean_absolute_error(y_reg_test, preds_knn),
        "R2": r2_score(y_reg_test, preds_knn),
        "Train(ms)": round(t_train_knn, 2),
        "Infer(ms)": round(t_infer_knn, 2),
    })

    # Actual vs Predicted plot for best regressor
    plot_actual_vs_predicted(
        y_reg_test,
        preds_lr_gd,
        title="Actual vs Predicted Exam Scores (Linear Regression GD)",
        save_path=os.path.join(output_dir, "actual_vs_predicted_scores.png")
    )

    reg_report.print_table()

    # ==========================================
    # CLASSIFICATION BENCHMARK: Academic Risk
    # ==========================================
    print("\n" + "=" * 50)
    print(">>> 2. CLASSIFICATION: PREDICTING ACADEMIC RISK (PASS vs RISK)")
    print("=" * 50)
    clf_report = BenchmarkReport(task_name="Student Academic Risk Prediction (Classification)")
    roc_curves_data: Dict[str, Any] = {}

    # 2A. Logistic Regression
    t0 = time.perf_counter()
    log_reg = LogisticRegression(learning_rate=0.1, max_epochs=500, random_seed=random_seed)
    log_reg.fit(X_train, y_clf_train)
    t_train_log = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    preds_log_reg = log_reg.predict(X_test)
    t_infer_log = (time.perf_counter() - t0) * 1000.0

    probs_log = log_reg.predict_proba(X_test)
    fpr_log, tpr_log, _ = roc_curve(y_clf_test, probs_log, pos_label=1)
    auc_log = roc_auc_score(y_clf_test, probs_log, pos_label=1)
    roc_curves_data["Logistic Regression"] = (fpr_log, tpr_log, auc_log)

    clf_report.add_result("Logistic Regression", {
        "Accuracy": accuracy_score(y_clf_test, preds_log_reg),
        "Precision": precision_score(y_clf_test, preds_log_reg, pos_label=1),
        "Recall": recall_score(y_clf_test, preds_log_reg, pos_label=1),
        "F1": f1_score(y_clf_test, preds_log_reg, pos_label=1),
        "ROC-AUC": auc_log,
        "Train(ms)": round(t_train_log, 2),
        "Infer(ms)": round(t_infer_log, 2),
    })

    plot_loss_curve(
        log_reg.loss_history_,
        title="Logistic Regression Binary Cross-Entropy Loss Curve",
        save_path=os.path.join(output_dir, "logistic_regression_loss.png")
    )

    # 2B. KNN Classifier
    t0 = time.perf_counter()
    knn_clf = KNNClassifier(n_neighbors=5, weights="distance")
    knn_clf.fit(X_train, y_clf_train)
    t_train_knnc = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    preds_knn_clf = knn_clf.predict(X_test)
    t_infer_knnc = (time.perf_counter() - t0) * 1000.0

    probs_knn_raw = knn_clf.predict_proba(X_test)
    probs_knn = [p.get(1, 0.0) for p in probs_knn_raw]
    fpr_knn, tpr_knn, _ = roc_curve(y_clf_test, probs_knn, pos_label=1)
    auc_knn = roc_auc_score(y_clf_test, probs_knn, pos_label=1)
    roc_curves_data["KNN (k=5)"] = (fpr_knn, tpr_knn, auc_knn)

    clf_report.add_result("KNN Classifier (k=5)", {
        "Accuracy": accuracy_score(y_clf_test, preds_knn_clf),
        "Precision": precision_score(y_clf_test, preds_knn_clf, pos_label=1),
        "Recall": recall_score(y_clf_test, preds_knn_clf, pos_label=1),
        "F1": f1_score(y_clf_test, preds_knn_clf, pos_label=1),
        "ROC-AUC": auc_knn,
        "Train(ms)": round(t_train_knnc, 2),
        "Infer(ms)": round(t_infer_knnc, 2),
    })

    # 2C. Decision Tree Classifier
    t0 = time.perf_counter()
    dt_clf = DecisionTreeClassifier(criterion="gini", max_depth=5, min_samples_split=4)
    dt_clf.fit(X_train, y_clf_train)
    t_train_dt = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    preds_dt = dt_clf.predict(X_test)
    t_infer_dt = (time.perf_counter() - t0) * 1000.0

    probs_dt_raw = dt_clf.predict_proba(X_test)
    probs_dt = [p.get(1, 0.0) for p in probs_dt_raw]
    fpr_dt, tpr_dt, _ = roc_curve(y_clf_test, probs_dt, pos_label=1)
    auc_dt = roc_auc_score(y_clf_test, probs_dt, pos_label=1)
    roc_curves_data["Decision Tree"] = (fpr_dt, tpr_dt, auc_dt)

    clf_report.add_result("Decision Tree (Depth=5)", {
        "Accuracy": accuracy_score(y_clf_test, preds_dt),
        "Precision": precision_score(y_clf_test, preds_dt, pos_label=1),
        "Recall": recall_score(y_clf_test, preds_dt, pos_label=1),
        "F1": f1_score(y_clf_test, preds_dt, pos_label=1),
        "ROC-AUC": auc_dt,
        "Train(ms)": round(t_train_dt, 2),
        "Infer(ms)": round(t_infer_dt, 2),
    })

    # Plot Decision Tree Feature Importances
    plot_feature_importances(
        encoded_feature_names,
        dt_clf.feature_importances_,
        title="Decision Tree Feature Importances (Academic Risk)",
        save_path=os.path.join(output_dir, "feature_importances.png")
    )

    # 2D. Gaussian Naive Bayes
    t0 = time.perf_counter()
    gnb = GaussianNaiveBayes()
    gnb.fit(X_train, y_clf_train)
    t_train_gnb = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    preds_gnb = gnb.predict(X_test)
    t_infer_gnb = (time.perf_counter() - t0) * 1000.0

    probs_gnb_raw = gnb.predict_proba(X_test)
    probs_gnb = [p[1] for p in probs_gnb_raw]
    fpr_gnb, tpr_gnb, _ = roc_curve(y_clf_test, probs_gnb, pos_label=1)
    auc_gnb = roc_auc_score(y_clf_test, probs_gnb, pos_label=1)
    roc_curves_data["Naive Bayes"] = (fpr_gnb, tpr_gnb, auc_gnb)

    clf_report.add_result("Gaussian Naive Bayes", {
        "Accuracy": accuracy_score(y_clf_test, preds_gnb),
        "Precision": precision_score(y_clf_test, preds_gnb, pos_label=1),
        "Recall": recall_score(y_clf_test, preds_gnb, pos_label=1),
        "F1": f1_score(y_clf_test, preds_gnb, pos_label=1),
        "ROC-AUC": auc_gnb,
        "Train(ms)": round(t_train_gnb, 2),
        "Infer(ms)": round(t_infer_gnb, 2),
    })

    # Plot Confusion Matrix for Logistic Regression
    cm = confusion_matrix(y_clf_test, preds_log_reg, labels=[0, 1])
    plot_confusion_matrix_heatmap(
        cm,
        labels=["0: Normal", "1: At-Risk"],
        title="Confusion Matrix - Logistic Regression",
        save_path=os.path.join(output_dir, "confusion_matrix.png")
    )

    # Plot ROC Curves for all classifiers
    plot_roc_curves(
        roc_curves_data,
        title="ROC Curves Comparison (Student Academic Risk)",
        save_path=os.path.join(output_dir, "roc_curves.png")
    )

    # Plot Classification Benchmark Comparison Bar Chart
    plot_classification_comparison(
        clf_report.records,
        title="Classification Models Benchmark Comparison",
        save_path=os.path.join(output_dir, "classification_comparison.png")
    )

    clf_report.print_table()

    print("\n[*] Experiment pipeline completed successfully!")
    print(f"[*] Visual artifacts saved to '{output_dir}/'")

    return {
        "regression_benchmark": reg_report.records,
        "classification_benchmark": clf_report.records,
        "n_samples": len(raw_X),
        "n_features": len(X_train[0]),
    }
