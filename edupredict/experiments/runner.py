"""
Reproducible end-to-end experiment runner for EduPredict.

Executes both student performance regression and academic risk classification pipelines,
benchmarking all implemented classical models from scratch.
"""

import os
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
from edupredict.metrics.classification import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from edupredict.visualization.plots import (
    plot_loss_curve,
    plot_confusion_matrix_heatmap,
    plot_feature_importances,
    plot_actual_vs_predicted,
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
    4. Feature scaling
    5. Regression benchmarking (Linear Regression vs KNN Regressor)
    6. Classification benchmarking (Logistic Regression vs KNN vs Decision Tree vs Naive Bayes)
    7. Diagnostic visual reporting
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

    # Step 2: Separate features and targets
    # Feature columns: numeric and categorical
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
    y_reg = dataset.get_column("final_exam_score")
    y_clf = dataset.get_column("academic_risk")

    # Step 3: Handle missing values via SimpleImputer
    # For numeric (first 4 cols), impute mean; for categorical (next 3 cols), impute mode
    num_cols_idx = [0, 1, 2, 3]
    cat_cols_idx = [4, 5, 6]

    num_X = [[row[i] for i in num_cols_idx] for row in raw_X]
    cat_X = [[row[i] for i in cat_cols_idx] for row in raw_X]

    num_imputer = SimpleImputer(strategy="mean")
    cat_imputer = SimpleImputer(strategy="mode")

    num_X_clean = num_imputer.fit_transform(num_X)
    cat_X_clean = cat_imputer.fit_transform(cat_X)

    # Step 4: Encode categorical features
    ohe = OneHotEncoder(drop_first=False)
    cat_X_encoded = ohe.fit_transform(cat_X_clean)

    # Combine numeric and encoded categorical features
    combined_X: list[list[float]] = []
    for r in range(len(num_X_clean)):
        combined_X.append([float(v) for v in num_X_clean[r]] + [float(v) for v in cat_X_encoded[r]])

    # Build feature column names list
    encoded_feature_names = ["study_hours", "attendance", "past_score", "assignments_completed"]
    for col_idx, cats in enumerate(ohe.categories_):
        cat_name = feature_cols[cat_cols_idx[col_idx]]
        for cat in cats:
            encoded_feature_names.append(f"{cat_name}_{cat}")

    # Step 5: Feature scaling
    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(combined_X)

    # Step 6: Train/Test Split (80/20)
    X_train, X_test, y_reg_train, y_reg_test = train_test_split(
        scaled_X, y_reg, test_size=0.2, random_seed=random_seed
    )
    _, _, y_clf_train, y_clf_test = train_test_split(
        scaled_X, y_clf, test_size=0.2, random_seed=random_seed, stratify=True
    )

    print(f"\n[*] Split completed: {len(X_train)} training samples, {len(X_test)} testing samples.")

    # ==========================================
    # REGRESSION BENCHMARK: Predicting Final Score
    # ==========================================
    print("\n" + "=" * 50)
    print(">>> 1. REGRESSION: PREDICTING FINAL EXAM SCORE")
    print("=" * 50)
    reg_report = BenchmarkReport(task_name="Student Final Exam Score Prediction (Regression)")

    # 1A. Linear Regression via Gradient Descent
    lr_gd = LinearRegression(solver="gd", learning_rate=0.05, max_epochs=600, random_seed=random_seed)
    lr_gd.fit(X_train, y_reg_train)
    preds_lr_gd = lr_gd.predict(X_test)
    reg_report.add_result("Linear Regression (GD)", {
        "MSE": mean_squared_error(y_reg_test, preds_lr_gd),
        "RMSE": root_mean_squared_error(y_reg_test, preds_lr_gd),
        "MAE": mean_absolute_error(y_reg_test, preds_lr_gd),
        "R2": r2_score(y_reg_test, preds_lr_gd),
    })

    # Save GD Loss Curve
    plot_loss_curve(
        lr_gd.loss_history_,
        title="Linear Regression (GD) MSE Loss Curve",
        save_path=os.path.join(output_dir, "linear_regression_loss.png")
    )

    # 1B. Linear Regression via Normal Equation
    lr_ne = LinearRegression(solver="normal_equation", l2_penalty=1e-4)
    lr_ne.fit(X_train, y_reg_train)
    preds_lr_ne = lr_ne.predict(X_test)
    reg_report.add_result("Linear Regression (Normal Eq)", {
        "MSE": mean_squared_error(y_reg_test, preds_lr_ne),
        "RMSE": root_mean_squared_error(y_reg_test, preds_lr_ne),
        "MAE": mean_absolute_error(y_reg_test, preds_lr_ne),
        "R2": r2_score(y_reg_test, preds_lr_ne),
    })

    # 1C. KNN Regressor
    knn_reg = KNNRegressor(n_neighbors=7, weights="distance")
    knn_reg.fit(X_train, y_reg_train)
    preds_knn = knn_reg.predict(X_test)
    reg_report.add_result("KNN Regressor (k=7)", {
        "MSE": mean_squared_error(y_reg_test, preds_knn),
        "RMSE": root_mean_squared_error(y_reg_test, preds_knn),
        "MAE": mean_absolute_error(y_reg_test, preds_knn),
        "R2": r2_score(y_reg_test, preds_knn),
    })

    # Actual vs Predicted plot for best regressor
    plot_actual_vs_predicted(
        y_reg_test,
        preds_lr_ne,
        title="Actual vs Predicted Exam Scores (Normal Eq)",
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

    # 2A. Logistic Regression
    log_reg = LogisticRegression(learning_rate=0.1, max_epochs=500, random_seed=random_seed)
    log_reg.fit(X_train, y_clf_train)
    preds_log_reg = log_reg.predict(X_test)
    clf_report.add_result("Logistic Regression", {
        "Accuracy": accuracy_score(y_clf_test, preds_log_reg),
        "Precision": precision_score(y_clf_test, preds_log_reg, pos_label=1),
        "Recall": recall_score(y_clf_test, preds_log_reg, pos_label=1),
        "F1": f1_score(y_clf_test, preds_log_reg, pos_label=1),
    })

    plot_loss_curve(
        log_reg.loss_history_,
        title="Logistic Regression Binary Cross-Entropy Loss Curve",
        save_path=os.path.join(output_dir, "logistic_regression_loss.png")
    )

    # 2B. KNN Classifier
    knn_clf = KNNClassifier(n_neighbors=5, weights="distance")
    knn_clf.fit(X_train, y_clf_train)
    preds_knn_clf = knn_clf.predict(X_test)
    clf_report.add_result("KNN Classifier (k=5)", {
        "Accuracy": accuracy_score(y_clf_test, preds_knn_clf),
        "Precision": precision_score(y_clf_test, preds_knn_clf, pos_label=1),
        "Recall": recall_score(y_clf_test, preds_knn_clf, pos_label=1),
        "F1": f1_score(y_clf_test, preds_knn_clf, pos_label=1),
    })

    # 2C. Decision Tree Classifier
    dt_clf = DecisionTreeClassifier(criterion="gini", max_depth=5, min_samples_split=4)
    dt_clf.fit(X_train, y_clf_train)
    preds_dt = dt_clf.predict(X_test)
    clf_report.add_result("Decision Tree (Depth=5)", {
        "Accuracy": accuracy_score(y_clf_test, preds_dt),
        "Precision": precision_score(y_clf_test, preds_dt, pos_label=1),
        "Recall": recall_score(y_clf_test, preds_dt, pos_label=1),
        "F1": f1_score(y_clf_test, preds_dt, pos_label=1),
    })

    # Plot Decision Tree Feature Importances
    plot_feature_importances(
        encoded_feature_names,
        dt_clf.feature_importances_,
        title="Decision Tree Feature Importances (Academic Risk)",
        save_path=os.path.join(output_dir, "feature_importances.png")
    )

    # 2D. Gaussian Naive Bayes
    gnb = GaussianNaiveBayes()
    gnb.fit(X_train, y_clf_train)
    preds_gnb = gnb.predict(X_test)
    clf_report.add_result("Gaussian Naive Bayes", {
        "Accuracy": accuracy_score(y_clf_test, preds_gnb),
        "Precision": precision_score(y_clf_test, preds_gnb, pos_label=1),
        "Recall": recall_score(y_clf_test, preds_gnb, pos_label=1),
        "F1": f1_score(y_clf_test, preds_gnb, pos_label=1),
    })

    # Plot Confusion Matrix for Best Classifier (e.g. Logistic Regression)
    cm = confusion_matrix(y_clf_test, preds_log_reg, labels=[0, 1])
    plot_confusion_matrix_heatmap(
        cm,
        labels=["0: Normal", "1: At-Risk"],
        title="Confusion Matrix - Logistic Regression",
        save_path=os.path.join(output_dir, "confusion_matrix.png")
    )

    clf_report.print_table()

    print("\n[*] Experiment pipeline completed successfully!")
    print(f"[*] Visual artifacts saved to '{output_dir}/'")

    return {
        "regression_benchmark": reg_report.records,
        "classification_benchmark": clf_report.records,
        "n_samples": len(scaled_X),
        "n_features": len(scaled_X[0]),
    }
