"""
EduPredict: Interactive Web Dashboard (Streamlit).

A clean, production-ready Classical Machine Learning interface powered by
Scikit-learn estimators and scratch-implemented linear algebra, statistics,
leakage-free preprocessing, loss functions, and evaluation metrics.
"""

import os
import sys
import time
from typing import Dict, Any, List, Optional

import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from edupredict.data.loader import load_csv, save_csv, validate_dataset
from edupredict.data.generator import generate_student_data
from edupredict.data.dataset import Dataset
from edupredict.data.schema import infer_schema, DatasetSchema
from edupredict.pipeline.pipeline import MLPipeline, ExperimentSuite
from edupredict.math.stats import correlation_matrix
from edupredict.visualization.plots import (
    plot_actual_vs_predicted,
    plot_confusion_matrix_heatmap,
    plot_roc_curves,
    plot_correlation_heatmap,
    plot_feature_distributions,
    plot_classification_comparison,
)
from edupredict.metrics.classification import roc_curve

st.set_page_config(
    page_title="EduPredict | Classical ML Framework",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1e3799;
        margin-bottom: 0.3rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4b6584;
        margin-bottom: 1.5rem;
    }
    .metric-badge {
        background-color: #f1f2f6;
        border-left: 4px solid #1e3799;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading dataset...")
def get_default_dataset() -> Dataset:
    """Load or generate the benchmark student academic performance dataset."""
    csv_path = os.path.join(PROJECT_ROOT, "data", "student_performance.csv")
    if not os.path.exists(csv_path):
        os.makedirs(os.path.join(PROJECT_ROOT, "data"), exist_ok=True)
        ds = generate_student_data(n_samples=600, random_seed=42, missing_rate=0.02)
        save_csv(csv_path, ds.headers, ds.rows)
        return ds
    headers, rows = load_csv(csv_path)
    return Dataset(headers=headers, data=rows)


# Sidebar
with st.sidebar:
    st.markdown("## 🎓 EduPredict")
    st.caption("Classical Machine Learning Framework")
    st.markdown("---")

    nav_choice = st.radio(
        "Navigation",
        [
            "📊 Dataset Overview & Validation",
            "📈 Regression Studio",
            "🎯 Classification Studio",
            "📋 Empirical Benchmark Suite",
            "📖 Architecture & Methods",
        ],
    )

    st.markdown("---")
    st.markdown("**Architecture Highlights:**")
    st.markdown("- **ML Estimators:** Scikit-learn")
    st.markdown("- **Linear Algebra:** Custom Scratch Engine")
    st.markdown("- **Statistics:** Custom Scratch Engine")
    st.markdown("- **Preprocessing:** Leak-Free Custom Fit")
    st.markdown("- **Losses & Gradients:** Analytical Scratch")
    st.markdown("- **Evaluation Metrics:** Scratch Formulations")
    st.markdown("- **Data Handling:** Standard Python (No Pandas)")


# Load Dataset
dataset = get_default_dataset()


# Helper to render dynamic prediction widgets based on schema
def render_dynamic_inputs(schema: DatasetSchema, ds: Dataset, key_prefix: str) -> Dict[str, Any]:
    """Generate form inputs dynamically based on schema feature types and data ranges."""
    input_values: Dict[str, Any] = {}

    st.markdown("##### Feature Values")
    col_a, col_b = st.columns(2)

    half = (len(schema.feature_names) + 1) // 2
    cols_left = schema.feature_names[:half]
    cols_right = schema.feature_names[half:]

    def render_col_field(feature_name: str, target_container):
        with target_container:
            col_vals = [row[ds.header_to_idx[feature_name]] for row in ds.rows if row[ds.header_to_idx[feature_name]] is not None]
            if feature_name in schema.numerical_features:
                num_vals = [float(v) for v in col_vals if isinstance(v, (int, float)) or str(v).replace('.', '', 1).isdigit()]
                if num_vals:
                    min_v = float(min(num_vals))
                    max_v = float(max(num_vals))
                    mean_v = float(sum(num_vals) / len(num_vals))
                    step = 0.5 if (max_v - min_v) <= 50 else 1.0
                    input_values[feature_name] = st.slider(
                        f"{feature_name.replace('_', ' ').title()}:",
                        min_value=round(min_v, 1),
                        max_value=round(max_v, 1),
                        value=round(mean_v, 1),
                        step=step,
                        key=f"{key_prefix}_{feature_name}",
                    )
                else:
                    input_values[feature_name] = st.number_input(
                        f"{feature_name.replace('_', ' ').title()}:",
                        value=0.0,
                        key=f"{key_prefix}_{feature_name}",
                    )
            else:
                cat_options = sorted(list(set(str(v) for v in col_vals)))
                if not cat_options:
                    cat_options = ["Standard"]
                input_values[feature_name] = st.selectbox(
                    f"{feature_name.replace('_', ' ').title()}:",
                    options=cat_options,
                    index=0,
                    key=f"{key_prefix}_{feature_name}",
                )

    for f in cols_left:
        render_col_field(f, col_a)
    for f in cols_right:
        render_col_field(f, col_b)

    return input_values


# ---------------------------------------------------------
# TAB 1: Dataset Overview & Validation
# ---------------------------------------------------------
if nav_choice == "📊 Dataset Overview & Validation":
    st.markdown("<div class='main-title'>📊 Dataset Overview & Validation</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Inspect tabular data structure, feature distributions, missing value counts, and integrity checks.</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", dataset.shape[0])
    c2.metric("Total Columns", dataset.shape[1])
    missing_dict = dataset.count_missing()
    total_missing = sum(missing_dict.values())
    c3.metric("Total Missing Values", total_missing)

    st.markdown("### 📋 Tabular Records Preview (First 8 Rows)")
    table_headers = dataset.headers
    table_rows = dataset.rows[:8]
    st.table([{h: row[i] for i, h in enumerate(table_headers)} for row in table_rows])

    st.markdown("### 🔍 Schema Validation & Missing Values by Feature")
    validation_report = validate_dataset(dataset.headers, dataset.rows)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**Column Missing Counts:**")
        missing_data = [{"Column": col, "Missing Count": count, "Missing Rate": f"{(count / len(dataset) * 100):.2f}%"} for col, count in missing_dict.items()]
        st.dataframe(missing_data, use_container_width=True)

    with col2:
        st.markdown("**Dataset Integrity Status:**")
        if validation_report.is_valid:
            st.success("✅ Dataset schema is valid: row lengths match headers, no malformed structural errors.")
        else:
            st.error(f"❌ Validation errors detected: {validation_report.errors}")
        st.caption("Note: Missing values (~2% rate) are intentionally present to validate our leak-free SimpleImputer.")

    st.markdown("---")
    st.markdown("### 📈 Numerical Feature Distributions & Correlation Matrix")
    c_p1, c_p2 = st.columns(2)
    artifacts_dir = os.path.join(PROJECT_ROOT, "artifacts")
    
    with c_p1:
        p_dist = os.path.join(artifacts_dir, "feature_distributions.png")
        if os.path.exists(p_dist):
            st.image(p_dist, caption="Feature Distributions (Computed and plotted from scratch data)")
    with c_p2:
        p_corr = os.path.join(artifacts_dir, "correlation_heatmap.png")
        if os.path.exists(p_corr):
            st.image(p_corr, caption="Pearson Correlation Heatmap (Computed using scratch correlation matrix)")


# ---------------------------------------------------------
# TAB 2: Regression Studio
# ---------------------------------------------------------
elif nav_choice == "📈 Regression Studio":
    st.markdown("<div class='main-title'>📈 Regression Studio</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Train, tune, and evaluate Scikit-learn continuous estimators using custom preprocessing and evaluation metrics.</div>", unsafe_allow_html=True)

    schema_reg = infer_schema(
        dataset.headers, dataset.rows,
        target_column="final_exam_score",
        task_type="regression",
        ignored_columns=["student_id", "academic_risk"],
    )

    col_config, col_results = st.columns([1, 1], gap="large")

    with col_config:
        st.subheader("⚙️ Model Configuration")
        model_name = st.selectbox(
            "Regression Algorithm:",
            ["Linear Regression", "Ridge Regression", "KNN Regressor", "Decision Tree Regressor"],
            key="reg_model_select"
        )

        model_params: Dict[str, Any] = {}
        if model_name == "Linear Regression":
            fit_intercept = st.checkbox("Fit Intercept (Bias)", value=True)
            model_params["fit_intercept"] = fit_intercept
        elif model_name == "Ridge Regression":
            alpha = st.slider("L2 Regularization Strength (alpha):", 0.01, 20.0, 1.0, 0.1)
            model_params["alpha"] = alpha
        elif model_name == "KNN Regressor":
            k = st.slider("Number of Neighbors (k):", 1, 25, 5, 1)
            weights = st.selectbox("Weighting:", ["uniform", "distance"])
            model_params["n_neighbors"] = k
            model_params["weights"] = weights
        elif model_name == "Decision Tree Regressor":
            max_depth = st.slider("Max Tree Depth:", 1, 15, 5, 1)
            min_samples_split = st.slider("Min Samples Split:", 2, 20, 2, 1)
            model_params["max_depth"] = max_depth
            model_params["min_samples_split"] = min_samples_split

        scaler_type = st.selectbox("Feature Scaler:", ["standard", "minmax"], index=0, key="reg_scaler")
        test_size = st.slider("Test Set Size:", 0.1, 0.4, 0.2, 0.05, key="reg_test_size")

        train_btn = st.button("🚀 Train & Evaluate Regression Model", type="primary", use_container_width=True)

    with col_results:
        st.subheader("📊 Live Evaluation Metrics")

        # Train pipeline if button pressed or if not yet in state
        if train_btn or "reg_pipeline" not in st.session_state or st.session_state.get("reg_model_trained") != model_name:
            raw_X, y = dataset.split_features_target(
                feature_cols=schema_reg.feature_names,
                target_col="final_exam_score"
            )
            y = [float(v) for v in y]

            pipe = MLPipeline(
                schema=schema_reg,
                model_name=model_name,
                model_params=model_params,
                scaler_type=scaler_type,
                random_seed=42,
            )
            pipe.fit(raw_X, y, test_size=test_size)
            st.session_state["reg_pipeline"] = pipe
            st.session_state["reg_model_trained"] = model_name

        active_pipe: MLPipeline = st.session_state["reg_pipeline"]
        metrics = active_pipe.test_metrics

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test R² Score", f"{metrics['r2']:.4f}")
        c2.metric("Test MSE", f"{metrics['mse']:.4f}")
        c3.metric("Test RMSE", f"{metrics['rmse']:.4f}")
        c4.metric("Test MAE", f"{metrics['mae']:.4f}")

        st.caption(f"⏱️ Training Time: **{active_pipe.training_time_ms:.2f} ms** | Test Inference: **{active_pipe.inference_time_ms:.2f} ms**")

        # Plot Actual vs Predicted
        temp_fig_path = os.path.join(PROJECT_ROOT, "artifacts", "temp_reg_plot.png")
        plot_actual_vs_predicted(
            active_pipe.y_test,
            active_pipe.y_test_pred,
            title=f"Actual vs. Predicted ({model_name} | R² = {metrics['r2']:.4f})",
            save_path=temp_fig_path
        )
        if os.path.exists(temp_fig_path):
            st.image(temp_fig_path, use_container_width=True)

    st.markdown("---")
    st.subheader("🔮 Single Student Continuous Score Prediction")
    st.markdown("Adjust student indicators below to predict their final score through the fitted pipeline:")

    reg_inputs = render_dynamic_inputs(schema_reg, dataset, key_prefix="reg")
    if reg_inputs:
        pred_score = active_pipe.predict_single(reg_inputs)
        bounded_score = max(0.0, min(100.0, float(pred_score)))

        st.markdown(
            f"<div class='metric-badge'>"
            f"<h4>Predicted Final Exam Score: <span style='color: #1e3799;'>{bounded_score:.2f} / 100</span></h4>"
            f"Processed through fitted SimpleImputer, OneHotEncoder, StandardScaler, and {model_name}."
            f"</div>",
            unsafe_allow_html=True
        )


# ---------------------------------------------------------
# TAB 3: Classification Studio
# ---------------------------------------------------------
elif nav_choice == "🎯 Classification Studio":
    st.markdown("<div class='main-title'>🎯 Classification Studio</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Train and evaluate Scikit-learn classifiers for academic risk detection using scratch preprocessing, splitting, and metrics.</div>", unsafe_allow_html=True)

    schema_clf = infer_schema(
        dataset.headers, dataset.rows,
        target_column="academic_risk",
        task_type="binary_classification",
        ignored_columns=["student_id", "final_exam_score"],
    )

    col_config, col_results = st.columns([1, 1], gap="large")

    with col_config:
        st.subheader("⚙️ Classifier Configuration")
        model_name = st.selectbox(
            "Classification Algorithm:",
            ["Logistic Regression", "KNN Classifier", "Decision Tree Classifier", "Gaussian Naive Bayes"],
            key="clf_model_select"
        )

        model_params: Dict[str, Any] = {}
        if model_name == "Logistic Regression":
            max_iter = st.slider("Max Iterations:", 100, 2000, 1000, 100)
            c_val = st.slider("Inverse Regularization (C):", 0.01, 10.0, 1.0, 0.1)
            model_params["max_iter"] = max_iter
            model_params["C"] = c_val
        elif model_name == "KNN Classifier":
            k = st.slider("Number of Neighbors (k):", 1, 25, 5, 1, key="clf_k")
            weights = st.selectbox("Weighting:", ["uniform", "distance"], key="clf_weights")
            model_params["n_neighbors"] = k
            model_params["weights"] = weights
        elif model_name == "Decision Tree Classifier":
            criterion = st.selectbox("Split Criterion:", ["gini", "entropy"])
            max_depth = st.slider("Max Tree Depth:", 1, 15, 5, 1, key="clf_depth")
            min_samples_split = st.slider("Min Samples Split:", 2, 20, 2, 1, key="clf_split")
            model_params["criterion"] = criterion
            model_params["max_depth"] = max_depth
            model_params["min_samples_split"] = min_samples_split
        elif model_name == "Gaussian Naive Bayes":
            var_smoothing = st.select_slider("Variance Smoothing:", options=[1e-11, 1e-9, 1e-7, 1e-5], value=1e-9)
            model_params["var_smoothing"] = var_smoothing

        scaler_type = st.selectbox("Feature Scaler:", ["standard", "minmax"], index=0, key="clf_scaler")
        test_size = st.slider("Test Set Size:", 0.1, 0.4, 0.2, 0.05, key="clf_test_size")

        train_btn = st.button("🚀 Train & Evaluate Classifier", type="primary", use_container_width=True)

    with col_results:
        st.subheader("📊 Live Evaluation Metrics")

        # Train pipeline if button pressed or if not yet in state
        if train_btn or "clf_pipeline" not in st.session_state or st.session_state.get("clf_model_trained") != model_name:
            raw_X, y = dataset.split_features_target(
                feature_cols=schema_clf.feature_names,
                target_col="academic_risk"
            )
            y = [int(v) for v in y]

            pipe = MLPipeline(
                schema=schema_clf,
                model_name=model_name,
                model_params=model_params,
                scaler_type=scaler_type,
                random_seed=42,
            )
            pipe.fit(raw_X, y, test_size=test_size)
            st.session_state["clf_pipeline"] = pipe
            st.session_state["clf_model_trained"] = model_name

        active_pipe: MLPipeline = st.session_state["clf_pipeline"]
        metrics = active_pipe.test_metrics

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
        c2.metric("Precision", f"{metrics['precision']:.4f}")
        c3.metric("Recall", f"{metrics['recall']:.4f}")
        c4.metric("F1-Score", f"{metrics['f1']:.4f}")

        if metrics.get("roc_auc") is not None:
            st.metric("ROC-AUC Score", f"{metrics['roc_auc']:.4f}")

        st.caption(f"⏱️ Training Time: **{active_pipe.training_time_ms:.2f} ms** | Test Inference: **{active_pipe.inference_time_ms:.2f} ms**")

        # Plot Confusion Matrix
        temp_cm_path = os.path.join(PROJECT_ROOT, "artifacts", "temp_cm_plot.png")
        plot_confusion_matrix_heatmap(
            metrics["confusion_matrix"],
            labels=["Normal (0)", "At-Risk (1)"],
            title=f"Confusion Matrix ({model_name})",
            save_path=temp_cm_path
        )
        if os.path.exists(temp_cm_path):
            st.image(temp_cm_path, use_container_width=True)

    st.markdown("---")
    st.subheader("🔮 Single Student Academic Risk Assessment")
    st.markdown("Adjust student indicators below to assess academic risk probability:")

    clf_inputs = render_dynamic_inputs(schema_clf, dataset, key_prefix="clf")
    if clf_inputs:
        pred_label = active_pipe.predict_single(clf_inputs)
        pred_proba = active_pipe.predict_single_proba(clf_inputs)

        risk_str = "🚨 At-Risk (Needs Academic Support)" if pred_label == 1 else "✅ On-Track (Good Academic Standing)"
        risk_color = "#eb4d4b" if pred_label == 1 else "#20bf6b"

        prob_msg = ""
        if pred_proba is not None:
            risk_p = pred_proba.get(1, pred_proba.get("1", 0.0))
            prob_msg = f"<p>Estimated Risk Probability: <b>{risk_p * 100:.1f}%</b></p>"

        st.markdown(
            f"<div class='metric-badge' style='border-left-color: {risk_color};'>"
            f"<h4>Risk Assessment: <span style='color: {risk_color};'>{risk_str}</span></h4>"
            f"{prob_msg}"
            f"</div>",
            unsafe_allow_html=True
        )


# ---------------------------------------------------------
# TAB 4: Empirical Benchmark Suite
# ---------------------------------------------------------
elif nav_choice == "📋 Empirical Benchmark Suite":
    st.markdown("<div class='main-title'>📋 Empirical Benchmark Suite</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Live, empirically computed benchmark comparisons across all supported Classical ML models.</div>", unsafe_allow_html=True)

    st.info("ℹ️ All results below are computed in real time by executing the leak-free pipeline on the current dataset.")

    if st.button("⚡ Run Full Benchmark Suite Live", type="primary"):
        with st.spinner("Executing all regression and classification experiments..."):
            # Task 1: Regression
            schema_reg = infer_schema(dataset.headers, dataset.rows, target_column="final_exam_score", task_type="regression", ignored_columns=["student_id", "academic_risk"])
            raw_X_reg, y_reg = dataset.split_features_target(feature_cols=schema_reg.feature_names, target_col="final_exam_score")
            y_reg = [float(v) for v in y_reg]
            suite_reg = ExperimentSuite(schema_reg, raw_X_reg, y_reg)
            rep_reg = suite_reg.run_all()

            # Task 2: Classification
            schema_clf = infer_schema(dataset.headers, dataset.rows, target_column="academic_risk", task_type="binary_classification", ignored_columns=["student_id", "final_exam_score"])
            raw_X_clf, y_clf = dataset.split_features_target(feature_cols=schema_clf.feature_names, target_col="academic_risk")
            y_clf = [int(v) for v in y_clf]
            suite_clf = ExperimentSuite(schema_clf, raw_X_clf, y_clf)
            rep_clf = suite_clf.run_all()

            st.session_state["benchmark_reg"] = rep_reg.records
            st.session_state["benchmark_clf"] = rep_clf.records

    if "benchmark_reg" in st.session_state and "benchmark_clf" in st.session_state:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 1. Final Exam Score Regression")
            st.table(st.session_state["benchmark_reg"])
        with c2:
            st.markdown("### 2. Academic Risk Classification")
            st.table(st.session_state["benchmark_clf"])

        st.markdown("---")
        st.markdown("### 📊 Diagnostic Visualizations (Generated from Run)")
        artifacts_dir = os.path.join(PROJECT_ROOT, "artifacts")
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            p_comp = os.path.join(artifacts_dir, "classification_comparison.png")
            if os.path.exists(p_comp):
                st.image(p_comp, caption="Comparative Classification Metric Profiles")
        with col_v2:
            p_roc = os.path.join(artifacts_dir, "roc_curves.png")
            if os.path.exists(p_roc):
                st.image(p_roc, caption="ROC Curves with Scratch Trapezoidal AUC Integration")
    else:
        st.write("Click the button above to run the live benchmark suite.")


# ---------------------------------------------------------
# TAB 5: Architecture & Methods
# ---------------------------------------------------------
elif nav_choice == "📖 Architecture & Methods":
    st.markdown("<div class='main-title'>📖 Architecture & Mathematical Methods</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>In-depth documentation of mathematical foundations, leak-free pipeline boundaries, and Scikit-learn estimator integration.</div>", unsafe_allow_html=True)

    with st.expander("📐 Mathematical Foundations & Linear Algebra", expanded=True):
        st.markdown(r"""
        EduPredict implements core linear algebra primitives in `edupredict/math/`:
        - **Vector Operations (`vectors.py`):** Dot products ($\mathbf{u} \cdot \mathbf{v} = \sum u_i v_i$), Euclidean ($\|\mathbf{u}-\mathbf{v}\|_2$) and Manhattan ($\|\mathbf{u}-\mathbf{v}\|_1$) distances.
        - **Matrix Operations (`matrices.py`):** Matrix multiplication, transpositions, and **Gauss-Jordan elimination with partial pivoting** for matrix inversion.
        - **Descriptive Statistics (`stats.py`):** Sample variance, standard deviation, covariance, and Pearson correlation matrices.
        - **Activations (`activations.py`):** Numerically safe sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$ and softmax with log-sum-exp normalization.
        """)

    with st.expander("🛡️ Data Leakage Prevention Architecture", expanded=True):
        st.markdown(r"""
        Data leakage occurs when test data statistics inadvertently contaminate the training process. EduPredict guarantees strict isolation:
        1. **Split-First Protocol:** `train_test_split` partitions raw samples before any transformer is initialized.
        2. **Train-Only Parameter Fitting:**
           - `SimpleImputer` learns feature means/modes strictly from $X_{\text{train}}$.
           - `OneHotEncoder` learns categorical category sets strictly from $X_{\text{train}}$. Unseen test categories are mapped to zero-vectors.
           - `StandardScaler` computes $\mu_{\text{train}}$ and $\sigma_{\text{train}}$ strictly from $X_{\text{train}}$.
        3. **Identical Sample Indices:** Multi-task targets are split jointly using shared indices to prevent row misalignment.
        """)

    with st.expander("🎯 Evaluation Metrics Implemented from First Principles", expanded=True):
        st.markdown(r"""
        All metrics reported by EduPredict are implemented from scratch in `edupredict/metrics/`:
        - **Regression:**
          - Mean Squared Error: $\text{MSE} = \frac{1}{m}\sum (\hat{y}_i - y_i)^2$
          - Root Mean Squared Error: $\text{RMSE} = \sqrt{\text{MSE}}$
          - Mean Absolute Error: $\text{MAE} = \frac{1}{m}\sum |\hat{y}_i - y_i|$
          - Coefficient of Determination: $R^2 = 1 - \frac{\sum (\hat{y}_i - y_i)^2}{\sum (y_i - \bar{y})^2}$
        - **Classification:**
          - $\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$
          - $\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$, $\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$, $F_1 = 2 \cdot \frac{P \cdot R}{P + R}$
          - **ROC-AUC Integration:** Evaluates dynamic decision thresholds and integrates under the curve using the trapezoidal rule:
            $$\text{AUC} = \sum_{i=1}^{k} \frac{\text{TPR}_i + \text{TPR}_{i-1}}{2} \times (\text{FPR}_i - \text{FPR}_{i-1})$$
        """)
