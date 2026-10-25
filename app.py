"""
EduPredict: Interactive Web Dashboard (Streamlit).

An educational Classical Machine Learning interface powered by custom estimators
(Linear Regression, KNN Classifier, Decision Tree Classifier) built entirely from scratch
with zero third-party ML dependencies (no Scikit-learn, no NumPy, no Pandas).
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
    plot_loss_curve,
    plot_actual_vs_predicted,
    plot_confusion_matrix_heatmap,
    plot_roc_curves,
    plot_correlation_heatmap,
    plot_feature_distributions,
    plot_classification_comparison,
    plot_model_complexity_curves,
)
from edupredict.metrics.classification import roc_curve
from edupredict.analysis.overfitting import (
    analyze_knn_complexity,
    analyze_decision_tree_complexity,
    analyze_linear_regression_regularization,
)
from edupredict.data.uci_metadata import (
    UCI_FEATURE_CONFIG,
    get_uci_feature_config,
    get_column_display_name,
    format_preview_records,
)

st.set_page_config(
    page_title="EduPredict | Classical ML Framework",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
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
    """Load the UCI Student Performance Mathematics dataset (data/student-mat.csv)."""
    uci_path = os.path.join(PROJECT_ROOT, "data", "student-mat.csv")

    if os.path.exists(uci_path):
        headers, rows = load_csv(uci_path)
        ds = Dataset(headers=headers, data=rows)
        # Augment with academic_risk target if G3 is present: Risk = 1 if G3 < 10 else 0
        if "G3" in ds.headers and "academic_risk" not in ds.headers:
            g3_idx = ds.header_to_idx["G3"]
            new_headers = list(ds.headers) + ["academic_risk"]
            new_rows = []
            for row in ds.rows:
                g3_val = row[g3_idx]
                risk = 1 if (g3_val is not None and float(g3_val) < 10.0) else 0
                new_rows.append(list(row) + [risk])
            return Dataset(headers=new_headers, data=new_rows)
        return ds

    # In-memory fallback if CSV is absent
    return generate_student_data(n_samples=600, random_seed=42, missing_rate=0.02)


# Load Dataset
dataset = get_default_dataset()
is_uci = "G3" in dataset.headers

# Sidebar Navigation
with st.sidebar:
    st.markdown("## 🎓 EduPredict")
    st.caption("Classical ML Implemented From Scratch")
    st.markdown("---")

    nav_choice = st.radio(
        "Navigation",
        [
            "📊 Dataset Overview & Validation",
            "📈 Regression Studio (Linear Regression)",
            "🎯 Classification Studio (KNN & Decision Tree)",
            "🔬 Overfitting & Underfitting Explorer",
            "📋 Empirical Benchmark Suite",
            "📖 Architecture & Methods",
        ],
    )

    st.markdown("---")
    st.markdown("**Dataset Provenance:**")
    if is_uci:
        st.info("📁 Primary Dataset: UCI Student Performance (Mathematics, 395 records, 33 attributes)")
    else:
        st.info("📁 Auxiliary Dataset: Synthetic Student Performance")

    st.markdown("**Implementation Policy:**")
    st.markdown("- **Core ML Models:** 100% Custom Scratch")
    st.markdown("- **Linear Algebra:** Custom Matrix Engine")
    st.markdown("- **Statistics:** Custom Scratch Stats")
    st.markdown("- **Preprocessing:** Leak-Free Transformers")
    st.markdown("- **Metrics:** Scratch Formulations")
    st.markdown("- **External ML Libs:** ZERO (No NumPy, No Pandas, No Sklearn)")


def render_dynamic_inputs(schema: DatasetSchema, ds: Dataset, key_prefix: str) -> Dict[str, Any]:
    """Generate form inputs with precise display labels, coded descriptions, and controls for UCI features."""
    input_values: Dict[str, Any] = {}
    st.markdown("##### Student Profile Attributes")
    col_a, col_b = st.columns(2)

    # Strictly exclude target and data-leakage columns from student input forms
    forbidden_features = {"G1", "G2", "G3", "academic_risk", "final_exam_score", "student_id"}
    valid_feature_names = [f for f in schema.feature_names if f not in forbidden_features]

    half = (len(valid_feature_names) + 1) // 2
    cols_left = valid_feature_names[:half]
    cols_right = valid_feature_names[half:]

    def render_col_field(feature_name: str, target_container):
        with target_container:
            # 1. UCI Configured Feature: Exact display label, control type, and formatted descriptions
            if feature_name in UCI_FEATURE_CONFIG:
                cfg = UCI_FEATURE_CONFIG[feature_name]
                label_text = f"{cfg['label']}:"
                help_text = cfg.get("help")
                control_type = cfg.get("control")

                if control_type == "select":
                    options = cfg["options"]
                    default_val = cfg.get("default", options[0])
                    idx = options.index(default_val) if default_val in options else 0
                    format_fn = cfg.get("format_func", str)
                    input_values[feature_name] = st.selectbox(
                        label_text,
                        options=options,
                        index=idx,
                        format_func=format_fn,
                        help=help_text,
                        key=f"{key_prefix}_{feature_name}",
                    )
                elif control_type == "slider":
                    min_val = int(cfg.get("min", 15))
                    max_val = int(cfg.get("max", 22))
                    default_val = int(cfg.get("default", min_val))
                    step_val = int(cfg.get("step", 1))
                    input_values[feature_name] = int(st.slider(
                        label_text,
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        step=step_val,
                        help=help_text,
                        key=f"{key_prefix}_{feature_name}",
                    ))
                elif control_type == "number":
                    min_val = int(cfg.get("min", 0))
                    max_val = int(cfg.get("max", 150))
                    default_val = int(cfg.get("default", 0))
                    step_val = int(cfg.get("step", 1))
                    input_values[feature_name] = int(st.number_input(
                        label_text,
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        step=step_val,
                        help=help_text,
                        key=f"{key_prefix}_{feature_name}",
                    ))
                return

            # 2. Generic Fallback for non-UCI / synthetic dataset features
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
    st.markdown("<div class='sub-title'>UCI Student Performance Mathematics Dataset structure, integrity checks, and descriptive statistics.</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Student Records", dataset.shape[0])
    c2.metric("Total Attributes", dataset.shape[1])
    missing_dict = dataset.count_missing()
    total_missing = sum(missing_dict.values())
    c3.metric("Missing Values", total_missing)

    st.markdown("### 📋 Tabular Records Preview (First 8 Rows)")
    table_headers = dataset.headers
    table_rows = dataset.rows[:8]
    st.table(format_preview_records(table_headers, table_rows))

    st.markdown("### 🔍 Schema Validation")
    validation_report = validate_dataset(dataset.headers, dataset.rows)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**Attribute Breakdown:**")
        st.dataframe(
            [{"Column": get_column_display_name(col), "Missing Count": count} for col, count in missing_dict.items() if count > 0]
            or [{"Status": "Zero Missing Values Detected"}],
            use_container_width=True,
        )

    with col2:
        st.markdown("**Dataset Integrity Status:**")
        if validation_report.is_valid:
            st.success("✅ Dataset schema is valid: row lengths match headers, no malformed structural errors.")
        else:
            st.error(f"❌ Validation errors: {validation_report.errors}")
        st.caption("Primary outcome: G3 (Final mathematics grade, scale 0–20). Academic risk defined as G3 < 10.")

    st.markdown("---")
    st.markdown("### 📈 Feature Distributions & Correlation Matrix")
    c_p1, c_p2 = st.columns(2)
    artifacts_dir = os.path.join(PROJECT_ROOT, "artifacts")

    with c_p1:
        p_dist = os.path.join(artifacts_dir, "feature_distributions.png")
        if os.path.exists(p_dist):
            st.image(p_dist, caption="Feature Distributions (Computed from scratch)")
    with c_p2:
        p_corr = os.path.join(artifacts_dir, "correlation_heatmap.png")
        if os.path.exists(p_corr):
            st.image(p_corr, caption="Pearson Correlation Heatmap (Computed with scratch matrix engine)")


# ---------------------------------------------------------
# TAB 2: Regression Studio (Linear Regression)
# ---------------------------------------------------------
elif nav_choice == "📈 Regression Studio (Linear Regression)":
    st.markdown("<div class='main-title'>📈 Regression Studio: Custom Linear Regression</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Predict final mathematics grade (G3) using custom Linear Regression with Normal Equation and Gradient Descent.</div>", unsafe_allow_html=True)

    target_col = "G3" if is_uci else "final_exam_score"
    ignored_cols = ["G1", "G2", "academic_risk"] if is_uci else ["student_id", "academic_risk"]

    schema_reg = infer_schema(
        dataset.headers, dataset.rows,
        target_column=target_col,
        task_type="regression",
        ignored_columns=ignored_cols,
    )

    col_config, col_results = st.columns([1, 1], gap="large")

    with col_config:
        st.subheader("⚙️ Solver & Hyperparameters")
        solver = st.radio("Optimization Solver:", ["gradient_descent", "normal_equation"], horizontal=True)

        model_params: Dict[str, Any] = {"solver": solver}
        fit_intercept = st.checkbox("Fit Intercept (Bias)", value=True)
        model_params["fit_intercept"] = fit_intercept

        if solver == "gradient_descent":
            lr = st.slider("Learning Rate (alpha):", 0.001, 0.1, 0.01, 0.005)
            epochs = st.slider("Maximum Epochs:", 100, 1000, 300, 50)
            momentum = st.slider("Polyak Momentum:", 0.0, 0.9, 0.0, 0.1)
            model_params["learning_rate"] = lr
            model_params["max_epochs"] = epochs
            model_params["momentum"] = momentum
            model_params["early_stopping"] = True
            model_params["patience"] = 10

        l2_reg = st.slider("L2 Ridge Penalty (lambda):", 0.0, 50.0, 0.0, 1.0)
        model_params["l2_reg"] = l2_reg

        scaler_type = st.selectbox("Feature Scaler:", ["standard", "minmax"], index=0)
        test_size = st.slider("Holdout Test Size:", 0.1, 0.4, 0.2, 0.05)

        train_btn = st.button("🚀 Train Custom Linear Regression", type="primary")

    with col_results:
        st.subheader("📊 Model Performance & Artifacts")
        if train_btn:
            raw_X, y = dataset.split_features_target(schema_reg.feature_names, target_col)
            y = [float(v) for v in y]

            with st.spinner("Training Custom Linear Regression..."):
                pipe = MLPipeline(
                    schema=schema_reg,
                    model_name="Linear Regression",
                    model_params=model_params,
                    scaler_type=scaler_type,
                    random_seed=42,
                )
                pipe.fit(raw_X, y, test_size=test_size)
                st.session_state["reg_pipe"] = pipe

        if "reg_pipe" in st.session_state:
            pipe = st.session_state["reg_pipe"]
            metrics = pipe.test_metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("R² Score", f"{metrics.get('r2', 0.0):.4f}")
            m2.metric("RMSE", f"{metrics.get('rmse', 0.0):.4f}")
            m3.metric("MAE", f"{metrics.get('mae', 0.0):.4f}")
            m4.metric("Train Time", f"{pipe.training_time_ms:.1f} ms")

            # Training Loss Curve if GD
            if hasattr(pipe.estimator, "loss_history_") and len(pipe.estimator.loss_history_) > 1:
                p_loss = os.path.join(PROJECT_ROOT, "artifacts", "temp_loss.png")
                plot_loss_curve(pipe.estimator.loss_history_, title="Loss Convergence Curve", save_path=p_loss)
                if os.path.exists(p_loss):
                    st.image(p_loss, caption="Gradient Descent Loss Progression")

            p_act_vs_pred = os.path.join(PROJECT_ROOT, "artifacts", "temp_reg_plot.png")
            plot_actual_vs_predicted(pipe.y_test, pipe.y_test_pred, save_path=p_act_vs_pred)
            if os.path.exists(p_act_vs_pred):
                st.image(p_act_vs_pred, caption="Actual vs. Predicted Final Grade")

    st.markdown("---")
    st.subheader("🧑‍🎓 Single Student Grade Inference")
    if "reg_pipe" in st.session_state:
        pipe = st.session_state["reg_pipe"]
        with st.form("reg_infer_form"):
            user_inputs = render_dynamic_inputs(schema_reg, dataset, key_prefix="reg_single")
            predict_btn = st.form_submit_button("Predict Final Math Grade")
            if predict_btn:
                pred_grade = pipe.predict_single(user_inputs)
                st.success(f"🎯 **Predicted Final Math Grade (G3): {pred_grade:.2f} / 20**")
                if pred_grade < 10.0:
                    st.warning("⚠️ Warning: Student is predicted to fall below the academic passing threshold (10.0).")
                else:
                    st.info("✅ Student is projected to successfully pass the mathematics course.")
    else:
        st.info("Train the model above to enable interactive single-student prediction.")


# ---------------------------------------------------------
# TAB 3: Classification Studio (KNN & Decision Tree)
# ---------------------------------------------------------
elif nav_choice == "🎯 Classification Studio (KNN & Decision Tree)":
    st.markdown("<div class='main-title'>🎯 Classification Studio: Academic Risk Detection</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Predict academic risk (G3 < 10) using custom K-Nearest Neighbors and Decision Tree classifiers.</div>", unsafe_allow_html=True)

    target_col = "academic_risk"
    ignored_cols = ["G1", "G2", "G3"] if is_uci else ["student_id", "final_exam_score"]

    schema_clf = infer_schema(
        dataset.headers, dataset.rows,
        target_column=target_col,
        task_type="binary_classification",
        ignored_columns=ignored_cols,
    )

    col_config, col_results = st.columns([1, 1], gap="large")

    with col_config:
        st.subheader("⚙️ Classifier Configuration")
        model_name = st.radio(
            "Select Custom Classifier:",
            ["KNN Classifier", "Decision Tree Classifier"],
            horizontal=True,
        )

        model_params: Dict[str, Any] = {}
        if model_name == "KNN Classifier":
            k_val = st.slider("Neighborhood Size (k):", 1, 31, 5, 2)
            metric_val = st.selectbox("Distance Metric:", ["euclidean", "manhattan"])
            weights_val = st.selectbox("Neighbor Weighting:", ["uniform", "distance"])
            model_params["n_neighbors"] = k_val
            model_params["metric"] = metric_val
            model_params["weights"] = weights_val
        else:
            max_depth = st.slider("Maximum Depth:", 1, 15, 4, 1)
            min_split = st.slider("Minimum Samples Split:", 2, 20, 2, 1)
            model_params["max_depth"] = max_depth
            model_params["min_samples_split"] = min_split
            model_params["random_seed"] = 42

        test_size = st.slider("Holdout Test Size:", 0.1, 0.4, 0.2, 0.05, key="clf_test_size")
        train_clf_btn = st.button("🚀 Train Custom Classifier", type="primary")

    with col_results:
        st.subheader("📊 Classification Metrics")
        if train_clf_btn:
            raw_X, y = dataset.split_features_target(schema_clf.feature_names, target_col)
            y = [int(v) for v in y]

            with st.spinner(f"Training {model_name}..."):
                pipe = MLPipeline(
                    schema=schema_clf,
                    model_name=model_name,
                    model_params=model_params,
                    random_seed=42,
                )
                pipe.fit(raw_X, y, test_size=test_size)
                st.session_state["clf_pipe"] = pipe

        if "clf_pipe" in st.session_state:
            pipe = st.session_state["clf_pipe"]
            metrics = pipe.test_metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{metrics.get('accuracy', 0.0) * 100:.1f}%")
            m2.metric("Precision", f"{metrics.get('precision', 0.0):.4f}")
            m3.metric("Recall", f"{metrics.get('recall', 0.0):.4f}")
            m4.metric("F1 Score", f"{metrics.get('f1', 0.0):.4f}")

            # Confusion Matrix Plot
            if "confusion_matrix" in metrics:
                p_cm = os.path.join(PROJECT_ROOT, "artifacts", "temp_cm_plot.png")
                plot_confusion_matrix_heatmap(
                    metrics["confusion_matrix"],
                    labels=["Normal (0)", "At-Risk (1)"],
                    title=f"Confusion Matrix: {pipe.model_name}",
                    save_path=p_cm,
                )
                if os.path.exists(p_cm):
                    st.image(p_cm, caption="Empirical Confusion Matrix on Holdout Test Data")

    st.markdown("---")
    st.subheader("🧑‍🎓 Single Student Risk Assessment")
    if "clf_pipe" in st.session_state:
        pipe = st.session_state["clf_pipe"]
        with st.form("clf_infer_form"):
            user_inputs = render_dynamic_inputs(schema_clf, dataset, key_prefix="clf_single")
            predict_clf_btn = st.form_submit_button("Assess Student Academic Risk")
            if predict_clf_btn:
                pred_risk = pipe.predict_single(user_inputs)
                proba_dict = pipe.predict_single_proba(user_inputs)

                if int(pred_risk) == 1:
                    st.error("🚨 **High Academic Risk Detected**: Student is likely to fail final mathematics (G3 < 10).")
                else:
                    st.success("✅ **Normal Academic Trajectory**: Student is on track to pass final mathematics.")

                if proba_dict:
                    st.write("**Model Class Probabilities:**")
                    st.json(proba_dict)
    else:
        st.info("Train the classifier above to enable interactive single-student risk assessment.")


# ---------------------------------------------------------
# TAB 4: Overfitting & Underfitting Explorer
# ---------------------------------------------------------
elif nav_choice == "🔬 Overfitting & Underfitting Explorer":
    st.markdown("<div class='main-title'>🔬 Overfitting & Underfitting Explorer</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Empirical analysis of model complexity curves demonstrating the Bias-Variance tradeoff on the UCI Mathematics dataset.</div>", unsafe_allow_html=True)

    artifacts_dir = os.path.join(PROJECT_ROOT, "artifacts")
    sub_tab1, sub_tab2, sub_tab3 = st.tabs([
        "KNN: Neighborhood Size (k)",
        "Decision Tree: Max Depth",
        "Linear Regression: L2 Regularization",
    ])

    with sub_tab1:
        st.markdown("### 🔍 K-Nearest Neighbors: Impact of k")
        st.markdown("""
        - **k = 1 (Overfitting / High Variance):** Training accuracy reaches 100%, but validation accuracy is suboptimal because the model memorizes idiosyncratic training noise.
        - **Optimal k (Balanced Tradeoff):** Neighborhood averaging filters out noise, reaching peak validation performance.
        - **Large k (Underfitting / High Bias):** Decision boundary becomes overly smooth, predicting global class frequencies and eroding predictive power.
        """)
        p_knn = os.path.join(artifacts_dir, "knn_overfitting_curve.png")
        if os.path.exists(p_knn):
            st.image(p_knn, use_container_width=True)

    with sub_tab2:
        st.markdown("### 🌲 Decision Tree: Impact of Maximum Depth")
        st.markdown("""
        - **Depth = 1 (Underfitting / High Bias):** A single decision stump cannot capture complex interactions among student habits, school support, and past failures.
        - **Intermediate Depth (Optimal):** Captures high-information splits without overfitting individual samples.
        - **Deep Tree (Overfitting / High Variance):** The tree creates tiny leaves matching individual student outliers, causing test performance to degrade.
        """)
        p_dt = os.path.join(artifacts_dir, "tree_overfitting_curve.png")
        if os.path.exists(p_dt):
            st.image(p_dt, use_container_width=True)

    with sub_tab3:
        st.markdown("### 📈 Linear Regression: Impact of L2 Penalty (Ridge Regularization)")
        st.markdown("""
        - **lambda ≈ 0 (Unregularized OLS):** Susceptible to multicollinearity and high variance when many categorical features are one-hot encoded.
        - **Optimal lambda:** Shrinks non-informative regression coefficients towards zero, improving generalization stability.
        - **Very Large lambda (Underfitting):** Over-penalizes parameters, forcing all coefficients close to 0 and predicting near the global target mean.
        """)
        p_lr = os.path.join(artifacts_dir, "linear_regression_regularization_curve.png")
        if os.path.exists(p_lr):
            st.image(p_lr, use_container_width=True)


# ---------------------------------------------------------
# TAB 5: Empirical Benchmark Suite
# ---------------------------------------------------------
elif nav_choice == "📋 Empirical Benchmark Suite":
    st.markdown("<div class='main-title'>📋 Empirical Benchmark Suite</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Reproducible benchmark results computed from scratch across all three custom model families.</div>", unsafe_allow_html=True)

    artifacts_dir = os.path.join(PROJECT_ROOT, "artifacts")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🏆 Classification Comparison")
        p_comp = os.path.join(artifacts_dir, "classification_comparison.png")
        if os.path.exists(p_comp):
            st.image(p_comp, caption="Comparative Performance: KNN vs Decision Tree")

        p_roc = os.path.join(artifacts_dir, "roc_curves.png")
        if os.path.exists(p_roc):
            st.image(p_roc, caption="Receiver Operating Characteristic (ROC) Curves")

    with col2:
        st.markdown("### 📉 Regression Performance")
        p_act = os.path.join(artifacts_dir, "actual_vs_predicted_scores.png")
        if os.path.exists(p_act):
            st.image(p_act, caption="Linear Regression Predictions vs Actual Math Grades")

        p_loss = os.path.join(artifacts_dir, "linear_regression_loss_curve.png")
        if os.path.exists(p_loss):
            st.image(p_loss, caption="Gradient Descent Training Loss Convergence")


# ---------------------------------------------------------
# TAB 6: Architecture & Methods
# ---------------------------------------------------------
elif nav_choice == "📖 Architecture & Methods":
    st.markdown("<div class='main-title'>📖 Architecture & Mathematical Foundations</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Mathematical formulation and architecture of EduPredict implemented entirely from first principles.</div>", unsafe_allow_html=True)

    st.markdown("""
    ### 1. Mathematical Operations Implemented From First Principles
    - **Linear Regression Solvers:**
      - **Normal Equation:** $\\mathbf{w} = (\\mathbf{X}^T \\mathbf{X} + \\lambda \\mathbf{I})^{-1} \\mathbf{X}^T \\mathbf{y}$ solved via Gauss-Jordan elimination with partial row pivoting.
      - **Gradient Descent:** $\\mathbf{w}_{t+1} = \\mathbf{w}_t - \\alpha \\mathbf{v}_t$ with Polyak momentum $\\mathbf{v}_t = \\beta \\mathbf{v}_{t-1} + \\nabla J(\\mathbf{w})$.
    - **K-Nearest Neighbors Classifier:**
      - Euclidean distance: $d(u, v) = \\sqrt{\\sum_{i=1}^n (u_i - v_i)^2}$ computed using pure Python `** 0.5`.
      - Manhattan distance: $d(u, v) = \\sum_{i=1}^n |u_i - v_i|$.
      - Weighted voting: $w_i = \\frac{1}{d(x, x_i) + \\epsilon}$ with deterministic tie-breaking.
    - **Decision Tree Classifier:**
      - Gini Impurity: $Gini(S) = 1 - \\sum_{k=1}^C p_k^2$.
      - Recursive binary partitioning on midpoint thresholds with stopping criteria (`max_depth`, `min_samples_split`, `min_samples_leaf`).
      - Feature importances based on weighted reduction in Gini impurity.

    ### 2. Zero-Leakage Pipeline Architecture
    1. **Pre-split Isolation:** Data records and targets are split jointly before any transformers are applied.
    2. **Train-Only Fitting:** `SimpleImputer`, `OneHotEncoder`, and `StandardScaler` compute summary parameters strictly on training folds.
    3. **Frozen Transformation:** Validation and test splits are transformed using the parameters frozen from the training split.
    4. **Anti-Leakage Safeguard:** Intermediate examination scores `G1` and `G2` are strictly excluded from the feature space.
    """)
