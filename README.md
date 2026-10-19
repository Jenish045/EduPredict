# EduPredict: A Classical Machine Learning Framework Implemented From Scratch

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Strict Dependency Policy](https://img.shields.io/badge/dependencies-Zero%20External%20ML%20Libs-success.svg)]()
[![Lab Project](https://img.shields.io/badge/ML%20Lab-30%20Marks-orange.svg)]()
[![Build & Tests](https://img.shields.io/badge/tests-78%20passed-brightgreen.svg)]()

> **Educational Classical Machine Learning Framework** designed to predict student academic performance (continuous exam scores) and academic risk (binary pass/fail or at-risk) by implementing all linear algebra, statistics, data preprocessing, optimization routines, ML algorithms, loss functions, and evaluation metrics **from scratch** using the Python standard library.

---

## 1. Strict Dependency Policy

Under the project policy, **core algorithms, data structures, and mathematical routines do NOT import or use**:
- ❌ NumPy
- ❌ Pandas
- ❌ Scikit-learn
- ❌ SciPy
- ❌ PyTorch / TensorFlow / Keras

All matrix algebra, statistical equations, iterative optimizers, loss gradients, decision trees, and validation metrics are developed from first principles using standard Python lists, dictionaries, `math`, `random`, `csv`, and `collections`.

*Permitted auxiliary packages:*
- `matplotlib` (strictly for rendering diagnostic PNG plots)
- `streamlit` (for interactive lab demo interface)

An automated AST static analysis test (`tests/test_dependency_compliance.py`) enforces this policy across the entire codebase.

---

## 2. Dataset & Educational Problem Formulation

See [DATASET.md](data/DATASET.md) for full schema definitions, provenance notes, and variable dictionaries.

### Objective A: Academic Performance Prediction (Regression)
- **Target Variable:** `final_exam_score` $\in [0.0, 100.0]$
- **Input Features:** `study_hours`, `attendance`, `past_score`, `assignments_completed`, `parental_education`, `internet_access`, `tutoring`.
- **Metrics:** MSE, RMSE, MAE, $R^2$ Score.

### Objective B: Academic Risk Early Warning (Classification)
- **Target Variable:** `academic_risk` $\in \{0, 1\}$
  - `0`: Projected Pass ($\ge 60.0\%$)
  - `1`: At-Risk / Needs Intervention ($< 60.0\%$)
- **Metrics:** Accuracy, Precision, Recall, $F_1$-score, Confusion Matrix, ROC-AUC.

### Data Leakage Prevention
1. **Target-Target Leakage:** Because `academic_risk` is derived from `final_exam_score`, neither target is ever included as an input feature for predicting the other.
2. **Transformer Fitting:** All imputation parameters (mean/mode) and scaling statistics ($\mu, \sigma, \min, \max$) are fit **strictly on `X_train`** and applied to `X_test` via `.transform()`.

---

## 3. Four Core Classical ML Algorithms (Implemented From Scratch)

### 1. Linear Regression (`edupredict.models.linear_regression`)
- **Hypothesis:** $\hat{y} = \mathbf{w}^T \mathbf{x} + b$
- **Loss Function:** Mean Squared Error:
  $$J(\mathbf{w}, b) = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2$$
- **Gradient Update:**
  $$\mathbf{w} \leftarrow \mathbf{w} - \alpha \left(\frac{2}{N} X^T (\mathbf{\hat{y}} - \mathbf{y}) + 2\lambda \mathbf{w}\right)$$
  $$b \leftarrow b - \alpha \left(\frac{2}{N} \sum_{i=1}^N (\hat{y}_i - y_i)\right)$$
- **Hyperparameters:** `learning_rate` ($\alpha$), `max_epochs`, `batch_size` (Batch/Mini-batch/SGD), `fit_intercept` (True/False), `momentum` ($\beta$), `l2_penalty` ($\lambda$).
- **Computational Complexity:**
  - Training: $O(\text{epochs} \cdot N \cdot d)$
  - Prediction: $O(d)$ per sample

### 2. Binary Logistic Regression (`edupredict.models.logistic_regression`)
- **Linear Logit Score:** $z = \mathbf{w}^T \mathbf{x} + b$
- **Sigmoid Activation (Numerically Stable):**
  $$\sigma(z) = \begin{cases} \frac{1}{1 + e^{-z}} & \text{if } z \ge 0 \\ \frac{e^z}{1 + e^z} & \text{if } z < 0 \end{cases}$$
- **Loss Function:** Binary Cross-Entropy (Log Loss) with probability clipping to $[\epsilon, 1 - \epsilon]$ ($\epsilon = 10^{-15}$):
  $$J(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^N [y_i \ln(\hat{p}_i) + (1 - y_i) \ln(1 - \hat{p}_i)]$$
- **Analytical Gradient with respect to logits $z$:**
  $$\frac{\partial J}{\partial z_i} = \frac{1}{N} (\hat{p}_i - y_i)$$
- **Decision Rule:** $\hat{y} = 1$ if $\hat{p} \ge \text{threshold}$ (default $0.5$) else $0$.
- **Computational Complexity:** Training $O(\text{epochs} \cdot N \cdot d)$, Inference $O(d)$.

### 3. K-Nearest Neighbours (`edupredict.models.knn`)
- **Pairwise Distance:** Exact Euclidean or Manhattan metric from `edupredict.math.vectors`:
  $$d(\mathbf{u}, \mathbf{v}) = \sqrt{\sum_{j=1}^d (u_j - v_j)^2}$$
- **Neighbour Selection & Weights:**
  - `uniform`: Each neighbor has equal vote $1$.
  - `distance`: Weighted vote $w_i = \frac{1}{d_i + 10^{-5}}$.
- **Deterministic Tie-Breaking:** In the event of a tie among the top majority vote candidates, the classifier selects the candidate with the smallest distance to its closest neighbor in the $k$-NN set, falling back to lexicographical label ordering.
- **Posterior Probability (`predict_proba`):** Normalizes neighbor vote weights into a calibrated class probability distribution.
- **Computational Complexity:** Training $O(1)$ (lazy learner), Inference $O(N_{\text{train}} \cdot d)$ per query sample.

### 4. Decision Tree Classifier (`edupredict.models.decision_tree`)
- **Splitting Criteria:**
  - **Gini Impurity:** $I_G(S) = 1 - \sum_{k=1}^K p_k^2$
  - **Entropy:** $H(S) = -\sum_{k=1}^K p_k \log_2(p_k)$
- **Information Gain / Impurity Reduction:**
  $$\Delta I = I(S) - \left(\frac{|S_L|}{|S|} I(S_L) + \frac{|S_R|}{|S|} I(S_R)\right)$$
- **Stopping Conditions:** Node purity ($I=0$), maximum tree depth (`max_depth`), or minimum sample threshold (`min_samples_split`).
- **Feature Importance:** Accumulated sample-weighted impurity reduction per feature, normalized to sum to $1.0$.
- **Computational Complexity:** Training $O(d \cdot N \log N)$, Inference $O(\text{depth})$.

---

## 4. Evaluation Metrics Implemented From Scratch

- **Confusion Matrix:** True Positives ($TP$), False Positives ($FP$), True Negatives ($TN$), False Negatives ($FN$).
- **Accuracy:** $\frac{TP + TN}{TP + TN + FP + FN}$
- **Precision:** $\frac{TP}{TP + FP}$ (with `zero_division=0.0` guard).
- **Recall (Sensitivity):** $\frac{TP}{TP + FN}$
- **$F_1$-Score:** $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
- **Regression Metrics:** MSE ($\frac{1}{N} \sum e_i^2$), RMSE ($\sqrt{\text{MSE}}$), MAE ($\frac{1}{N} \sum |e_i|$), and $R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}$.
- **ROC Curve & AUC Score (`roc_curve`, `roc_auc_score`):**
  - Sorts candidate decision thresholds descending.
  - Computes $(FPR, TPR)$ trajectory from $(0, 0)$ to $(1, 1)$.
  - Calculates Area Under Curve using the trapezoidal numerical integration rule:
    $$\text{AUC} = \sum_{i=1}^{M-1} \frac{TPR_i + TPR_{i-1}}{2} \cdot |FPR_i - FPR_{i-1}|$$

---

## 5. Fair Model Comparison & Empirical Results

Evaluated on the 600-student cohort with identical 80% train / 20% test partition, leak-free preprocessing, and execution timing:

### Task 1: Predicting Final Exam Score (Regression)
| Model | MSE | RMSE | MAE | $R^2$ Score | Training Time | Prediction Time |
|---|---|---|---|---|---|---|
| **Linear Regression (GD)** | **15.0003** | **3.8730** | **3.0972** | **0.8997** | 872 ms | 0.93 ms |
| **Linear Regression (Normal Eq)** | **14.9979** | **3.8727** | **3.0970** | **0.8997** | 44 ms | 1.37 ms |
| **KNN Regressor ($k=7$)** | 41.8624 | 6.4701 | 5.1624 | 0.7200 | 0.79 ms | 1019 ms |

*Insight:* Linear Regression achieves an exceptional $R^2 \approx 0.90$ with identical convergence between the closed-form Normal Equation and Gradient Descent. KNN is non-parametric ($O(1)$ train) but has $1000\times$ higher inference latency due to exhaustive distance calculation.

### Task 2: Academic Risk Classification (Binary: Pass vs. At-Risk)
| Model | Accuracy | Precision | Recall | $F_1$-Score | ROC-AUC | Training Time | Prediction Time |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** | **56.67%** | **0.5714** | **0.4138** | **0.4800** | **0.5545** | 2514 ms | 1.99 ms |
| **Gaussian Naive Bayes** | 50.83% | 0.4909 | 0.4655 | 0.4779 | 0.5412 | 8.4 ms | 4.00 ms |
| **Decision Tree (Depth=5)** | 49.17% | 0.4444 | 0.2069 | 0.2824 | 0.4702 | 713 ms | 0.32 ms |
| **KNN Classifier ($k=5$)** | 47.50% | 0.4545 | 0.4310 | 0.4425 | 0.4665 | 1.07 ms | 946 ms |

*Generated Diagnostic Figures in `artifacts/`:*
- `linear_regression_loss.png` (GD MSE convergence)
- `logistic_regression_loss.png` (BCE log-loss curve)
- `actual_vs_predicted_scores.png` (1:1 regression fit)
- `confusion_matrix.png` (Logistic Regression confusion matrix)
- `feature_importances.png` (Decision Tree feature importance bars)
- `roc_curves.png` (Comparative ROC curves for all classifiers)

---

## 6. Setup & Automated Verification Instructions

### Step 1: Install Optional Visualization Tools
```powershell
pip install -r requirements.txt
```

### Step 2: Run All 78 Automated Unit Tests
```powershell
python -m unittest discover -s tests -v
```

### Step 3: Run the Strict Dependency Compliance Test
```powershell
python -m unittest tests/test_dependency_compliance.py -v
```

### Step 4: Execute the End-to-End Educational Benchmark
```powershell
python run_experiments.py --seed 42 --output_dir artifacts
```

---

## 7. Command to Launch the Next Phase
To launch the interactive Streamlit student risk inference dashboard:
```powershell
streamlit run app.py
```
