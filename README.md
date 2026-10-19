# EduPredict: A Classical Machine Learning Framework Implemented From Scratch

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Strict Dependency Policy](https://img.shields.io/badge/dependencies-Zero%20External%20ML%20Libs-success.svg)]()
[![Lab Project](https://img.shields.io/badge/ML%20Lab-30%20Marks-orange.svg)]()
[![Build & Tests](https://img.shields.io/badge/tests-74%20passed-brightgreen.svg)]()

> **Educational Classical Machine Learning Framework** designed to predict student academic performance (continuous exam scores) and academic risk (binary pass/fail or at-risk) by implementing all linear algebra, statistics, data preprocessing, optimization routines, ML algorithms, loss functions, and evaluation metrics **from scratch** using the Python standard library.

---

## 1. Strict Dependency Compliance

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
- **Metrics:** Accuracy, Precision, Recall, $F_1$-score, Confusion Matrix.

### Data Leakage Prevention
1. **Target-Target Leakage:** Because `academic_risk` is derived from `final_exam_score`, neither target is ever included as an input feature for predicting the other.
2. **Transformer Fitting:** All imputation parameters (mean/mode) and scaling statistics ($\mu, \sigma, \min, \max$) are fit **strictly on `X_train`** and applied to `X_test` via `.transform()`.

---

## 3. Architecture & Modular Package Structure

```
EduPredict/
├── edupredict/                              # Core Classical ML Framework
│   ├── math/
│   │   ├── vectors.py                       # Vector add, sub, mul, scalar mul, dot product, norms, Euclidean/Manhattan
│   │   ├── matrices.py                      # Matrix transpose, multiplication, vector mul, Gauss-Jordan inversion
│   │   ├── stats.py                         # Summation, mean, median, mode, variance, std dev, covariance, Pearson r
│   │   └── activations.py                   # Numerically stable Sigmoid, Softmax, probability clipping, safe log
│   ├── data/
│   │   ├── loader.py                        # CSV parsing, schema validation, malformed row checks, ValidationReport
│   │   ├── dataset.py                       # Tabular Dataset container with schema summary & column slicing
│   │   └── generator.py                     # Deterministic synthetic student dataset generator
│   ├── preprocessing/
│   │   ├── imputer.py                       # SimpleImputer (mean, median, mode, constant)
│   │   ├── encoder.py                       # LabelEncoder & OneHotEncoder (with unknown category handling)
│   │   └── scaler.py                        # StandardScaler (z-score) & MinMaxScaler (constant feature safe)
│   ├── losses/
│   │   ├── regression.py                    # MSE, MAE, analytical MSE gradient
│   │   └── classification.py                # Binary Cross-Entropy (Log Loss), Hinge Loss, analytical gradients
│   ├── optimization/
│   │   └── gradient_descent.py              # Batch GD, Mini-Batch, SGD, Momentum, Early Stopping, Initializers
│   ├── models/
│   │   ├── base.py                          # Abstract BaseModel with fit/predict validation
│   │   ├── linear_regression.py             # Gradient Descent & Normal Equation OLS solvers
│   │   ├── logistic_regression.py           # Binary Logistic Regression with Sigmoid & BCE
│   │   ├── knn.py                           # KNNClassifier & KNNRegressor (uniform & distance weighted)
│   │   ├── decision_tree.py                 # DecisionTreeClassifier (Gini Impurity & Entropy Gain)
│   │   └── naive_bayes.py                   # GaussianNaiveBayes (log-space Gaussian PDF)
│   ├── metrics/
│   │   ├── regression.py                    # MSE, RMSE, MAE, R^2 Coefficient of Determination
│   │   └── classification.py                # Accuracy, Confusion Matrix, Precision, Recall, F1, Report
│   ├── model_selection/
│   │   ├── split.py                         # train_test_split (stratified/random), KFold, StratifiedKFold
│   │   └── validation.py                    # cross_val_score
│   ├── visualization/
│   │   └── plots.py                         # Loss curves, confusion matrix, residuals, feature importance
│   └── experiments/
│       ├── benchmark.py                     # Tabular comparison reporter
│       └── runner.py                        # End-to-end benchmark pipeline execution
├── tests/                                   # 74 Automated Unit Tests
│   ├── test_math_vectors.py                 # 10 tests: operations, norms, distances, edge cases
│   ├── test_math_matrices.py                # 7 tests: matmul, transpose, Gaussian elimination inversion
│   ├── test_math_stats.py                   # 9 tests: mean, median, mode, variance, covariance, Pearson r
│   ├── test_math_activations.py             # 5 tests: stable sigmoid, safe log, softmax, clipping
│   ├── test_data_loader.py                  # 8 tests: CSV I/O, malformed rows, schema validation, generator
│   ├── test_preprocessing.py                # 9 tests: Imputer, Encoders, Scalers, constant features, leakage
│   ├── test_losses.py                       # 8 tests: MSE, MAE, BCE, Hinge, finite-difference gradient checks
│   ├── test_optimization.py                 # 6 tests: Batching, momentum, initializers, early stopping
│   ├── test_models.py                       # 7 tests: LinearReg (GD & NE), LogisticReg, KNN, Tree, NB
│   ├── test_metrics.py                      # 2 tests: Regression and classification evaluation
│   ├── test_model_selection.py              # 5 tests: Splitting, determinism, StratifiedKFold, cross_val
│   └── test_dependency_compliance.py        # AST compliance check for zero prohibited ML libraries
├── data/
│   ├── DATASET.md                           # Comprehensive dataset documentation & provenance
│   └── student_performance.csv              # Local reproducible 600-student dataset
├── artifacts/                               # Diagnostic visual charts
│   ├── linear_regression_loss.png
│   ├── logistic_regression_loss.png
│   ├── actual_vs_predicted_scores.png
│   ├── confusion_matrix.png
│   └── feature_importances.png
├── requirements.txt
├── run_experiments.py                       # Main experiment execution script
└── README.md
```

---

## 4. Mathematical Foundations & Viva Defense Guide

During viva examination, students can explain and demonstrate the mathematical foundations directly from our codebase:

### 1. Vector & Matrix Linear Algebra
- **Dot Product:** $\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{n} u_i v_i$
- **Matrix Multiplication:** $C_{ij} = \sum_{k=1}^{p} A_{ik} B_{kj}$
- **Matrix Inversion:** Implemented via Gauss-Jordan Elimination with partial pivoting across row operations to invert $(X^T X)$ for closed-form regression.

### 2. Descriptive & Inferential Statistics
- **Sample Mean:** $\bar{x} = \frac{1}{N} \sum_{i=1}^N x_i$
- **Sample Variance:** $s^2 = \frac{1}{N - 1} \sum_{i=1}^N (x_i - \bar{x})^2$
- **Sample Covariance:** $\text{Cov}(x, y) = \frac{1}{N - 1} \sum_{i=1}^N (x_i - \bar{x})(y_i - \bar{y})$
- **Pearson Correlation:** $r = \frac{\text{Cov}(x, y)}{s_x \cdot s_y}$

### 3. Numerical Stability in Sigmoid & Cross-Entropy
- **Numerically Stable Sigmoid:**
  $$\sigma(z) = \begin{cases} \frac{1}{1 + e^{-z}} & \text{if } z \ge 0 \\ \frac{e^z}{1 + e^z} & \text{if } z < 0 \end{cases}$$
  *Prevents exponential overflow when $z \ll 0$.*
- **Probability Clipping:** Clamps $p \in [\epsilon, 1 - \epsilon]$ where $\epsilon = 10^{-15}$ to eliminate catastrophic $\log(0) = -\infty$ floating-point errors.

### 4. Loss Functions & Analytical Gradients
- **Mean Squared Error (MSE):**
  $$J(\mathbf{w}, b) = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2, \quad \frac{\partial J}{\partial \hat{y}_i} = \frac{2}{N} (\hat{y}_i - y_i)$$
- **Binary Cross-Entropy (BCE):**
  $$J(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^N [y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i)], \quad \frac{\partial J}{\partial z_i} = \frac{1}{N} (\hat{p}_i - y_i)$$
- **Hinge Loss:**
  $$L = \frac{1}{N} \sum_{i=1}^N \max(0, 1 - y_i \cdot \hat{y}_i)$$
- **Finite-Difference Verification:** Verified by unit tests comparing analytical gradients against centered numerical difference:
  $$\frac{\partial J}{\partial \theta_j} \approx \frac{J(\theta_j + \epsilon) - J(\theta_j - \epsilon)}{2\epsilon}$$

### 5. Ordinary Least Squares (Normal Equation)
Solves the normal equation in closed form:
$$\mathbf{\theta} = (X_{\text{aug}}^T X_{\text{aug}} + \lambda I)^{-1} X_{\text{aug}}^T \mathbf{y}$$
where $\lambda$ is a small Ridge regularization term guarding against multicollinearity.

### 6. Optimization Routines
- **Batch GD, Mini-Batch GD, SGD:** Configurable batch slicing with seeded shuffling.
- **Polyak Momentum:** $v_{t} = \beta v_{t-1} + \alpha \nabla_{\theta} J(\theta)$, $\theta_t = \theta_{t-1} - v_t$.
- **Early Stopping & Convergence:** Tracks best loss with configurable `patience` and `min_delta`.

---

## 5. Setup & Execution Instructions

### Step 1: Install Optional Visualization Tools
```powershell
pip install -r requirements.txt
```

### Step 2: Run All Automated Unit Tests (74 tests)
```powershell
python -m unittest discover -s tests -v
```

### Step 3: Run the Strict Dependency Compliance Test
```powershell
python -m unittest tests/test_dependency_compliance.py -v
```

### Step 4: Execute the End-to-End Educational ML Experiment
```powershell
python run_experiments.py --seed 42 --output_dir artifacts
```

---

## 6. Experimental Benchmark Results

Running on a 600-student cohort with 80% train / 20% test partition:

### Task 1: Predicting Final Exam Score (Regression)
| Model | MSE | RMSE | MAE | $R^2$ Score |
|---|---|---|---|---|
| **Linear Regression (Normal Eq)** | **14.9979** | **3.8727** | **3.0970** | **0.8997** |
| **Linear Regression (Gradient Descent)** | **14.9979** | **3.8727** | **3.0970** | **0.8997** |
| **KNN Regressor ($k=7$, distance-weighted)** | 42.2466 | 6.4997 | 5.1792 | 0.7174 |

### Task 2: Academic Risk Classification (Normal vs. At-Risk)
| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Logistic Regression** | **55.83%** | **0.5581** | **0.4138** | **0.4752** |
| **Gaussian Naive Bayes** | 50.83% | 0.4909 | 0.4655 | 0.4779 |
| **Decision Tree (Max Depth = 5)** | 49.17% | 0.4444 | 0.2069 | 0.2824 |
| **KNN Classifier ($k=5$)** | 47.50% | 0.4545 | 0.4310 | 0.4425 |

---

## 7. Command to Launch the Next Phase
To launch the interactive Streamlit student risk dashboard in the next phase:
```powershell
streamlit run app.py
```
