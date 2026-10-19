# EduPredict: A Classical Machine Learning Framework Implemented From Scratch

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Strict Dependency Policy](https://img.shields.io/badge/dependencies-Zero%20External%20ML%20Libs-success.svg)]()
[![Lab Project](https://img.shields.io/badge/ML%20Lab-30%20Marks-orange.svg)]()
[![Build & Tests](https://img.shields.io/badge/tests-63%20passed-brightgreen.svg)]()

> **Educational Classical Machine Learning Framework** designed to predict student academic performance (continuous exam scores) and academic risk (binary pass/fail or at-risk) by implementing all linear algebra, statistics, optimization routines, ML models, and evaluation metrics **from scratch** using pure Python standard library.

---

## 1. Strict Dependency Compliance

Under the project policy, **core algorithms, data structures, and mathematical routines do NOT import or use**:
- ❌ NumPy
- ❌ Pandas
- ❌ Scikit-learn
- ❌ SciPy
- ❌ PyTorch / TensorFlow / Keras

All matrix algebra, statistical equations, iterative optimizers, loss gradients, decision trees, and validation metrics are developed from first principles using standard Python lists, dictionaries, `math`, `random`, `csv`, and `collections`.

*Permitted optional auxiliary packages:*
- `matplotlib` (strictly for rendering diagnostic PNG plots)
- `streamlit` (for interactive lab demo interface)

An automated AST static analysis test (`tests/test_dependency_compliance.py`) enforces this policy across the entire codebase.

---

## 2. Architecture & Modular Package Structure

```
EduPredict/
├── edupredict/                      # Core Classical ML Framework
│   ├── __init__.py                  # Package root
│   ├── math/                        # Mathematical & Linear Algebra Foundations
│   │   ├── __init__.py
│   │   ├── vectors.py               # Vector addition, dot product, norms, Euclidean/Manhattan distance
│   │   ├── matrices.py              # Matrix multiplication, transpose, Gaussian elimination inversion
│   │   ├── stats.py                 # Mean, median, mode, variance, std dev, covariance, Pearson r
│   │   └── activations.py           # Stable Sigmoid, Softmax, probability clipping, safe logarithm
│   ├── data/                        # Tabular Data Ingestion & Dataset Container
│   │   ├── __init__.py
│   │   ├── loader.py                # CSV reader, auto-type caster, writer
│   │   ├── dataset.py               # Tabular Dataset container, column slicing, schema summary
│   │   └── generator.py             # Deterministic synthetic student dataset generator
│   ├── preprocessing/               # Feature Engineering & Cleansing
│   │   ├── __init__.py
│   │   ├── imputer.py               # SimpleImputer (mean, median, mode, constant)
│   │   ├── encoder.py               # LabelEncoder, OneHotEncoder
│   │   └── scaler.py                # StandardScaler (z-score), MinMaxScaler
│   ├── losses/                      # Objective Functions & Analytical Gradients
│   │   ├── __init__.py
│   │   ├── regression.py            # MSE, MAE, MSE analytical gradient
│   │   └── classification.py        # Binary Cross-Entropy (Log Loss) and logit gradient
│   ├── optimization/                # First-Order Optimizers
│   │   ├── __init__.py
│   │   └── gradient_descent.py      # Batch GD, Mini-Batch GD, SGD with Momentum
│   ├── models/                      # Classical ML Model Implementations
│   │   ├── __init__.py
│   │   ├── base.py                  # Abstract BaseModel with fit/predict validation
│   │   ├── linear_regression.py     # Gradient Descent & Normal Equation OLS solvers
│   │   ├── logistic_regression.py   # Binary Logistic Regression with Sigmoid & BCE
│   │   ├── knn.py                   # KNNClassifier & KNNRegressor (uniform & distance weighted)
│   │   ├── decision_tree.py         # DecisionTreeClassifier (Gini Impurity & Entropy Gain)
│   │   └── naive_bayes.py           # GaussianNaiveBayes (log-space Gaussian PDF)
│   ├── metrics/                     # Model Evaluation Metrics
│   │   ├── __init__.py
│   │   ├── regression.py            # MSE, RMSE, MAE, R^2 Coefficient of Determination
│   │   └── classification.py        # Accuracy, Confusion Matrix, Precision, Recall, F1, Report
│   ├── model_selection/             # Validation & Partitioning
│   │   ├── __init__.py
│   │   ├── split.py                 # train_test_split (stratified/random), KFold cross-validator
│   │   └── validation.py            # cross_val_score
│   ├── visualization/               # Diagnostic Plotting
│   │   ├── __init__.py
│   │   └── plots.py                 # Loss curves, confusion matrix, residuals, feature importance
│   └── experiments/                 # Benchmarking & Experimentation
│       ├── __init__.py
│       ├── benchmark.py             # Tabular comparison reporter
│       └── runner.py                # End-to-end benchmark pipeline execution
├── tests/                           # Comprehensive Unit Test Suite
│   ├── __init__.py
│   ├── test_math_vectors.py         # 10 tests: operations, norms, distances, edge cases
│   ├── test_math_matrices.py        # 7 tests: matmul, transpose, Gaussian elimination inversion
│   ├── test_math_stats.py           # 9 tests: mean, median, mode, variance, covariance, Pearson r
│   ├── test_math_activations.py     # 5 tests: stable sigmoid, safe log, softmax, clipping
│   ├── test_data_loader.py          # 4 tests: CSV I/O, Dataset class, generator determinism
│   ├── test_preprocessing.py        # 6 tests: Imputer, LabelEncoder, OneHotEncoder, Scalers
│   ├── test_losses.py               # 5 tests: MSE, MAE, BCE, analytical gradients
│   ├── test_optimization.py         # 3 tests: Batching, momentum parameter updates
│   ├── test_models.py               # 7 tests: LinearReg (GD & NE), LogisticReg, KNN, Tree, NB
│   ├── test_metrics.py              # 2 tests: Regression and classification evaluation
│   ├── test_model_selection.py      # 4 tests: Splitting, stratification, K-fold, cross_val
│   └── test_dependency_compliance.py# AST compliance check for zero prohibited ML libraries
├── data/                            # Datasets
│   └── student_performance.csv      # Generated 600-student realistic academic dataset
├── artifacts/                       # Diagnostic visual charts
│   ├── linear_regression_loss.png
│   ├── logistic_regression_loss.png
│   ├── actual_vs_predicted_scores.png
│   ├── confusion_matrix.png
│   └── feature_importances.png
├── requirements.txt
├── run_experiments.py               # Experiment pipeline entry point
└── README.md
```

---

## 3. Mathematical Foundations & Viva Defense Guide

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
  $$J(\mathbf{w}, b) = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2, \quad \frac{\partial J}{\partial \mathbf{w}} = \frac{2}{N} X^T (\mathbf{\hat{y}} - \mathbf{y}), \quad \frac{\partial J}{\partial b} = \frac{2}{N} \sum (\hat{y}_i - y_i)$$
- **Binary Cross-Entropy (BCE):**
  $$J(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^N [y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i)]$$
  $$\frac{\partial J}{\partial z_i} = \frac{1}{N} (\hat{p}_i - y_i)$$

### 5. Ordinary Least Squares (Normal Equation)
Solves the normal equation in closed form:
$$\mathbf{\theta} = (X_{\text{aug}}^T X_{\text{aug}} + \lambda I)^{-1} X_{\text{aug}}^T \mathbf{y}$$
where $\lambda$ is a small Ridge regularization term guarding against multicollinearity.

### 6. Gaussian Naive Bayes Log-Likelihood
Evaluates continuous feature likelihoods in log-space:
$$\log P(x_j | C_k) = -\frac{1}{2} \ln(2\pi \sigma_{kj}^2) - \frac{(x_j - \mu_{kj})^2}{2\sigma_{kj}^2}$$

---

## 4. Setup & Execution Instructions

### Prerequisites
- Python 3.10, 3.11, or 3.12 (standard distribution)
- `pip`

### Step 1: Install Optional Visualization Tools
```powershell
pip install -r requirements.txt
```

### Step 2: Run All Automated Unit Tests (63 tests)
```powershell
python -m unittest discover -s tests -v
```
All 63 tests across math, preprocessing, losses, optimization, models, metrics, and compliance will run and report `OK`.

### Step 3: Run the Compliance Check Directly
To verify that no prohibited ML libraries are imported anywhere in `edupredict`:
```powershell
python -m unittest tests/test_dependency_compliance.py -v
```

### Step 4: Execute the End-to-End Educational ML Experiment
```powershell
python run_experiments.py --seed 42 --output_dir artifacts
```

---

## 5. Experimental Benchmark Results

Running on a 600-student cohort with 80% train / 20% test partition:

### Task 1: Predicting Final Exam Score (Regression)
| Model | MSE | RMSE | MAE | $R^2$ Score |
|---|---|---|---|---|
| **Linear Regression (Normal Eq)** | **14.9979** | **3.8727** | **3.0970** | **0.8997** |
| **Linear Regression (Gradient Descent)** | **14.9979** | **3.8727** | **3.0970** | **0.8997** |
| **KNN Regressor ($k=7$, distance-weighted)** | 42.2466 | 6.4997 | 5.1792 | 0.7174 |

*Key finding:* The closed-form analytical Normal Equation and iterative Gradient Descent converge to the identical optimal solution ($R^2 \approx 0.90$), validating both numerical solvers.

### Task 2: Academic Risk Classification (Normal vs. At-Risk)
| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Logistic Regression** | **55.83%** | **0.5581** | **0.4138** | **0.4752** |
| **Gaussian Naive Bayes** | 50.83% | 0.4909 | 0.4655 | 0.4779 |
| **Decision Tree (Max Depth = 5)** | 49.17% | 0.4444 | 0.2069 | 0.2824 |
| **KNN Classifier ($k=5$)** | 47.50% | 0.4545 | 0.4310 | 0.4425 |

---

## 6. Known Limitations & Next Steps

1. **Matrix Inversion Complexity:** The scratch Gauss-Jordan elimination algorithm is $O(d^3)$. It is exceptionally fast and educational for tabular feature dimensions ($d \le 100$), but for massive feature spaces ($d > 5000$), gradient descent solvers should be preferred.
2. **Phase 2 Scope:** In the next phase, we can introduce:
   - Interactive Streamlit Web UI dashboard for live student risk inference.
   - Multiclass classification extensions (predicting Letter Grades: A, B, C, D, F).
   - Additional ensemble models from scratch (Random Forest with bagging).

---

## 7. Command to Launch the Next Phase
To launch the interactive Streamlit dashboard interface in the next phase:
```powershell
streamlit run app.py
```
