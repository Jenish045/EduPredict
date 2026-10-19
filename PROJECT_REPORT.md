# College ML Lab Project Report: EduPredict
## Classical Machine Learning Framework Implemented From Scratch

---

**Course:** Machine Learning Laboratory  
**Project Title:** EduPredict: A Classical Machine Learning Framework Implemented From Scratch for Academic Performance & Risk Prediction  
**Evaluation Weight:** 30 Marks  
**Implementation Mode:** 100% From Scratch (Zero External ML / Numerical Libraries)  
**Programming Language:** Python 3.10+ (Standard Library Only)  

---

## Abstract

In this project, we design, implement, evaluate, and deploy **EduPredict**, a comprehensive classical machine learning framework engineered entirely from first principles using standard Python. The system addresses a critical institutional educational challenge: forecasting continuous student final examination performance and detecting early academic risk ($< 60\%$ threshold). Under a strict zero-third-party-ML dependency constraint, the underlying linear algebra, matrix inversion, descriptive statistics, missing value imputation, categorical encoding, standardization, loss functions, analytical gradients, first-order optimization routines, four classical ML algorithms (Linear Regression, Logistic Regression, $k$-Nearest Neighbors, and Decision Trees), and complete evaluation suites (including ROC curves and trapezoidal AUC integration) were developed from scratch. We benchmarked the models on a 600-student cohort using a leak-free 80/20 train/test experimental protocol. Linear Regression achieved an $R^2$ score of $0.8997$ with a root mean squared error of $3.87$ marks, while Logistic Regression achieved a classification accuracy of $56.67\%$ with an ROC-AUC of $0.5545$. An interactive Streamlit web dashboard and JSON serialization module were built to demonstrate real-time inference and deployment readiness.

---

## 1. Introduction & Background

Machine Learning is often taught through high-level libraries such as Scikit-Learn, PyTorch, and Pandas. While these tools facilitate rapid prototyping, they obscure the foundational mathematical mechanics of gradient optimization, numerical stability, loss derivation, and data leakage safeguards. 

The objective of this project is to demystify classical machine learning by implementing an end-to-end educational framework without relying on NumPy, Pandas, Scikit-Learn, SciPy, or deep learning frameworks. By building every component from standard library primitives, this project establishes a transparent, auditable, and mathematically rigorous reference for college-level ML evaluation.

---

## 2. Problem Formulation

### 2.1 Predictive Objectives
The educational predictive challenge is structured into two complementary tasks:
1. **Regression Task (Continuous Outcome):** Predict a student's final examination score:
   $$\hat{y} \in [0.0, 100.0]$$
   given their historical academic engagement, prior exam performance, attendance records, and socio-environmental factors.
2. **Classification Task (Binary Early Warning):** Classify whether a student is in good academic standing ($y = 0$) or at-risk of academic failure ($y = 1$):
   $$y = \begin{cases} 1 & \text{if } \text{final\_exam\_score} < 60.0 \\ 0 & \text{if } \text{final\_exam\_score} \ge 60.0 \end{cases}$$

### 2.2 Input Feature Space
The feature vector $\mathbf{x} \in \mathbb{R}^d$ includes:
- Numerical features: Weekly self-study hours ($[0, 50]$), lecture attendance percentage ($[0, 100]$), prior midterm exam score ($[0, 100]$), and lab assignments completed ($[0, 10]$).
- Categorical features: Parental education level (`HighSchool`, `Bachelor`, `Master`), high-speed internet availability (`Yes`, `No`), and academic tutoring support (`Yes`, `No`).

---

## 3. Strict Dependency Policy

To ensure complete implementation authenticity:
- **Prohibited Modules:** `numpy`, `pandas`, `sklearn`, `scipy`, `statsmodels`, `torch`, `tensorflow`, `keras`, `jax`.
- **Permitted Modules:** Python standard library (`math`, `random`, `csv`, `collections`, `json`, `time`, `typing`, `unittest`, `ast`).
- **Auxiliary Presentation Modules:** `matplotlib` (strictly for rendering diagnostic PNG canvases) and `streamlit` (for interactive lab demonstrations).
- **Compliance Enforcement:** An automated AST static code scanner (`tests/test_dependency_compliance.py`) parses the abstract syntax tree of every source file in the package to guarantee zero prohibited imports.

---

## 4. System Architecture & Methodology

The EduPredict framework is modularized into discrete functional layers:

```
EduPredict/
├── edupredict/
│   ├── math/            # Vector, matrix, statistics, and activation engines
│   ├── data/            # CSV ingestion, validation, and container objects
│   ├── preprocessing/   # Leakage-free imputers, encoders, and scalers
│   ├── losses/          # Analytical loss functions and exact gradients
│   ├── optimization/    # Batch, mini-batch, SGD, and momentum optimizers
│   ├── models/          # Linear/Logistic Regression, KNN, Decision Tree, Naive Bayes
│   ├── model_selection/ # Train/test splits, K-Fold, and Stratified K-Fold
│   ├── metrics/         # MSE, RMSE, MAE, R2, Acc, Precision, Recall, F1, ROC-AUC
│   ├── visualization/   # Diagnostic plotting routines
│   └── experiments/     # Benchmarking pipeline and timing suite
├── tests/               # 84 automated unit and compliance tests
├── artifacts/           # Saved PNG diagnostic plots and JSON models
├── app.py               # Interactive Streamlit web application
└── run_experiments.py   # CLI entry point
```

### 4.1 Data Leakage Prevention
In classical ML, data leakage occurs when information from outside the training dataset is inadvertently used to fit model parameters. EduPredict enforces strict isolation:
1. All preprocessors (`SimpleImputer`, `OneHotEncoder`, `StandardScaler`) are fit **only** on the training split $X_{\text{train}}$.
2. Test partitions $X_{\text{test}}$ and live single-student queries are strictly transformed using the frozen training parameters ($\mu_{\text{train}}, \sigma_{\text{train}}$).
3. The continuous regression target and binary classification target are isolated from feature representations to prevent target-target leakage.

---

## 5. Mathematical Formulations & Algorithms

### 5.1 Linear Regression
- **Hypothesis:** $\hat{y} = \mathbf{w}^T \mathbf{x} + b$
- **Objective Function (MSE):**
  $$J(\mathbf{w}, b) = \frac{1}{m} \sum_{i=1}^m (\mathbf{w}^T \mathbf{x}_i + b - y_i)^2 + \lambda \|\mathbf{w}\|_2^2$$
- **Gradient Update:**
  $$\mathbf{w} \leftarrow \mathbf{w} - \alpha \left(\frac{2}{m} \mathbf{X}^T (\mathbf{X}\mathbf{w} + b - \mathbf{y}) + 2\lambda \mathbf{w}\right)$$
  $$b \leftarrow b - \alpha \left(\frac{2}{m} \sum_{i=1}^m (\mathbf{w}^T \mathbf{x}_i + b - y_i)\right)$$
- **Closed-Form Normal Equation:**
  $$\mathbf{w}^* = (\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}$$
  where matrix inversion is performed via Gauss-Jordan elimination with partial row pivoting.

### 5.2 Logistic Regression
- **Probability Hypothesis:** $P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b)$
- **Numerically Stable Sigmoid:**
  $$\sigma(z) = \begin{cases} \frac{1}{1 + e^{-z}} & \text{if } z \ge 0 \\ \frac{e^z}{1 + e^z} & \text{if } z < 0 \end{cases}$$
- **Objective Function (Binary Cross-Entropy):**
  $$J(\mathbf{w}, b) = -\frac{1}{m} \sum_{i=1}^m [y_i \ln(\hat{p}_i) + (1 - y_i) \ln(1 - \hat{p}_i)] + \lambda \|\mathbf{w}\|_2^2$$
  with probabilities clamped to $[10^{-15}, 1 - 10^{-15}]$ to prevent $\ln(0)$ divergence.
- **Analytical Gradient with respect to logits $z_i$:**
  $$\frac{\partial J}{\partial z_i} = \frac{1}{m} (\hat{p}_i - y_i)$$
  $$\nabla_{\mathbf{w}} J = \frac{1}{m} \mathbf{X}^T (\hat{\mathbf{p}} - \mathbf{y}) + 2\lambda \mathbf{w}$$

### 5.3 K-Nearest Neighbors (KNN)
- **Distance Metrics:** Exact Euclidean ($L_2$) and Manhattan ($L_1$) distance.
- **Weighting Schemes:** Uniform ($w_i = 1$) or inverse-distance ($w_i = \frac{1}{d_i + 10^{-8}}$).
- **Deterministic Tie-Breaking:** In classification, ties in vote count are resolved by picking the class with the minimum Euclidean distance to its closest neighbor in the $k$-NN set, falling back to lexicographical sorting.
- **Posterior Probability Estimation:**
  $$P(y = c \mid \mathbf{x}) = \frac{\sum_{i \in N_c} w_i}{\sum_{i=1}^k w_i}$$

### 5.4 Decision Tree Classifier
- **Recursive Splitting:** Evaluates candidate thresholds for each feature to maximize impurity reduction:
  $$\Delta I = I(S) - \left(\frac{|S_L|}{|S|} I(S_L) + \frac{|S_R|}{|S|} I(S_R)\right)$$
- **Supported Criteria:** Gini Impurity ($1 - \sum p_k^2$) and Shannon Entropy ($-\sum p_k \log_2 p_k$).
- **Stopping Rules:** Maximum tree depth, minimum samples per split, and pure node identification.
- **Feature Importance:** Aggregated sample-weighted impurity reduction per feature, normalized to $1.0$.

### 5.5 ROC Curve & AUC Computation
- Evaluates True Positive Rate ($\text{TPR}$) and False Positive Rate ($\text{FPR}$) across all unique predicted probability thresholds.
- Computes Area Under the Curve using the trapezoidal numerical integration rule:
  $$\text{AUC} = \sum_{i=1}^{M-1} \frac{\text{TPR}_i + \text{TPR}_{i-1}}{2} \cdot |\text{FPR}_i - \text{FPR}_{i-1}|$$

---

## 6. Experimental Setup & Results

### 6.1 Dataset & Partitioning
- Total samples: 600 students.
- Training split: 480 students ($80\%$).
- Testing split: 120 students ($20\%$).
- Stratified partitioning was applied for classification to preserve identical class balance across folds.

### 6.2 Empirical Benchmark: Final Exam Score Regression
| Model | MSE | RMSE | MAE | $R^2$ Score | Training Time | Inference Time |
|---|---|---|---|---|---|---|
| **Linear Regression (GD)** | **15.0003** | **3.8730** | **3.0972** | **0.8997** | 1830 ms | 1.94 ms |
| **Linear Regression (Normal Eq)** | **14.9979** | **3.8727** | **3.0970** | **0.8997** | 43.6 ms | 1.83 ms |
| **KNN Regressor ($k=7$)** | 41.8624 | 6.4701 | 5.1624 | 0.7200 | 1.03 ms | 973 ms |

**Analysis:** Linear Regression exhibits superior predictive capability ($R^2 \approx 0.90$), confirming strong linear relationships between study hours, attendance, prior scores, and final achievement. The closed-form Normal Equation converged to identical parameters 40× faster than iterative gradient descent.

### 6.3 Empirical Benchmark: Academic Risk Classification
| Model | Accuracy | Precision | Recall | $F_1$-Score | ROC-AUC | Training Time | Inference Time |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** | **56.67%** | **0.5714** | **0.4138** | **0.4800** | **0.5545** | 1909 ms | 1.96 ms |
| **Gaussian Naive Bayes** | 50.83% | 0.4909 | 0.4655 | 0.4779 | 0.5412 | 4.95 ms | 2.19 ms |
| **Decision Tree (Depth=5)** | 49.17% | 0.4444 | 0.2069 | 0.2824 | 0.4702 | 715 ms | 0.33 ms |
| **KNN Classifier ($k=5$)** | 47.50% | 0.4545 | 0.4310 | 0.4425 | 0.4665 | 0.97 ms | 983 ms |

---

## 7. Verification & Automated Testing Suite

The framework includes **84 automated unit tests** in `tests/` covering:
1. `test_math_vectors.py`: Dot products, norms, distances, and zero-length guards.
2. `test_math_matrices.py`: Matmul, transpose, and Gauss-Jordan inversion with singular matrix detection.
3. `test_math_stats.py`: Mean, median, mode, variance, covariance, and Pearson correlation.
4. `test_math_activations.py`: Stable sigmoid, softmax, clipping, and safe logarithm.
5. `test_data_loader.py`: CSV parsing, malformed row detection, and schema validation.
6. `test_preprocessing.py`: Imputation, encoding, scaling, and data leakage isolation.
7. `test_losses.py`: Analytical gradients verified against centered finite differences ($< 10^{-5}$ tolerance).
8. `test_optimization.py`: Polyak momentum updates, batching, and early stopping.
9. `test_models.py`: Dual-solver linear regression, logistic regression, KNN, and decision trees.
10. `test_metrics.py`: Classification report, confusion matrix, ROC curve, and trapezoidal AUC.
11. `test_serialization.py`: JSON state export and import for all six models.
12. `test_multiclass.py`: One-vs-Rest meta-classification across three-category grades.
13. `test_dependency_compliance.py`: AST static analysis guaranteeing zero prohibited ML library imports.

All 84 tests pass in **0.40 seconds**.

---

## 8. Conclusion & Viva Defense Summary

EduPredict successfully demonstrates that a complete, robust, and mathematically sound Classical Machine Learning pipeline can be designed and executed without third-party ML dependencies. The project fulfills all curriculum requirements for the 30-mark college laboratory evaluation:
- Complete algorithmic transparency from first principles.
- Strict scientific isolation against data leakage.
- Direct mathematical verification of analytical gradients against numerical finite differences.
- Production-grade deployment via JSON model checkpoints and an interactive Streamlit user dashboard.

---

## 9. References
1. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*. Springer.
2. Cortez, P., & Silva, A. M. G. (2008). *Using Data Mining to Predict Secondary School Student Performance*. EUROSIS-ETI.
3. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
4. Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations*. Johns Hopkins University Press.
