# EduPredict Student Performance Dataset Documentation

## 1. Overview & Provenance

The primary dataset used in **EduPredict** is an educational benchmark dataset modeled after classical educational data mining schemas (specifically structured closely to the renowned UCI Machine Learning Repository *Student Performance* benchmark by Cortez and Silva, 2008).

> **Provenance Notice:**
> In accordance with academic integrity standards, this local dataset (`data/student_performance.csv`) is **explicitly synthetic**, deterministically generated based on well-established empirical relationships in higher-education analytics. It is **never** described as real confidential student personal records. It enables 100% offline, reproducible ML lab experiments without external network downloads or student privacy concerns.

---

## 2. Problem Formulation

The framework investigates two distinct Classical ML objectives on this cohort:

### Objective A: Academic Performance Prediction (Regression)
- **Target Variable:** `final_exam_score`
- **Type:** Continuous numeric $\in [0.0, 100.0]$
- **Objective:** Predict the exact percentage score a student achieves on the cumulative final examination given their historical and behavioral indicators during the semester.
- **Evaluation Metrics:** Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), Coefficient of Determination ($R^2$).

### Objective B: Early Academic Risk Detection (Classification)
- **Target Variable:** `academic_risk`
- **Type:** Binary indicator $\{0, 1\}$
  - `0 (Normal / On-Track)`: Student is projected to successfully pass the course ($\ge 60.0\%$).
  - `1 (At-Risk / Needs Intervention)`: Student is in danger of failing ($< 60.0\%$) and flagged for early academic counseling.
- **Evaluation Metrics:** Classification Accuracy, Confusion Matrix (TP, FP, TN, FN), Precision, Recall, and $F_1$-score.

---

## 3. Schema & Feature Dictionary

| Feature Name | Type | Allowed Values / Range | Description |
| :--- | :--- | :--- | :--- |
| `student_id` | Identifier | `STU_0001` - `STU_0600` | Unique student record identifier (excluded from model feature matrix). |
| `study_hours` | Numeric (Float) | $4.0 - 40.0$ hours/week | Self-reported weekly study hours outside lecture time. |
| `attendance` | Numeric (Float) | $45.0\% - 100.0\%$ | Lecture and lab attendance percentage across the semester. |
| `past_score` | Numeric (Float) | $30.0 - 95.0$ marks | Performance score achieved in prerequisite / midterm exams. |
| `assignments_completed` | Numeric (Integer) | $0 - 10$ assignments | Count of completed and submitted weekly lab assignments. |
| `parental_education` | Categorical | `HighSchool`, `Bachelor`, `Master` | Highest educational attainment level of student's guardians. |
| `internet_access` | Categorical | `Yes`, `No` | Availability of dedicated high-speed broadband at residence. |
| `tutoring` | Categorical | `Yes`, `No` | Enrollment in supplemental tutoring or remedial coaching. |
| **`final_exam_score`** | **Target (Regression)** | **$0.0 - 100.0$** | Continuous cumulative exam score. |
| **`academic_risk`** | **Target (Classification)**| **`0` or `1`** | `1` if `final_exam_score < 60.0` else `0`. |

---

## 4. Preprocessing & Data Cleansing

1. **Missing Value Injection & Handling:**
   - Real-world student records frequently have missing attendance entries or survey omissions. The generator injects realistic missing values (~2% rate) marked as `None` / empty cells.
   - Handled via `SimpleImputer` using column mean for numeric features and column mode for categorical features.
2. **Encoding:**
   - Categorical columns are encoded via `OneHotEncoder` or `LabelEncoder` from scratch.
3. **Scaling:**
   - Standardized via `StandardScaler` ($z = \frac{x - \mu}{\sigma}$) or scaled via `MinMaxScaler` ($[0, 1]$).

---

## 5. Critical Note on Data Leakage Prevention

In educational data science, **data leakage** can easily corrupt model validation:

1. **Target-Target Leakage:**
   - Because `academic_risk` is directly derived from `final_exam_score` thresholding, **neither target must EVER be included as an input feature** when predicting the other. The pipeline strictly isolates the 7 feature columns (`study_hours`, `attendance`, `past_score`, `assignments_completed`, `parental_education`, `internet_access`, `tutoring`) from both target columns.
2. **Identifier Leakage:**
   - `student_id` is an arbitrary sequential key and is stripped before model training.
3. **Transformer Fit Leakage:**
   - Imputation parameters ($\mu$, mode) and scaling parameters ($\mu, \sigma, \min, \max$) must **always be computed on `X_train` only**, then applied to `X_test` via `.transform()`. Fitting transformers on the combined dataset is strictly prevented and verified by automated unit tests.
