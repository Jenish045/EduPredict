"""
Unified, leak-free machine learning pipeline for EduPredict.

Orchestrates data splitting, train-only preprocessing (imputation, encoding, scaling),
Scikit-learn model training, and custom from-scratch metric evaluation.
Eliminates data leakage and provides seamless single-record inference.
"""

import time
from typing import List, Dict, Any, Optional, Tuple, Union

from edupredict.data.dataset import Dataset
from edupredict.data.schema import DatasetSchema, infer_schema
from edupredict.model_selection.split import train_test_split
from edupredict.model_selection.validation import KFold, StratifiedKFold
from edupredict.preprocessing.imputer import SimpleImputer
from edupredict.preprocessing.encoder import OneHotEncoder
from edupredict.preprocessing.scaler import StandardScaler, MinMaxScaler
from edupredict.models.registry import get_model, list_models
from edupredict.metrics.regression import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from edupredict.metrics.classification import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)


class MLPipeline:
    """
    Unified ML pipeline that strictly prevents data leakage by fitting all preprocessors
    only on the training split, and evaluating models using scratch-implemented metrics.
    """

    def __init__(
        self,
        schema: DatasetSchema,
        model_name: str,
        model_params: Optional[Dict[str, Any]] = None,
        imputer_strategy_num: str = "mean",
        imputer_strategy_cat: str = "mode",
        scaler_type: str = "standard",
        random_seed: int = 42,
    ) -> None:
        self.schema = schema
        self.model_name = model_name
        self.model_params = model_params or {}
        self.imputer_strategy_num = imputer_strategy_num
        self.imputer_strategy_cat = imputer_strategy_cat
        self.scaler_type = scaler_type
        self.random_seed = random_seed

        # Preprocessors
        self.num_imputer: Optional[SimpleImputer] = None
        self.cat_imputer: Optional[SimpleImputer] = None
        self.ohe: Optional[OneHotEncoder] = None
        self.scaler: Optional[Union[StandardScaler, MinMaxScaler]] = None

        # Estimator
        self.estimator: Optional[Any] = None
        self.is_fitted: bool = False

        # Stored training artifacts and evaluation metrics
        self.train_metrics: Dict[str, Any] = {}
        self.test_metrics: Dict[str, Any] = {}
        self.training_time_ms: float = 0.0
        self.inference_time_ms: float = 0.0

        # Cached split data for inspection and plotting
        self.X_train_scaled: List[List[float]] = []
        self.X_test_scaled: List[List[float]] = []
        self.y_train: List[Any] = []
        self.y_test: List[Any] = []
        self.y_test_pred: List[Any] = []
        self.y_test_proba: Optional[List[float]] = None

    def _extract_feature_subsets(
        self, raw_records: List[List[Any]]
    ) -> Tuple[List[List[Any]], List[List[Any]]]:
        """Split raw feature rows into numerical and categorical matrices."""
        num_cols = self.schema.numerical_features
        cat_cols = self.schema.categorical_features
        all_features = self.schema.feature_names

        num_indices = [all_features.index(c) for c in num_cols]
        cat_indices = [all_features.index(c) for c in cat_cols]

        num_rows = [[row[i] for i in num_indices] for row in raw_records] if num_indices else [[] for _ in raw_records]
        cat_rows = [[row[i] for i in cat_indices] for row in raw_records] if cat_indices else [[] for _ in raw_records]

        return num_rows, cat_rows

    def fit(
        self,
        raw_X: List[List[Any]],
        y: List[Any],
        test_size: float = 0.2,
    ) -> "MLPipeline":
        """
        Execute leak-free end-to-end training:
        1. Split raw data and targets jointly before any transformation.
        2. Fit imputers, encoder, and scaler strictly on X_train.
        3. Transform X_train and X_test using the frozen transformers.
        4. Fit Scikit-learn estimator on X_train.
        5. Evaluate using custom scratch metrics.
        """
        # Validate task type and targets
        is_classification = "classification" in self.schema.task_type
        if is_classification:
            stratify_target = y
        else:
            stratify_target = None

        # 1. Split BEFORE fitting any transformer
        X_tr, X_te, y_tr, y_te = train_test_split(
            raw_X, y,
            test_size=test_size,
            random_seed=self.random_seed,
            stratify=stratify_target,
        )

        self.y_train = y_tr
        self.y_test = y_te

        # 2. Extract numerical and categorical columns
        num_tr, cat_tr = self._extract_feature_subsets(X_tr)
        num_te, cat_te = self._extract_feature_subsets(X_te)

        # 3. Fit imputers ON TRAIN ONLY
        processed_tr_parts: List[List[List[float]]] = []
        processed_te_parts: List[List[List[float]]] = []

        if self.schema.numerical_features:
            self.num_imputer = SimpleImputer(strategy=self.imputer_strategy_num)
            clean_num_tr = self.num_imputer.fit_transform(num_tr)
            clean_num_te = self.num_imputer.transform(num_te)
            processed_tr_parts.append([[float(v) for v in row] for row in clean_num_tr])
            processed_te_parts.append([[float(v) for v in row] for row in clean_num_te])

        if self.schema.categorical_features:
            self.cat_imputer = SimpleImputer(strategy=self.imputer_strategy_cat)
            clean_cat_tr = self.cat_imputer.fit_transform(cat_tr)
            clean_cat_te = self.cat_imputer.transform(cat_te)

            self.ohe = OneHotEncoder(drop_first=False)
            enc_cat_tr = self.ohe.fit_transform(clean_cat_tr)
            enc_cat_te = self.ohe.transform(clean_cat_te)
            processed_tr_parts.append([[float(v) for v in row] for row in enc_cat_tr])
            processed_te_parts.append([[float(v) for v in row] for row in enc_cat_te])

        # Combine processed parts
        combined_tr: List[List[float]] = []
        combined_te: List[List[float]] = []
        n_tr = len(X_tr)
        n_te = len(X_te)

        for i in range(n_tr):
            row_tr: List[float] = []
            for part in processed_tr_parts:
                row_tr.extend(part[i])
            combined_tr.append(row_tr)

        for i in range(n_te):
            row_te: List[float] = []
            for part in processed_te_parts:
                row_te.extend(part[i])
            combined_te.append(row_te)

        # 4. Fit scaler ON TRAIN ONLY
        if self.scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()

        self.X_train_scaled = self.scaler.fit_transform(combined_tr)
        self.X_test_scaled = self.scaler.transform(combined_te)

        # 5. Fit Estimator
        t0 = time.perf_counter()
        task_str = "classification" if is_classification else "regression"
        self.estimator = get_model(self.model_name, task=task_str, **self.model_params)
        self.estimator.fit(self.X_train_scaled, self.y_train)
        self.training_time_ms = (time.perf_counter() - t0) * 1000.0

        # 6. Inference and Evaluation on Test Set using scratch metrics
        t_inf0 = time.perf_counter()
        self.y_test_pred = list(self.estimator.predict(self.X_test_scaled))
        self.inference_time_ms = (time.perf_counter() - t_inf0) * 1000.0

        if is_classification and hasattr(self.estimator, "predict_proba"):
            try:
                raw_proba = self.estimator.predict_proba(self.X_test_scaled)
                # If binary, extract positive class probability
                if len(raw_proba[0]) == 2:
                    self.y_test_proba = [float(p[1]) for p in raw_proba]
                else:
                    self.y_test_proba = None
            except Exception:
                self.y_test_proba = None

        self.test_metrics = self._compute_metrics(self.y_test, self.y_test_pred, self.y_test_proba)
        self.is_fitted = True
        return self

    def _compute_metrics(
        self,
        y_true: List[Any],
        y_pred: List[Any],
        y_proba: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Compute performance metrics using scratch implementations."""
        metrics: Dict[str, Any] = {}

        if "classification" in self.schema.task_type:
            # Classification metrics from scratch
            metrics["accuracy"] = accuracy_score(y_true, y_pred)
            metrics["precision"] = precision_score(y_true, y_pred, zero_division=0.0)
            metrics["recall"] = recall_score(y_true, y_pred, zero_division=0.0)
            metrics["f1"] = f1_score(y_true, y_pred, zero_division=0.0)
            metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)

            if y_proba is not None:
                try:
                    metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
                except Exception:
                    metrics["roc_auc"] = None
            else:
                metrics["roc_auc"] = None
        else:
            # Regression metrics from scratch
            y_true_f = [float(v) for v in y_true]
            y_pred_f = [float(v) for v in y_pred]
            metrics["mse"] = mean_squared_error(y_true_f, y_pred_f)
            metrics["rmse"] = root_mean_squared_error(y_true_f, y_pred_f)
            metrics["mae"] = mean_absolute_error(y_true_f, y_pred_f)
            metrics["r2"] = r2_score(y_true_f, y_pred_f)

        return metrics

    def transform_records(self, raw_records: List[List[Any]]) -> List[List[float]]:
        """Transform unseen raw feature rows into scaled features using fitted transformers."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before transforming records.")

        num_rows, cat_rows = self._extract_feature_subsets(raw_records)
        processed_parts: List[List[List[float]]] = []

        if self.num_imputer is not None:
            clean_num = self.num_imputer.transform(num_rows)
            processed_parts.append([[float(v) for v in row] for row in clean_num])

        if self.cat_imputer is not None and self.ohe is not None:
            clean_cat = self.cat_imputer.transform(cat_rows)
            enc_cat = self.ohe.transform(clean_cat)
            processed_parts.append([[float(v) for v in row] for row in enc_cat])

        combined: List[List[float]] = []
        for i in range(len(raw_records)):
            row_c: List[float] = []
            for part in processed_parts:
                row_c.extend(part[i])
            combined.append(row_c)

        return self.scaler.transform(combined)

    def predict(self, raw_records: List[List[Any]]) -> List[Any]:
        """Predict outcomes for raw records."""
        scaled = self.transform_records(raw_records)
        return list(self.estimator.predict(scaled))

    def predict_proba(self, raw_records: List[List[Any]]) -> List[Any]:
        """Predict probabilities for raw records if supported."""
        if not hasattr(self.estimator, "predict_proba"):
            raise AttributeError(f"Estimator '{self.model_name}' does not support predict_proba.")
        scaled = self.transform_records(raw_records)
        return self.estimator.predict_proba(scaled).tolist()

    def predict_single(self, record_dict: Dict[str, Any]) -> Any:
        """
        Predict target for a single input dictionary mapping feature names to values.
        """
        row = [record_dict.get(f) for f in self.schema.feature_names]
        preds = self.predict([row])
        return preds[0]

    def predict_single_proba(self, record_dict: Dict[str, Any]) -> Optional[Dict[Any, float]]:
        """Predict class probabilities for a single input dictionary."""
        if not hasattr(self.estimator, "predict_proba"):
            return None
        row = [record_dict.get(f) for f in self.schema.feature_names]
        raw_proba = self.predict_proba([row])[0]
        classes = getattr(self.estimator, "classes_", list(range(len(raw_proba))))
        return {cls_name: float(prob) for cls_name, prob in zip(classes, raw_proba)}

    def cross_validate(self, raw_X: List[List[Any]], y: List[Any], k: int = 5) -> Dict[str, float]:
        """
        Perform leak-free k-fold cross validation.
        In each fold, preprocessors are fitted strictly on the fold's training indices.
        """
        is_classification = "classification" in self.schema.task_type
        if is_classification:
            splitter = StratifiedKFold(n_splits=k, shuffle=True, random_seed=self.random_seed)
        else:
            splitter = KFold(n_splits=k, shuffle=True, random_seed=self.random_seed)

        fold_metrics: List[Dict[str, Any]] = []

        for train_idx, val_idx in splitter.split(raw_X, y):
            X_train_f = [raw_X[i] for i in train_idx]
            y_train_f = [y[i] for i in train_idx]
            X_val_f = [raw_X[i] for i in val_idx]
            y_val_f = [y[i] for i in val_idx]

            # Fit preprocessors on fold training set only
            num_tr, cat_tr = self._extract_feature_subsets(X_train_f)
            num_val, cat_val = self._extract_feature_subsets(X_val_f)

            parts_tr: List[List[List[float]]] = []
            parts_val: List[List[List[float]]] = []

            if self.schema.numerical_features:
                num_imp = SimpleImputer(strategy=self.imputer_strategy_num)
                c_num_tr = num_imp.fit_transform(num_tr)
                c_num_val = num_imp.transform(num_val)
                parts_tr.append([[float(v) for v in r] for r in c_num_tr])
                parts_val.append([[float(v) for v in r] for r in c_num_val])

            if self.schema.categorical_features:
                cat_imp = SimpleImputer(strategy=self.imputer_strategy_cat)
                c_cat_tr = cat_imp.fit_transform(cat_tr)
                c_cat_val = cat_imp.transform(cat_val)
                ohe = OneHotEncoder(drop_first=False)
                e_cat_tr = ohe.fit_transform(c_cat_tr)
                e_cat_val = ohe.transform(c_cat_val)
                parts_tr.append([[float(v) for v in r] for r in e_cat_tr])
                parts_val.append([[float(v) for v in r] for r in e_cat_val])

            comb_tr: List[List[float]] = []
            for i in range(len(X_train_f)):
                row_c: List[float] = []
                for p in parts_tr:
                    row_c.extend(p[i])
                comb_tr.append(row_c)

            comb_val: List[List[float]] = []
            for i in range(len(X_val_f)):
                row_c: List[float] = []
                for p in parts_val:
                    row_c.extend(p[i])
                comb_val.append(row_c)

            scaler = StandardScaler() if self.scaler_type == "standard" else MinMaxScaler()
            X_tr_sc = scaler.fit_transform(comb_tr)
            X_val_sc = scaler.transform(comb_val)

            task_str = "classification" if is_classification else "regression"
            model = get_model(self.model_name, task=task_str, **self.model_params)
            model.fit(X_tr_sc, y_train_f)

            preds = list(model.predict(X_val_sc))
            proba = None
            if is_classification and hasattr(model, "predict_proba"):
                try:
                    p = model.predict_proba(X_val_sc)
                    if len(p[0]) == 2:
                        proba = [float(row[1]) for row in p]
                except Exception:
                    proba = None

            fold_res = self._compute_metrics(y_val_f, preds, proba)
            fold_metrics.append(fold_res)

        # Average numeric metrics across folds
        avg_metrics: Dict[str, float] = {}
        first_res = fold_metrics[0]
        for key, val in first_res.items():
            if isinstance(val, (int, float)):
                scores = [fm[key] for fm in fold_metrics if fm.get(key) is not None]
                avg_metrics[f"cv_{key}_mean"] = sum(scores) / len(scores) if scores else 0.0

        return avg_metrics


class ExperimentSuite:
    """
    Evaluates multiple candidate models on a dataset using the leak-free MLPipeline.
    Generates genuine, empirically computed benchmark comparison reports.
    """

    def __init__(
        self,
        schema: DatasetSchema,
        raw_X: List[List[Any]],
        y: List[Any],
        random_seed: int = 42,
    ) -> None:
        self.schema = schema
        self.raw_X = raw_X
        self.y = y
        self.random_seed = random_seed
        self.pipelines: Dict[str, MLPipeline] = {}
        self.results: Dict[str, Dict[str, Any]] = {}

    def run_all(self, model_names: Optional[List[str]] = None) -> Any:
        """Run all compatible models and build a BenchmarkReport."""
        from edupredict.experiments.benchmark import BenchmarkReport

        task_str = "classification" if "classification" in self.schema.task_type else "regression"
        available_models = list_models(task_str)
        selected_models = model_names or available_models

        report = BenchmarkReport(task_name=f"{self.schema.target_column} ({task_str})")

        for m_name in selected_models:
            pipe = MLPipeline(
                schema=self.schema,
                model_name=m_name,
                random_seed=self.random_seed,
            )
            pipe.fit(self.raw_X, self.y)
            self.pipelines[m_name] = pipe

            # Prepare report entry
            metrics_entry: Dict[str, float] = {}
            for k, v in pipe.test_metrics.items():
                if isinstance(v, (int, float)):
                    metrics_entry[k.upper()] = float(v)
            metrics_entry["TRAIN_TIME_MS"] = round(pipe.training_time_ms, 2)
            metrics_entry["INFER_TIME_MS"] = round(pipe.inference_time_ms, 2)

            self.results[m_name] = {
                "metrics": pipe.test_metrics,
                "train_time_ms": pipe.training_time_ms,
                "infer_time_ms": pipe.inference_time_ms,
                "pipeline": pipe,
            }
            report.add_result(m_name, metrics_entry)

        return report
