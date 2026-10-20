"""
Compliance scanner verifying strict dependency boundaries.

Scans the Abstract Syntax Tree (AST) of all source files in edupredict/ to guarantee:
1. Prohibited numerical/data packages (numpy, pandas, scipy, torch, tensorflow, etc.)
   are NEVER imported anywhere in edupredict/.
2. Scikit-learn is strictly forbidden in custom modules:
   - edupredict/math/
   - edupredict/preprocessing/
   - edupredict/metrics/
   - edupredict/losses/
   - edupredict/optimization/
   - edupredict/data/
   - edupredict/model_selection/
3. Scikit-learn is permitted ONLY for predictive estimators in edupredict/models/
   and edupredict/pipeline/, and may NOT replace preprocessing, splitting, or metrics.
"""

import unittest
import ast
import os

GLOBAL_PROHIBITED_MODULES = {
    "numpy",
    "pandas",
    "scipy",
    "torch",
    "tensorflow",
    "keras",
    "statsmodels",
    "seaborn",
    "jax",
}

SCRATCH_ONLY_DIRECTORIES = {
    "math",
    "preprocessing",
    "metrics",
    "losses",
    "optimization",
    "data",
    "model_selection",
}


class TestDependencyCompliance(unittest.TestCase):

    def test_global_prohibited_libraries_never_imported(self):
        """Verify NumPy, Pandas, SciPy, Torch, etc. are never imported anywhere in edupredict."""
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "edupredict"))
        violations = []

        for dirpath, _, filenames in os.walk(root_dir):
            for file in filenames:
                if file.endswith(".py"):
                    filepath = os.path.join(dirpath, file)
                    relpath = os.path.relpath(filepath, root_dir)

                    with open(filepath, "r", encoding="utf-8") as f:
                        tree = ast.parse(f.read(), filename=filepath)

                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                top_level = alias.name.split(".")[0]
                                if top_level in GLOBAL_PROHIBITED_MODULES:
                                    violations.append(f"{relpath}: import {alias.name}")
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                top_level = node.module.split(".")[0]
                                if top_level in GLOBAL_PROHIBITED_MODULES:
                                    violations.append(f"{relpath}: from {node.module} import ...")

        self.assertEqual(
            violations,
            [],
            "Prohibited global dependency imports detected in edupredict package:\n" + "\n".join(violations)
        )

    def test_sklearn_strictly_isolated_to_model_estimators(self):
        """
        Verify Scikit-learn is NEVER imported in custom scratch modules
        (math, preprocessing, metrics, losses, optimization, data, model_selection).
        """
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "edupredict"))
        violations = []

        for scratch_dir in SCRATCH_ONLY_DIRECTORIES:
            target_path = os.path.join(root_dir, scratch_dir)
            if not os.path.exists(target_path):
                continue

            for dirpath, _, filenames in os.walk(target_path):
                for file in filenames:
                    if file.endswith(".py"):
                        filepath = os.path.join(dirpath, file)
                        relpath = os.path.relpath(filepath, root_dir)

                        with open(filepath, "r", encoding="utf-8") as f:
                            tree = ast.parse(f.read(), filename=filepath)

                        for node in ast.walk(tree):
                            if isinstance(node, ast.Import):
                                for alias in node.names:
                                    top_level = alias.name.split(".")[0]
                                    if top_level == "sklearn":
                                        violations.append(f"{relpath}: import {alias.name}")
                            elif isinstance(node, ast.ImportFrom):
                                if node.module:
                                    top_level = node.module.split(".")[0]
                                    if top_level == "sklearn":
                                        violations.append(f"{relpath}: from {node.module} import ...")

        self.assertEqual(
            violations,
            [],
            "Scikit-learn imported inside custom from-scratch modules:\n" + "\n".join(violations)
        )

    def test_sklearn_utilities_never_replace_preprocessing_or_metrics(self):
        """
        Verify sklearn.preprocessing, sklearn.metrics, or sklearn.model_selection
        are NEVER imported anywhere in edupredict.
        """
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "edupredict"))
        forbidden_sklearn_submodules = {
            "sklearn.preprocessing",
            "sklearn.metrics",
            "sklearn.model_selection",
        }
        violations = []

        for dirpath, _, filenames in os.walk(root_dir):
            for file in filenames:
                if file.endswith(".py"):
                    filepath = os.path.join(dirpath, file)
                    relpath = os.path.relpath(filepath, root_dir)

                    with open(filepath, "r", encoding="utf-8") as f:
                        tree = ast.parse(f.read(), filename=filepath)

                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                for forbidden in forbidden_sklearn_submodules:
                                    if alias.name.startswith(forbidden):
                                        violations.append(f"{relpath}: import {alias.name}")
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                for forbidden in forbidden_sklearn_submodules:
                                    if node.module.startswith(forbidden):
                                        violations.append(f"{relpath}: from {node.module} import ...")

        self.assertEqual(
            violations,
            [],
            "Forbidden Scikit-learn utility modules imported in edupredict:\n" + "\n".join(violations)
        )


if __name__ == "__main__":
    unittest.main()
