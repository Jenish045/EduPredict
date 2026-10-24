"""
Compliance scanner verifying strict zero-dependency boundaries.

Scans the Abstract Syntax Tree (AST) of all source files in edupredict/ to guarantee:
1. Zero third-party machine learning / numerical libraries (numpy, pandas, scipy,
   sklearn, torch, tensorflow, etc.) anywhere in the entire edupredict package.
2. Zero standard library math and statistics modules anywhere in the edupredict package
   (all algorithms and math are written from absolute first principles).
"""

import unittest
import ast
import os

GLOBAL_PROHIBITED_MODULES = {
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "torch",
    "tensorflow",
    "keras",
    "statsmodels",
    "seaborn",
    "jax",
    "math",
    "statistics",
}


class TestDependencyCompliance(unittest.TestCase):

    def test_global_prohibited_libraries_never_imported(self):
        """
        Verify NumPy, Pandas, SciPy, Scikit-learn, Torch, math, statistics, etc.
        are NEVER imported anywhere in edupredict.
        """
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
            "Prohibited dependency imports detected in edupredict package:\n" + "\n".join(violations),
        )


if __name__ == "__main__":
    unittest.main()
