"""
Compliance scanner verifying strict zero-external-ML-dependency policy.

Scans the AST (Abstract Syntax Tree) of all source files in edupredict/ to guarantee
no prohibited third-party ML, numerical, or data processing packages are imported.
"""

import unittest
import ast
import os

PROHIBITED_MODULES = {
    "numpy",
    "pandas",
    "sklearn",
    "scipy",
    "torch",
    "tensorflow",
    "keras",
    "statsmodels",
    "seaborn",
    "jax",
}


class TestDependencyCompliance(unittest.TestCase):

    def test_no_prohibited_imports_in_edupredict(self):
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
                                if top_level in PROHIBITED_MODULES:
                                    violations.append(f"{relpath}: import {alias.name}")
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                top_level = node.module.split(".")[0]
                                if top_level in PROHIBITED_MODULES:
                                    violations.append(f"{relpath}: from {node.module} import ...")

        self.assertEqual(
            violations,
            [],
            f"Prohibited dependency imports detected in edupredict package:\n" + "\n".join(violations)
        )


if __name__ == "__main__":
    unittest.main()
