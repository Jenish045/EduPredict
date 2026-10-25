"""
EduPredict Main Experiment Entrypoint.

Executes reproducible end-to-end Classical Machine Learning experiments
using custom from-scratch model estimators (Linear Regression, KNN, Decision Tree),
first-principles preprocessing, custom linear algebra, and scratch metrics.
"""

import argparse
from edupredict.experiments.runner import run_full_experiment


def main():
    parser = argparse.ArgumentParser(
        description="EduPredict: Classical ML Framework"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to CSV dataset (generates synthetic student dataset if omitted)"
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target column name for custom dataset"
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        choices=["regression", "binary_classification", "multiclass_classification"],
        help="Task type for custom dataset (inferred if omitted)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts",
        help="Directory to save generated charts and figures"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()
    results = run_full_experiment(
        csv_path=args.data,
        output_dir=args.output_dir,
        random_seed=args.seed,
        target_column=args.target,
        task_type=args.task,
    )
    print("\nExperiment finished with status: SUCCESS")


if __name__ == "__main__":
    main()
