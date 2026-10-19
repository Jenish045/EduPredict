"""
EduPredict Main Experiment Entrypoint.

Executes reproducible end-to-end Classical Machine Learning experiments
predicting student academic performance and academic risk from scratch.
"""

import sys
import argparse
from edupredict.experiments.runner import run_full_experiment


def main():
    parser = argparse.ArgumentParser(
        description="EduPredict: Classical ML Framework Implemented From Scratch"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to student performance CSV dataset (generates synthetic if omitted)"
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
        random_seed=args.seed
    )
    print("\nExperiment finished with status: SUCCESS")


if __name__ == "__main__":
    main()
