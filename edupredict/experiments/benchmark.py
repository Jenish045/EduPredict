"""
Benchmark evaluation and tabular reporting for model comparisons.
"""

from typing import Dict, List, Any


class BenchmarkReport:
    """
    Collects evaluation metrics across multiple candidate models and generates
    formatted comparison tables.
    """

    def __init__(self, task_name: str) -> None:
        self.task_name = task_name
        self.records: List[Dict[str, Any]] = []

    def add_result(self, model_name: str, metrics: Dict[str, float]) -> None:
        """Add evaluation metrics for a model."""
        entry = {"Model": model_name}
        entry.update(metrics)
        self.records.append(entry)

    def to_markdown_table(self) -> str:
        """Convert recorded benchmark metrics to a GitHub-flavored Markdown table."""
        if not self.records:
            return "No benchmark results recorded."

        # Collect unique metric keys
        all_keys = ["Model"]
        for r in self.records:
            for k in r:
                if k not in all_keys:
                    all_keys.append(k)

        lines = [f"### Benchmark: {self.task_name}\n"]
        header_row = "| " + " | ".join(all_keys) + " |"
        sep_row = "| " + " | ".join(["---"] * len(all_keys)) + " |"
        lines.append(header_row)
        lines.append(sep_row)

        for r in self.records:
            vals = []
            for k in all_keys:
                v = r.get(k, "N/A")
                if isinstance(v, float):
                    vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")

        return "\n".join(lines)

    def print_table(self) -> None:
        """Print benchmark comparison to console."""
        print(self.to_markdown_table())
