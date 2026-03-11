from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .evaluation import EvaluationRecord


def records_to_dataframe(records: Iterable[EvaluationRecord]) -> pd.DataFrame:
    return pd.DataFrame([record.to_dict() for record in records])


def summarize_records(records: Iterable[EvaluationRecord]) -> pd.DataFrame:
    df = records_to_dataframe(records)
    if df.empty:
        return pd.DataFrame(
            [{"metric": "execution_accuracy", "value": 0.0}, {"metric": "syntax_valid_rate", "value": 0.0}]
        )
    summary = pd.DataFrame(
        [
            {"metric": "execution_accuracy", "value": float(df["result_correct"].mean())},
            {"metric": "syntax_valid_rate", "value": float(df["syntax_valid"].mean())},
            {"metric": "execution_success_rate", "value": float(df["execution_success"].mean())},
            {"metric": "average_latency_seconds", "value": float(df["latency_seconds"].mean())},
        ]
    )
    return summary


def write_markdown_report(summary: pd.DataFrame, path: str | Path, title: str = "Benchmark Summary") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    markdown = [f"# {title}", "", summary.to_markdown(index=False), ""]
    path.write_text("\n".join(markdown), encoding="utf-8")
    return path
