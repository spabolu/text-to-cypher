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
            [
                {"metric": "total_examples", "value": 0.0},
                {"metric": "exact_match_rate", "value": 0.0},
                {"metric": "execution_accuracy", "value": 0.0},
                {"metric": "syntax_valid_rate", "value": 0.0},
            ]
        )
    summary = pd.DataFrame(
        [
            {"metric": "total_examples", "value": float(len(df))},
            {"metric": "exact_match_rate", "value": float(df["exact_match"].mean())},
            {"metric": "execution_accuracy", "value": float(df["result_correct"].mean())},
            {"metric": "syntax_valid_rate", "value": float(df["syntax_valid"].mean())},
            {"metric": "execution_success_rate", "value": float(df["execution_success"].mean())},
            {"metric": "average_latency_seconds", "value": float(df["latency_seconds"].mean())},
        ]
    )
    return summary


def compare_runs(
    baseline_records: Iterable[EvaluationRecord],
    tuned_records: Iterable[EvaluationRecord],
) -> pd.DataFrame:
    baseline_df = records_to_dataframe(baseline_records).rename(
        columns={
            "model_id": "baseline_model_id",
            "generated_cypher": "baseline_generated_cypher",
            "exact_match": "baseline_exact_match",
            "syntax_valid": "baseline_syntax_valid",
            "execution_success": "baseline_execution_success",
            "result_correct": "baseline_result_correct",
            "latency_seconds": "baseline_latency_seconds",
            "error": "baseline_error",
        }
    )
    tuned_df = records_to_dataframe(tuned_records).rename(
        columns={
            "model_id": "tuned_model_id",
            "generated_cypher": "tuned_generated_cypher",
            "exact_match": "tuned_exact_match",
            "syntax_valid": "tuned_syntax_valid",
            "execution_success": "tuned_execution_success",
            "result_correct": "tuned_result_correct",
            "latency_seconds": "tuned_latency_seconds",
            "error": "tuned_error",
        }
    )
    merged = baseline_df.merge(
        tuned_df,
        on=["sample_id", "schema_id", "question", "expected_cypher"],
        how="inner",
        validate="one_to_one",
    )
    merged["execution_improved"] = (
        merged["tuned_result_correct"].astype(int) - merged["baseline_result_correct"].astype(int)
    )
    merged["exact_match_improved"] = (
        merged["tuned_exact_match"].astype(int) - merged["baseline_exact_match"].astype(int)
    )
    merged["latency_delta_seconds"] = merged["tuned_latency_seconds"] - merged["baseline_latency_seconds"]
    return merged


def summarize_comparison(comparison_df: pd.DataFrame) -> pd.DataFrame:
    if comparison_df.empty:
        return pd.DataFrame(
            [
                {"metric": "execution_accuracy_delta", "value": 0.0},
                {"metric": "exact_match_rate_delta", "value": 0.0},
                {"metric": "new_execution_wins", "value": 0.0},
            ]
        )
    return pd.DataFrame(
        [
            {"metric": "execution_accuracy_delta", "value": float(comparison_df["execution_improved"].mean())},
            {"metric": "exact_match_rate_delta", "value": float(comparison_df["exact_match_improved"].mean())},
            {
                "metric": "new_execution_wins",
                "value": float((comparison_df["execution_improved"] > 0).sum()),
            },
            {
                "metric": "execution_regressions",
                "value": float((comparison_df["execution_improved"] < 0).sum()),
            },
            {
                "metric": "average_latency_delta_seconds",
                "value": float(comparison_df["latency_delta_seconds"].mean()),
            },
        ]
    )


def write_markdown_report(summary: pd.DataFrame, path: str | Path, title: str = "Benchmark Summary") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    markdown = [f"# {title}", "", summary.to_markdown(index=False), ""]
    path.write_text("\n".join(markdown), encoding="utf-8")
    return path
