from cypher_slm.evaluation import EvaluationRecord
from cypher_slm.reporting import compare_runs, summarize_comparison, summarize_records


def _record(
    *,
    sample_id: str,
    model_id: str,
    expected_cypher: str,
    generated_cypher: str,
    exact_match: bool,
    syntax_valid: bool,
    execution_success: bool,
    result_correct: bool,
    latency_seconds: float,
) -> EvaluationRecord:
    return EvaluationRecord(
        sample_id=sample_id,
        model_id=model_id,
        schema_id="schema",
        question=f"question-{sample_id}",
        expected_cypher=expected_cypher,
        generated_cypher=generated_cypher,
        exact_match=exact_match,
        syntax_valid=syntax_valid,
        execution_success=execution_success,
        result_correct=result_correct,
        latency_seconds=latency_seconds,
    )


def test_summarize_records_includes_exact_match_rate():
    records = [
        _record(
            sample_id="0",
            model_id="base",
            expected_cypher="MATCH (n) RETURN count(n) AS c",
            generated_cypher="MATCH (n) RETURN count(n) AS c",
            exact_match=True,
            syntax_valid=True,
            execution_success=True,
            result_correct=True,
            latency_seconds=1.0,
        ),
        _record(
            sample_id="1",
            model_id="base",
            expected_cypher="MATCH (n) RETURN n.name AS name",
            generated_cypher="MATCH (n) RETURN n.id AS id",
            exact_match=False,
            syntax_valid=True,
            execution_success=True,
            result_correct=False,
            latency_seconds=2.0,
        ),
    ]
    summary = summarize_records(records)
    summary_map = dict(zip(summary["metric"], summary["value"]))
    assert summary_map["total_examples"] == 2.0
    assert summary_map["exact_match_rate"] == 0.5
    assert summary_map["execution_accuracy"] == 0.5


def test_compare_runs_highlights_execution_improvements():
    baseline = [
        _record(
            sample_id="0",
            model_id="base",
            expected_cypher="MATCH (n) RETURN count(n) AS c",
            generated_cypher="MATCH (n) RETURN count(n) AS c",
            exact_match=True,
            syntax_valid=True,
            execution_success=True,
            result_correct=True,
            latency_seconds=1.0,
        ),
        _record(
            sample_id="1",
            model_id="base",
            expected_cypher="MATCH (n) RETURN n.name AS name",
            generated_cypher="MATCH (n) RETURN n.id AS id",
            exact_match=False,
            syntax_valid=True,
            execution_success=True,
            result_correct=False,
            latency_seconds=2.0,
        ),
    ]
    tuned = [
        _record(
            sample_id="0",
            model_id="tuned",
            expected_cypher="MATCH (n) RETURN count(n) AS c",
            generated_cypher="MATCH (n) RETURN count(n) AS c",
            exact_match=True,
            syntax_valid=True,
            execution_success=True,
            result_correct=True,
            latency_seconds=1.2,
        ),
        _record(
            sample_id="1",
            model_id="tuned",
            expected_cypher="MATCH (n) RETURN n.name AS name",
            generated_cypher="MATCH (n) RETURN n.name AS name",
            exact_match=True,
            syntax_valid=True,
            execution_success=True,
            result_correct=True,
            latency_seconds=2.1,
        ),
    ]
    comparison_df = compare_runs(baseline, tuned)
    comparison_summary = summarize_comparison(comparison_df)
    summary_map = dict(zip(comparison_summary["metric"], comparison_summary["value"]))
    assert summary_map["execution_accuracy_delta"] == 0.5
    assert summary_map["exact_match_rate_delta"] == 0.5
    assert summary_map["new_execution_wins"] == 1.0
    assert summary_map["execution_regressions"] == 0.0
