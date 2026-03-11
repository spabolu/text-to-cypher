from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from neo4j import GraphDatabase

from .data import CypherExample, normalize_cypher


@dataclass(slots=True)
class EvaluationRecord:
    sample_id: str
    model_id: str
    schema_id: str
    question: str
    expected_cypher: str
    generated_cypher: str
    syntax_valid: bool
    execution_success: bool
    result_correct: bool
    latency_seconds: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for record in records:
        normalized.append({key: record[key] for key in sorted(record)})
    return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True))


def compare_result_sets(expected: Iterable[dict[str, Any]], actual: Iterable[dict[str, Any]]) -> bool:
    return normalize_records(expected) == normalize_records(actual)


def execute_cypher(
    uri: str,
    username: str,
    password: str,
    query: str,
    database: str | None = None,
) -> list[dict[str, Any]]:
    driver = GraphDatabase.driver(uri, auth=(username, password))
    with driver.session(database=database) as session:
        result = session.run(query)
        rows = [record.data() for record in result]
    driver.close()
    return rows


def evaluate_examples(
    examples: Iterable[CypherExample],
    model_id: str,
    generator,
    neo4j_uri: str,
    neo4j_username: str,
    neo4j_password: str,
    database: str | None = None,
) -> list[EvaluationRecord]:
    records: list[EvaluationRecord] = []
    for index, example in enumerate(examples):
        started = time.perf_counter()
        generated = generator(example.schema_text, example.question)
        latency = time.perf_counter() - started
        syntax_valid = bool(normalize_cypher(generated))
        execution_success = False
        result_correct = False
        error = None
        try:
            expected_rows = execute_cypher(neo4j_uri, neo4j_username, neo4j_password, example.cypher, database)
            actual_rows = execute_cypher(neo4j_uri, neo4j_username, neo4j_password, generated, database)
            execution_success = True
            result_correct = compare_result_sets(expected_rows, actual_rows)
        except Exception as exc:  # pragma: no cover - depends on live Neo4j
            error = str(exc)
        records.append(
            EvaluationRecord(
                sample_id=f"sample-{index}",
                model_id=model_id,
                schema_id=example.schema_id,
                question=example.question,
                expected_cypher=example.cypher,
                generated_cypher=generated,
                syntax_valid=syntax_valid,
                execution_success=execution_success,
                result_correct=result_correct,
                latency_seconds=latency,
                error=error,
            )
        )
    return records


def save_evaluation_records(records: Iterable[EvaluationRecord], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=True) + "\n")
    return path
