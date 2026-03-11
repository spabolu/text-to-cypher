from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from datasets import Dataset, load_dataset

from .prompts import SYSTEM_PROMPT, render_user_prompt

PUBLIC_DATASETS: tuple[str, ...] = (
    "vedana17/text-to-cypher",
    "iprahara/text_to_cypher",
)


@dataclass(slots=True)
class CypherExample:
    schema_id: str
    schema_text: str
    question: str
    cypher: str
    source: str
    difficulty: str = "unknown"
    split: str = "train"
    result_signature: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_cypher(query: str) -> str:
    query = query.replace("\r", "\n")
    query = re.sub(r"`+", "`", query)
    query = re.sub(r"[ \t]+", " ", query)
    query = re.sub(r"\s*\n\s*", "\n", query)
    query = query.strip().rstrip(";")
    return query


def canonical_example_key(example: CypherExample) -> tuple[str, str, str]:
    return (
        example.schema_id,
        normalize_whitespace(example.question).lower(),
        normalize_cypher(example.cypher).lower(),
    )


def infer_difficulty(query: str) -> str:
    lowered = query.lower()
    if any(token in lowered for token in ("optional match", "collect(", "avg(", "sum(", "count(", "order by", "limit")):
        return "medium"
    if lowered.count("-") > 2 or "where" in lowered:
        return "medium"
    return "easy"


def _extract_field(record: dict, candidates: Sequence[str], default: str = "") -> str:
    for key in candidates:
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    return default


def _load_dataset_split(dataset_name: str):
    try:
        dataset = load_dataset(dataset_name)
    except Exception as exc:  # pragma: no cover - network/runtime dependent
        print(f"Skipping {dataset_name}: {exc}")
        return []
    splits = []
    for split_name, split in dataset.items():
        for row in split:
            splits.append((split_name, row))
    return splits


def load_public_examples(dataset_names: Sequence[str] = PUBLIC_DATASETS) -> list[CypherExample]:
    examples: list[CypherExample] = []
    for dataset_name in dataset_names:
        for split_name, row in _load_dataset_split(dataset_name):
            question = _extract_field(row, ("question", "text", "instruction", "prompt", "input"))
            cypher = _extract_field(row, ("cypher", "query", "output", "answer", "completion"))
            schema_text = _extract_field(
                row,
                ("schema", "schema_text", "graph_schema", "context"),
                default="Schema not provided in public source.",
            )
            if not question or not cypher:
                continue
            example = CypherExample(
                schema_id=_extract_field(row, ("schema_id",), default=dataset_name.replace("/", "_")),
                schema_text=normalize_whitespace(schema_text),
                question=normalize_whitespace(question),
                cypher=normalize_cypher(cypher),
                source=dataset_name,
                difficulty=infer_difficulty(cypher),
                split=split_name,
            )
            examples.append(example)
    return deduplicate_examples(examples)


def deduplicate_examples(examples: Iterable[CypherExample]) -> list[CypherExample]:
    deduped: list[CypherExample] = []
    seen: set[tuple[str, str, str]] = set()
    for example in examples:
        key = canonical_example_key(example)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(example)
    return deduped


def split_examples(
    examples: Sequence[CypherExample],
    held_out_schema_ids: Sequence[str] | None = None,
    validation_fraction: float = 0.1,
) -> list[CypherExample]:
    held_out = set(held_out_schema_ids or [])
    assigned: list[CypherExample] = []
    seen_count = 0
    for example in examples:
        copy = CypherExample(**example.to_dict())
        if copy.schema_id in held_out:
            copy.split = "test"
        else:
            seen_count += 1
            copy.split = "validation" if seen_count % max(int(1 / max(validation_fraction, 0.01)), 2) == 0 else "train"
        assigned.append(copy)
    return assigned


def build_training_corpus(
    public_examples: Sequence[CypherExample],
    synthetic_examples: Sequence[CypherExample],
    held_out_schema_ids: Sequence[str] | None = None,
    validation_fraction: float = 0.1,
) -> list[CypherExample]:
    combined = deduplicate_examples([*public_examples, *synthetic_examples])
    return split_examples(
        combined,
        held_out_schema_ids=held_out_schema_ids,
        validation_fraction=validation_fraction,
    )


def examples_to_dataset(examples: Sequence[CypherExample]) -> Dataset:
    rows = []
    for example in examples:
        rows.append(
            {
                **example.to_dict(),
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": render_user_prompt(example.schema_text, example.question)},
                ],
                "completion": [{"role": "assistant", "content": example.cypher}],
            }
        )
    return Dataset.from_list(rows)


def save_jsonl(examples: Sequence[CypherExample], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), ensure_ascii=True) + "\n")
    return path


def load_jsonl(path: str | Path) -> list[CypherExample]:
    path = Path(path)
    rows: list[CypherExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(CypherExample(**json.loads(line)))
    return rows
