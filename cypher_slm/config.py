from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ArtifactPaths:
    root: Path = Path("artifacts")
    raw_data: Path = field(default_factory=lambda: Path("artifacts/raw"))
    processed_data: Path = field(default_factory=lambda: Path("artifacts/processed"))
    synthetic_data: Path = field(default_factory=lambda: Path("artifacts/synthetic"))
    model_outputs: Path = field(default_factory=lambda: Path("artifacts/models"))
    reports: Path = field(default_factory=lambda: Path("artifacts/reports"))

    def ensure(self) -> "ArtifactPaths":
        for path in (
            self.root,
            self.raw_data,
            self.processed_data,
            self.synthetic_data,
            self.model_outputs,
            self.reports,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self


@dataclass(slots=True)
class TrainingConfig:
    base_model: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    output_dir: str = "artifacts/models/smollm2-360m-cypher"
    max_length: int = 768
    learning_rate: float = 2e-4
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 50
    max_steps: int = -1
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    seed: int = 42


@dataclass(slots=True)
class RunConfig:
    artifacts: ArtifactPaths = field(default_factory=ArtifactPaths)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str | None = None
