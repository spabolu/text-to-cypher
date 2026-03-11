from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from datasets import DatasetDict
from peft import AutoPeftModelForCausalLM, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from trl import SFTConfig, SFTTrainer

from .config import TrainingConfig
from .data import CypherExample, examples_to_dataset
from .prompts import SYSTEM_PROMPT, render_user_prompt


def current_cuda_dtype():
    ensure_cuda_available()
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def build_quantization_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=current_cuda_dtype(),
        bnb_4bit_use_double_quant=True,
    )


def build_lora_config(config: TrainingConfig) -> LoraConfig:
    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )


def ensure_cuda_available() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU required. This project intentionally supports only the industry-standard CUDA + QLoRA path."
        )


def configure_cuda_backend() -> None:
    ensure_cuda_available()
    torch.set_float32_matmul_precision("high")


def build_sft_config(config: TrainingConfig) -> SFTConfig:
    configure_cuda_backend()
    bf16 = torch.cuda.is_bf16_supported()
    return SFTConfig(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        bf16=bf16,
        fp16=not bf16,
        max_length=config.max_length,
        save_total_limit=2,
        report_to=[],
        completion_only_loss=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        seed=config.seed,
    )


def build_dataset_dict(examples: list[CypherExample]) -> DatasetDict:
    grouped: dict[str, list[CypherExample]] = {"train": [], "validation": [], "test": []}
    for example in examples:
        grouped.setdefault(example.split, []).append(example)
    return DatasetDict({split: examples_to_dataset(rows) for split, rows in grouped.items() if rows})


def _resolve_model_dtype():
    return current_cuda_dtype()


def load_model_and_tokenizer(model_name: str, quantized: bool = True):
    configure_cuda_backend()
    model_path = Path(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    quantization_config = build_quantization_config() if quantized else None
    if model_path.exists() and (model_path / "adapter_config.json").exists():
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=_resolve_model_dtype(),
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype=_resolve_model_dtype(),
        )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return model, tokenizer


def train_qlora(examples: list[CypherExample], config: TrainingConfig) -> SFTTrainer:
    configure_cuda_backend()
    datasets = build_dataset_dict(examples)
    model, tokenizer = load_model_and_tokenizer(
        config.base_model,
        quantized=True,
    )
    model.config.use_cache = False
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=datasets["train"],
        eval_dataset=datasets.get("validation"),
        peft_config=build_lora_config(config),
        args=build_sft_config(config),
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    return trainer


def build_generation_pipeline(model_name_or_path: str):
    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path,
        quantized=False,
    )
    task = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    task.model.generation_config.max_length = None
    return task


def generate_query(generator, schema_text: str, question: str, max_new_tokens: int = 160) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_user_prompt(schema_text, question)},
    ]
    outputs = generator(messages, max_new_tokens=max_new_tokens, do_sample=False)
    generated_payload = outputs[0]["generated_text"]
    if isinstance(generated_payload, list):
        generated = generated_payload[-1]["content"]
    else:
        generated = str(generated_payload)
    return generated.strip()


def export_training_config(config: TrainingConfig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(asdict(config)), encoding="utf-8")
    return path
