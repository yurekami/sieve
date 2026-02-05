"""
Cartridges KV Cache Training for MTOB

Trains a KV cache (Cartridge) on the synthesized MTOB data using
the Self-Study context distillation objective.

This matches the configuration from:
cartridges/examples/benchmarks/mtob/mtob_train.py

Usage:
    NUM_TOKENS=4096 MODEL=qwen8b python -m mtob.cartridges_baseline.cartridges_train

Or run via pydrantic:
    NUM_TOKENS=4096 MODEL=qwen8b python mtob/cartridges_baseline/cartridges_train.py
"""

import os

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.data.mtob.evals import MTOBKalamangToEnglishGenerateDataset
from cartridges.initialization import KVFromText
from cartridges.train import GenerationEvalConfig, TrainConfig
from cartridges.models.config import HFModelConfig
from cartridges.datasets import DataSource, TrainDataset
from cartridges.utils.wandb import WandBConfig


NUM_TOKENS = int(os.environ.get("NUM_TOKENS", "4096"))

MODEL = os.environ.get("MODEL", "qwen8b")

# Check for custom data sources from environment
CUSTOM_DATA_SOURCES = os.environ.get("DATA_SOURCES", "")

if MODEL == "qwen4b":
    from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM

    if CUSTOM_DATA_SOURCES:
        data_sources = [s.strip() for s in CUSTOM_DATA_SOURCES.split(",") if s.strip()]
    else:
        data_sources = [
            "hazyresearch/m07d28_mtob_synthesize_qwen3-4b_n65536-0",
            "hazyresearch/m07d28_mtob_synthesize_qwen3-4b_n65536-1",
        ]
    model = HFModelConfig(
        pretrained_model_name_or_path="Qwen/Qwen3-4B",
        model_cls=FlexQwen3ForCausalLM,
    )
elif MODEL == "qwen8b":
    from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM

    if CUSTOM_DATA_SOURCES:
        data_sources = [s.strip() for s in CUSTOM_DATA_SOURCES.split(",") if s.strip()]
    else:
        # Use the 4B data for now - 8B data not yet released by hazyresearch
        data_sources = [
            "hazyresearch/m07d28_mtob_synthesize_qwen3-4b_n65536-0",
            "hazyresearch/m07d28_mtob_synthesize_qwen3-4b_n65536-1",
        ]
    model = HFModelConfig(
        pretrained_model_name_or_path="Qwen/Qwen3-8B",
        model_cls=FlexQwen3ForCausalLM,
    )
elif MODEL == "llama":
    from cartridges.models.llama.modeling_llama import FlexLlamaForCausalLM

    if CUSTOM_DATA_SOURCES:
        data_sources = [s.strip() for s in CUSTOM_DATA_SOURCES.split(",") if s.strip()]
    else:
        data_sources = [
            "hazyresearch/m07d28_mtob_synthesize_llama-3.2-3b_n65536-0",
            "hazyresearch/m07d28_mtob_synthesize_llama-3.2-3b_n65536-1",
        ]
    model = HFModelConfig(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
        model_cls=FlexLlamaForCausalLM,
    )
else:
    raise ValueError(f"Invalid model: {MODEL}. Use 'llama', 'qwen4b', or 'qwen8b'")


def get_data_sources():
    """Convert data source strings to DataSource objects."""
    sources = []
    for source in data_sources:
        source = source.strip()
        if not source:
            continue
        # Detect if it's a local path or HuggingFace dataset
        # DataSource only accepts: 'local', 'wandb', or 'hf'
        if source.startswith("/") or source.startswith("."):
            sources.append(DataSource(path=source, type="local"))
        else:
            sources.append(DataSource(path=source, type="hf"))
    return sources


configs = []
for lr in [2e-2]:
    config = TrainConfig(
        model=model,
        kv_cache_initializer=KVFromText.Config(
            max_tokens=NUM_TOKENS,
        ),
        lr=lr,
        epochs=1,
        global_batch_size=32,
        dataset=TrainDataset.Config(
            data_sources=get_data_sources(),
            top_k_logits=20,
            packed_seq_length=2048,
            packing_mode="truncate",
        ),
        save_every_n_steps=512,
        generate_eval_every_n_steps=128,
        generate_evals=[
            GenerationEvalConfig(
                name_for_wandb="mtob-ke-test",
                dataset=MTOBKalamangToEnglishGenerateDataset.Config(use_cot=False),
                batch_size=16,
                generate_max_new_tokens=128,
                num_samples=1,
                temperature=0,
            ),
        ],
        loss_eval_every_n_steps=512,
        loss_evals=[],
        distributed_backend="gloo",
        wandb=WandBConfig(tags=["train", "mtob"]),
        output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
        name=FormatStringVariable("mtob_train_lr{lr}_toks{kv_cache_initializer.max_tokens}"),
    )
    configs.append(config)


if __name__ == "__main__":
    pydrantic.main(configs)
