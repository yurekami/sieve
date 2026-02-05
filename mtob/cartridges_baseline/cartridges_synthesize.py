"""
Cartridges Self-Study Synthesis for MTOB

Replicates the paper's Self-Study data synthesis approach using Cartridges' 
existing MTOBResource and SelfStudySynthesizer.

This matches the configuration from:
cartridges/examples/benchmarks/mtob/mtob_synthesize.py

Usage:
    python -m mtob.cartridges_baseline.cartridges_synthesize

Or run via pydrantic:
    python mtob/cartridges_baseline/cartridges_synthesize.py
"""

import os

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.clients.openai import OpenAIClient
from cartridges.data.mtob.resources import MTOBResource
from cartridges.synthesize import SynthesizeConfig
from cartridges.synthesizers.self_study import SelfStudySynthesizer
from cartridges.utils.wandb import WandBConfig


# Use OpenAI-compatible client for SGLang server
# This matches the approach that works in retail
client = OpenAIClient.Config(
    base_url=os.environ.get("CARTRIDGES_SGLANG_URL", "http://localhost:8000") + "/v1",
    model_name=os.environ.get("CARTRIDGES_MODEL_NAME", "Qwen/Qwen3-8B"),
    api_key="EMPTY",
)

# Setup - use latex_and_sentences to match the paper exactly
# "latex_and_sentences" uses full LaTeX book (as in the paper)
# "medium_and_sentences" uses medium version
SETUP = os.environ.get("MTOB_SETUP", "latex_and_sentences")
PROB_THINKING = float(os.environ.get("CARTRIDGES_PROB_THINKING", "0.0"))

config = SynthesizeConfig(
    synthesizer=SelfStudySynthesizer.Config(
        client=client,
        max_rounds=1,
        prob_thinking=PROB_THINKING,
        use_tools_a=False,
        use_tools_b=False,
        tools=[],
        resources=[
            MTOBResource.Config(
                seed_prompts=[
                    "structuring",
                    "summarization",
                    "question",
                    "use_case",
                    "creative",
                ],
                setup=SETUP,
            )
        ],
    ),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    num_samples=int(os.environ.get("CARTRIDGES_NUM_SAMPLES", "65536")),
    batch_size=32,
    max_num_batches_in_parallel=16,
    name=FormatStringVariable("mtob_synthesize_n{num_samples}"),
    run_id=FormatStringVariable("{name}"),
    wandb=WandBConfig(tags=["mtob_synthesis"]),
    upload_to_wandb=False,
    save_wandb_preview=False,
)


if __name__ == "__main__":
    pydrantic.main([config])
