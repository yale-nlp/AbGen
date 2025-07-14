import argparse
import json
import logging
import os
from typing import Dict, List

import dotenv
import vllm
from huggingface_hub import login

# Evaluation‑specific prompt builders
from utils.evaluation_prompt import (
    build_importance_single_message,
    build_faithfulness_single_message,
    build_soundness_single_message,
    build_overall_single_message,
)

dotenv.load_dotenv()
login(token=os.getenv("HF_TOKEN"))

def _parse_response(text: str) -> Dict[str, str | int]:
    text = text.strip()

    # Remove a chain‑of‑thought section if the model included one
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    return text


def _build_messages(data: List[Dict], fn) -> List[List[Dict]]:
    """Convert each example into a *conversation* expected by vLLM.chat."""
    return [fn(example) for example in data]


# Mapping from aspect → prompt‑builder ------------------------------------------------
ASPECT_TO_BUILDER = {
    "importance": build_importance_single_message,
    "faithfulness": build_faithfulness_single_message,
    "soundness": build_soundness_single_message,
    "overall": build_overall_single_message,
}

# -----------------------------------------------------------------------------
# Main script
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run vLLM evaluation on a dataset (importance / faithfulness / soundness)."
    )
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output JSON file")
    parser.add_argument("--model", type=str, required=True, help="HF model name or local path")
    parser.add_argument("--tensor_parallel_size", type=int, default=2, help="Tensor parallelism for vLLM")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Generation length")
    args = parser.parse_args()

    # I/O setup ----------------------------------------------------------------
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    print(f"Saving outputs to {args.output_file}")

    with open(args.input_file, "r") as fp:
        data: List[Dict] = json.load(fp)
    data = data

    # A working copy that we will enrich with evaluation scores
    enriched_data = [dict(example) for example in data]

    # Initialise the LLM --------------------------------------------------------
    llm = vllm.LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.98,
        max_num_seqs=128,
        enforce_eager=True,
    )

    sampling_params = vllm.SamplingParams(
        temperature=1,  # determinism is nice for evaluation
        max_tokens=args.max_tokens,
    )

    # Run evaluation for each aspect ------------------------------------------
    for aspect, builder in ASPECT_TO_BUILDER.items():

        msgs = _build_messages(data, builder)
        outputs = llm.chat(msgs, sampling_params, use_tqdm=True)

        for example, generation in zip(enriched_data, outputs):
            raw_text = generation.outputs[0].text
            example[aspect] = _parse_response(raw_text)

    # Persist results ----------------------------------------------------------
    with open(args.output_file, "w") as fp:
        json.dump(enriched_data, fp, indent=4)

    print("Done!")
