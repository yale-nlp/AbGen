import datasets
import argparse
import logging
import json, os
import dotenv
from huggingface_hub import login
from datasets import Dataset, DatasetDict
import sys
from typing import List, Dict
from utils.evaluation_prompt import *
from utils.api import process_data, client_dict
from tqdm.asyncio import tqdm_asyncio
import asyncio
from pydantic import BaseModel

# Example: python meta_evaluation/opensource_api_meta_evaluation.py --model DeepSeek-V3-0324

# Load environment variables from .env file
dotenv.load_dotenv()


async def _generate_response(data: List[Dict], model_name: str, build_messages, client_dict) -> List[Dict]:
    """LLM‑powered classification wrapper."""

    return await process_data(
        data,
        client_dict=client_dict,
        build_messages=build_messages,
        response_model=None,
        enable_structured_output=False,
        model=model_name,
        max_concurrency=32,
        max_tokens=20480,
    )

aspect_dict = {
    "importance": build_importance_single_message,
    "faithfulness": build_faithfulness_single_message,
    "soundness": build_soundness_single_message,
    "overall": build_overall_single_message,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vLLM inference on a dataset.")
    parser.add_argument("--input_file", type=str, default="human_evaluation.json", help="Path to the input JSON file")
    parser.add_argument("--output_dir", type=str, default="meta_evaluation_outputs/raw_outputs", help="Directory to save the output JSON file")
    parser.add_argument("--model", type=str, required=True, help="api model name")
    parser.add_argument("--max_tokens", type=int, default=1024)
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    data = json.load(open(args.input_file, "r"))
    # hard copy
    orig_data = [dict(example) for example in data]

    for aspect in aspect_dict.keys():
        build_message_func = aspect_dict[aspect]
        outputs = asyncio.run(_generate_response(
            data,
            model_name = args.model,
            client_dict = client_dict,
            build_messages = build_message_func,
        ))

        for example, output in zip(orig_data, outputs):
            example[f"{aspect}"] = output["model_response"]
    
    base_modelname = args.model.split("/")[-1]
    with open(os.path.join(output_dir, f"{base_modelname}.json"), "w") as f:
        json.dump(orig_data, f, indent=4)