import datasets
import argparse
import logging
import json, os
import dotenv
from huggingface_hub import login
from datasets import Dataset, DatasetDict
import sys
from typing import List, Dict
from utils.inference_prompt import prepare_user_prompt, build_single_message
from utils.api import process_data, client_dict
from tqdm.asyncio import tqdm_asyncio
import asyncio

# Load environment variables from .env file
dotenv.load_dotenv()

async def _generate_response(data: List[Dict], model_name: str, build_messages, client_dict) -> List[Dict]:
    """LLM‑powered classification wrapper."""

    return await process_data(
        data,
        client_dict=client_dict,
        build_messages=build_messages,
        response_model=None,
        model=model_name,
        enable_structured_output=False,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vLLM inference on a dataset.")
    parser.add_argument("--input_file", type=str, default = "test.json")
    parser.add_argument("--output_dir", type=str, default="model_outputs/test")
    parser.add_argument("--model", type=str, required=True, help="api model name")
    parser.add_argument("--max_tokens", type=int, default=1024)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    filename = args.model.split("/")[-1] + ".json"
    output_filepath = os.path.join(args.output_dir, filename)
    print(f"Saving outputs to {output_filepath}")

    # Load dataset
    data = json.load(open(args.input_file, "r"))


    outputs = asyncio.run(_generate_response(
        data,
        model_name = args.model,
        client_dict = client_dict,
        build_messages = build_single_message,
    ))
        
    with open(output_filepath, "w") as f:
        json.dump(outputs, f, indent=4)