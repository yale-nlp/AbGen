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

# Load environment variables from .env file
dotenv.load_dotenv()

class EVALUATION_RESPONSE(BaseModel):
    explanation: str
    score: int



async def _generate_response(data: List[Dict], model_name: str, build_messages, client_dict) -> List[Dict]:
    """LLM‑powered classification wrapper."""

    return await process_data(
        data,
        client_dict=client_dict,
        build_messages=build_messages,
        response_model=EVALUATION_RESPONSE,
        model=model_name,
        max_concurrency=32,
        max_tokens=8192,
    )

aspect_dict = {
    "importance": build_importance_single_message,
    "faithfulness": build_faithfulness_single_message,
    "soundness": build_soundness_single_message,
    "overall": build_overall_single_message,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vLLM inference on a dataset.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file containing the dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the outputs")
    parser.add_argument("--model", default="gpt-4.1-mini", help="api model name")
    parser.add_argument("--max_tokens", type=int, default=1024)
    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    filename = os.path.basename(args.input_file)
    output_filepath = os.path.join(args.output_dir, filename)
    print(f"Saving outputs to {output_filepath}")

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
            try:
                example[f"{aspect}"] = {
                    "explanation": output["explanation"],
                    "score": output["score"]
                }
            except Exception as e:
                logging.error(f"Error processing example", e)
                example[f"{aspect}"] = {
                    "explanation": "Error processing response",
                    "score": 0
                }
        
    with open(output_filepath, "w") as f:
        json.dump(orig_data, f, indent=4)