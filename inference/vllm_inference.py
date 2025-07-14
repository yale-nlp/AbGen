import datasets
import argparse
import logging
import json, os
import vllm
import dotenv
from huggingface_hub import login
from datasets import Dataset, DatasetDict
from vllm.config import CompilationConfig, CompilationLevel
import sys
from utils.inference_prompt import prepare_user_prompt, build_single_message

# Load environment variables from .env file
dotenv.load_dotenv()

login(token=os.getenv("HF_TOKEN"))

def process_reasoning_model(response):
    if "</think>" in response:
        return response.split("</think>")[0].strip(), response.split("</think>")[1].strip()
    else:
        return "", response.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vLLM inference on a dataset.")
    parser.add_argument("--input_file", type=str, default = "test.json")
    parser.add_argument("--output_dir", type=str, default="model_outputs/test")
    parser.add_argument("--model", type=str, required=True, help="vLLM model name")
    parser.add_argument("--tensor_parallel_size", type=int, default=2, help="Tensor parallel size for vLLM")
    parser.add_argument("--max_tokens", type=int, default=1024)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    filename = args.model.split("/")[-1] + ".json"
    output_filepath = os.path.join(args.output_dir, filename)
    print(f"Saving outputs to {output_filepath}")

    llm = vllm.LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.98,
        max_num_seqs=128,
        enforce_eager=True
    )

    sampling_params = vllm.SamplingParams(
        temperature=1.0,
        max_tokens=args.max_tokens,
    )

    # Load dataset
    data = json.load(open(args.input_file, "r"))
    messages = load_data(data)
    outputs = llm.chat(messages, sampling_params, use_tqdm=True)

    for example, output in zip(data, outputs):
        if "<think>" in output.outputs[0].text:
            thinking_process, response = process_reasoning_model(output.outputs[0].text)
            example["model_thinking_process"] = thinking_process
            example["model_response"] = response
        else:
            example["model_response"] = output.outputs[0].text.strip()
        
    with open(output_filepath, "w") as f:
        json.dump(data, f, indent=4)