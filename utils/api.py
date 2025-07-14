import asyncio
import random
from typing import Callable, Dict, List, Sequence, Type, Optional, Union

import openai
import backoff
from pydantic import BaseModel
import pydantic_core
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI, AsyncAzureOpenAI
import dotenv
import os

dotenv.load_dotenv()

azure_client = AsyncAzureOpenAI(
    azure_endpoint=os.getenv("OUR_AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("OUR_AZURE_OPENAI_API_KEY"),
    api_version="2025-01-01-preview",
)

# openai_client = AsyncOpenAI(
#     api_key=os.getenv("OPENAI_API_KEY"),
# )

gemini_client = AsyncOpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.getenv("GEMINI_API_KEY"),
)

deepseek_client = AsyncOpenAI(
    base_url="https://api.deepseek.com/v1/",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

together_client = AsyncOpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=os.getenv("TOGETHER_API_KEY"),
)

client_dict: Dict[str, List[openai.AsyncClient]] = {
    "gpt-4o": azure_client,
    "gpt-4.1": azure_client,
    "gpt-4.1-mini": azure_client,
    "o4-mini": azure_client,
    "DeepSeek-R1-0528": azure_client,
    "DeepSeek-V3-0324": azure_client,
    "Llama-4-Scout-17B-16E-Instruct": azure_client,
    "Llama-4-Maverick-17B-128E-Instruct-FP8": azure_client,
    "gemini-2.5-flash-preview-05-20": gemini_client,
    "gemini-2.5-pro-preview-05-06": gemini_client,
    "deepseek-chat": deepseek_client,
    "deepseek-reasoner": deepseek_client,
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": together_client,
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": together_client,
    "Qwen/Qwen3-235B-A22B-fp8-tput": together_client,
}

__all__: Sequence[str] = (
    "process_single_example",
    "process_single_example_unstructured",
    "process_data",
)


# ---------------------------------------------------------------------------
# Low‑level helpers
# ---------------------------------------------------------------------------

async def _call_llm(
    *,
    client_dict: Dict[str, List[openai.AsyncClient]],
    model: str,
    messages: List[Dict[str, str]],
    response_model: Type[BaseModel],
    temperature: float = 1.0,
    max_tokens: int = 512,
) -> BaseModel:
    """Structured helper that selects a random client for *model* and invokes the
    *Chat Completions* endpoint with Pydantic schema parsing enabled.
    """

    client = client_dict[model]
    response = await client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        response_format=response_model,
    )
    return response.choices[0].message.parsed


async def _call_llm_unstructured(
    *,
    client_dict: Dict[str, List[openai.AsyncClient]],
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 1.0,
    max_tokens: int = 512,
) -> str:
    """Unstructured helper that selects a random client for *model* and invokes the
    *Chat Completions* endpoint without any schema parsing. Returns raw text
    content from the first choice.
    """

    client = client_dict[model]
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_tokens,
    )
    if model == "deepseek-reasoner":
        return response.choices[0].message.content, response.choices[0].message.reasoning_content
    if "qwen3" in model.lower():
        response = response.choices[0].message.content
        if "</think>" in response:
            return response.split("</think>")[1].strip(), response.split("</think>")[0].strip()
        else:
            return response.strip(), ""

    return response.choices[0].message.content, ""


# ---------------------------------------------------------------------------
# Back‑off decorator (shared by both structured and unstructured paths)
# ---------------------------------------------------------------------------

def _backoff_decorator():
    """
    Returns a back-off decorator wired for the usual OpenAI transient failures.
    We *exclude* `pydantic_core._pydantic_core.ValidationError` so we can give it
    its own 10-try loop and graceful fallback inside `process_single_example`.
    """
    return backoff.on_exception(
        backoff.expo,
        (
            openai.RateLimitError,
            openai.LengthFinishReasonError,
            openai.APITimeoutError,
        ),
        max_tries=10,
    )


# ---------------------------------------------------------------------------
# Single-example processing helper
# ---------------------------------------------------------------------------

@_backoff_decorator()
async def process_single_example(
    example: Dict,
    *,
    client_dict: Dict[str, List[openai.AsyncClient]],
    build_messages: Callable[[Dict], List[Dict[str, str]]],
    response_model: Type[BaseModel],
    model: str,
    post_process: Optional[Callable[[Dict, BaseModel, str], Dict]] = None,
    temperature: float = 1.0,
    max_tokens: int = 512,
) -> Optional[Dict]:
    """
    Enrich *one* example via a structured LLM call.

    For `pydantic_core._pydantic_core.ValidationError` we try up to 10 times.
    If the 10th attempt still fails, we swallow the error and return `None`
    so that the caller can decide how to handle a “hard” validation failure.
    """

    # Retry loop specific to ValidationError
    for attempt in range(10):
        try:
            parsed = await _call_llm(
                client_dict=client_dict,
                model=model,
                messages=build_messages(example),
                response_model=response_model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            break  # success ― exit loop
        except pydantic_core._pydantic_core.ValidationError as ve:
            if attempt < 9:
                continue
            # 10th failure → give up gracefully
            print(f"ValidationError (10/10) for {example}: {ve}")
            return None
        except (openai.ContentFilterFinishReasonError,
                openai.BadRequestError) as e:
            # Non-retryable in our logic → abort immediately
            print(f"Error processing example {example}: {e}")
            return None

    # Optional post-processing
    if post_process is not None:
        return post_process(example, parsed, model)

    # In-place update by default
    if parsed is None:
        return None
    example.update(parsed.dict())
    return example


@_backoff_decorator()
async def process_single_example_unstructured(
    example: Dict,
    *,
    client_dict: Dict[str, List[openai.AsyncClient]],
    build_messages: Callable[[Dict], List[Dict[str, str]]],
    model: str,
    post_process: Optional[Callable[[Dict, Union[str, BaseModel], str], Dict]] = None,
    temperature: float = 1.0,
    max_tokens: int = 512,
) -> Dict:
    """Generic helper to enrich *one* example via **unstructured** LLM call.

    The LLM output is returned as raw text. If *post_process* is provided it
    receives this text; otherwise the text is stored under the key
    ``"response"`` in the *example* dict.
    """

    try:
        output = await _call_llm_unstructured(
            client_dict=client_dict,
            model=model,
            messages=build_messages(example),
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except (openai.ContentFilterFinishReasonError, openai.BadRequestError, pydantic_core._pydantic_core.ValidationError) as e:
        print(f"Error processing example {example}: {e}")
        return None

    if output[1]:
        example["model_thinking_process"] = output[1]
        example["model_response"] = output[0]
    else:
        example["model_response"] = output[0]
    
    return example


# ---------------------------------------------------------------------------
# High‑level orchestrator
# ---------------------------------------------------------------------------

async def process_data(
    data: List[Dict],
    *,
    client_dict: Dict[str, List[openai.AsyncClient]],
    build_messages: Callable[[Dict], List[Dict[str, str]]],
    response_model: Type[BaseModel],
    model: str,
    enable_structured_output: bool = True,
    post_process: Optional[Callable[[Dict, Union[BaseModel, str], str], Dict]] = None,
    max_concurrency: int = 16,
    temperature: float = 1.0,
    max_tokens: int = 2048,
) -> List[Dict]:
    """High‑level orchestrator that processes *data* in parallel.

    Args:
        data: List of examples (dicts) to be processed.
        client_dict: Mapping ``{model_name: [async_clients...]}``.
        build_messages: Callable that turns each example into an OpenAI messages
            list.
        response_model: Pydantic model used for parsing structured results. This
            is only required when ``enable_structured_output`` is *True*.
        models: Pool of model names to randomly sample from for each request.
        enable_structured_output: If *True* (default) each request routes through
            the structured helpers and returns parsed Pydantic objects.
            Otherwise the raw text helpers are used.
        post_process: Optional hook to combine the parsed/raw response with the
            original example. See :pyfunc:`process_single_example` and
            :pyfunc:`process_single_example_unstructured`.
        max_concurrency: Hard cap on concurrent OpenAI requests.
        temperature / max_tokens: Passed through to the OpenAI call.

    Returns:
        A list containing the processed examples in the same order as *data*.
    """

    semaphore = asyncio.Semaphore(max_concurrency)

    async def _worker(ex: Dict) -> Dict:
        async with semaphore:
            model
            if enable_structured_output:
                return await process_single_example(
                    ex,
                    client_dict=client_dict,
                    build_messages=build_messages,
                    response_model=response_model,
                    model=model,
                    post_process=post_process,  # type: ignore[arg-type]
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                return await process_single_example_unstructured(
                    ex,
                    client_dict=client_dict,
                    build_messages=build_messages,
                    model=model,
                    post_process=post_process,  # type: ignore[arg-type]
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

    tasks = [asyncio.create_task(_worker(ex)) for ex in data]
    return await tqdm_asyncio.gather(*tasks, desc="Processing Items", total=len(tasks))