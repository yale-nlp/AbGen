#!/usr/bin/env python3
"""
Re‑formats every example in an input JSON file so that each of the four
review features (importance, faithfulness, soundness, overall) becomes

    {
        "explanation": <original text>,
        "score": <integer 0‑5>
    }

**How the score is found**
-------------------------
1. If the text contains the phrase "Therefore, the score is:" (case‑
   insensitive) *anywhere*, the first **single** digit 1‑5 that follows
   that phrase is taken as the score.
2. Otherwise, the script falls back to the previous heuristic:
   * Split the text into *segments* on: periods (.), exclamation points
     (!), question marks (?), **or** one‑or‑more newline characters.
   * Walk backwards to locate the last *meaningful* segment (one that
     contains at least one alphanumeric character).
   * If **exactly one** digit 1‑5 appears in that segment, use it as the
     score; otherwise assign 0.

This covers tricky cases such as:
* The score line being followed by an explanatory sentence.
* The score appearing in bold (e.g. "**3**").
* Trailing markdown fences (```), quotes, or whitespace.

Usage
-----
    python reformat_reviews.py --input reviews.json --output reformatted.json
If --output is omitted, the transformed JSON is printed to stdout.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURE_KEYS: tuple[str, ...] = (
    "importance",
    "faithfulness",
    "soundness",
    "overall",
)

# Match exactly one digit 1–5 surrounded by word boundaries
DIGIT_RE = re.compile(r"\b([1-5])\b")

# Phrase‑based regex: capture first digit 1‑5 after "Therefore, the score is:"
PHRASE_RE = re.compile(
    r"therefore,?\s*the\s+score\s+is[^0-9]*([1-5])",
    re.IGNORECASE | re.DOTALL,
)

# Sentence splitting: ., !, ?, or one‑or‑more \n / \r\n
SENT_SPLIT_RE = re.compile(r"[.!?]|\n+")

# Detect at least one alphanumeric char
ALNUM_RE = re.compile(r"[A-Za-z0-9]")

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def extract_score(text: str) -> int:
    """Extract the review score (1‑5) from *text* according to the rules."""
    # 1. Look for the explicit phrase "Therefore, the score is" first.
    phrase_digits = PHRASE_RE.findall(text)
    unique_digits = {d for d in phrase_digits}
    if len(unique_digits) == 1:
        return int(next(iter(unique_digits)))

    # 2. Fallback: last meaningful sentence heuristic
    segments: List[str] = [seg.strip() for seg in SENT_SPLIT_RE.split(text)]

    last_meaningful: str | None = None
    for seg in reversed(segments):
        if seg and ALNUM_RE.search(seg):
            last_meaningful = seg
            break

    if not last_meaningful:
        return 0

    digits = DIGIT_RE.findall(last_meaningful)
    return int(digits[0]) if len(digits) == 1 else 0


def transform_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap specified feature strings in the required dict format."""
    for key in FEATURE_KEYS:
        value = example.get(key)
        if isinstance(value, str):
            example[key] = {
                "explanation": value,
                "score": extract_score(value),
            }
    return example


def iterate_examples(data: Union[List[Any], Dict[str, Any]]) -> Iterable[Any]:
    """Yield each sub‑object that should be treated as a single example."""
    if isinstance(data, list):
        yield from data
    elif isinstance(data, dict):
        # Distinguish {id: example, ...} from a single example dict
        if all(isinstance(v, dict) for v in data.values()):
            yield from data.values()
        else:
            yield data
    else:
        raise TypeError("Unsupported JSON top‑level structure")

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Re‑format review features in a JSON file.")
    parser.add_argument("--input", "-i", required=True, help="Path to the input JSON file.")
    parser.add_argument("--output", "-o", help="Write transformed JSON here; default: stdout")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        sys.exit(f"Input file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    for example in iterate_examples(data):
        transform_example(example)

    if args.output:
        output_path = Path(args.output)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
    else:
        json.dump(data, sys.stdout, ensure_ascii=False, indent=2)
        print()


if __name__ == "__main__":  # pragma: no cover
    main()
