#!/usr/bin/env python3
"""Meta-evaluation script

This script aggregates human annotation scores and automatic evaluation
outputs, computes **Kendall τ** correlations at the *system level* and
**Pearson r** correlations at the *instance level* for each evaluator.
It prints the results as nicely formatted tables and also saves them as
CSV files in the folder ``meta_evaluation_results``.

Folder structure created/expected
---------------------------------
* ``human_evaluation.json`` – human annotations (input)
* ``evaluation_processed_outputs/*.json`` – automatic evaluator outputs (input)
* ``meta_evaluation_results/`` – will be created if it does not exist and will
  contain two files:
  * ``system_level_kendall_tau.csv``
  * ``instance_level_pearson_r.csv``
"""

import csv
import glob
import json
import os
from collections import defaultdict
from typing import Any, Dict, List

from scipy.stats import kendalltau, pearsonr, rankdata  # rankdata still needed for Kendall
from tabulate import tabulate

# ---------------------------------------------------------------------------
# Paths – adjust if your data live elsewhere
# ---------------------------------------------------------------------------

HUMAN_EVAL_PATH = "human_evaluation.json"
EVAL_DIR = "meta_evaluation_outputs/processed_outputs"
RESULTS_DIR = "meta_evaluation_outputs/meta_evaluation_results"
SYSTEM_CSV = os.path.join(RESULTS_DIR, "system_level_kendall_tau.csv")
INSTANCE_CSV = os.path.join(RESULTS_DIR, "instance_level_pearson_r.csv")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ASPECTS = ["importance", "faithfulness", "soundness", "overall"]

def get_system_name(item: Dict[str, Any]) -> str:
    """Extract the system/model name from a meta_id field."""
    return item["meta_id"].split("@")[-1]


def mean(lst: List[float]) -> float:
    """Return the mean of *lst* or NaN if the list is empty."""
    return sum(lst) / len(lst) if lst else float("nan")


def kendall_corr(x: List[float], y: List[float]) -> float:
    """Compute Kendall τ on *rankings* derived from the score lists."""
    return kendalltau(rankdata(x), rankdata(y))[0]

# ---------------------------------------------------------------------------
# Load human annotations
# ---------------------------------------------------------------------------

with open(HUMAN_EVAL_PATH, "r", encoding="utf-8") as f:
    human_data = json.load(f)

SYSTEMS = sorted({get_system_name(item) for item in human_data})

# Per-system aggregated means
a_human_sys: Dict[str, Dict[str, float]] = {s: {a: [] for a in ASPECTS} for s in SYSTEMS}
# Per-instance lookup
human_inst: Dict[str, Dict[str, float]] = {}

for item in human_data:
    sys = get_system_name(item)

    # Compute instance-level scores (overwriting/creating overall)
    inst_scores = {
        "importance": item["importance"]["score"],
        "faithfulness": item["faithfulness"]["score"],
        "soundness": item["soundness"]["score"],
    }
    inst_scores["overall"] = mean(list(inst_scores.values()))

    # Store per-system lists and per-instance dict
    for aspect, score in inst_scores.items():
        a_human_sys[sys][aspect].append(score)
    human_inst[item["meta_id"]] = inst_scores

# Collapse per-system lists into means
for sys in SYSTEMS:
    for aspect in ASPECTS:
        a_human_sys[sys][aspect] = mean(a_human_sys[sys][aspect])

# ---------------------------------------------------------------------------
# Load automatic evaluator scores
# ---------------------------------------------------------------------------

a_eval_sys: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
    lambda: {s: {a: [] for a in ASPECTS} for s in SYSTEMS}
)
a_eval_inst: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)

for file_path in glob.glob(os.path.join(EVAL_DIR, "*.json")):
    evaluator = os.path.splitext(os.path.basename(file_path))[0]
    with open(file_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    for item in eval_data:
        sys = get_system_name(item)
        meta_id = item["meta_id"]

        # Collect per-system and per-instance scores
        for aspect in ASPECTS:
            score = item[aspect]["score"] if aspect != "overall" else item[aspect]["score"]
            a_eval_sys[evaluator][sys][aspect].append(score)
            a_eval_inst[evaluator].setdefault(meta_id, {})[aspect] = score

# Collapse automatic scores to means per system
for evaluator, sys_dict in a_eval_sys.items():
    for sys in SYSTEMS:
        for aspect in ASPECTS:
            sys_dict[sys][aspect] = mean(sys_dict[sys][aspect])

# ---------------------------------------------------------------------------
# Correlation calculations
# ---------------------------------------------------------------------------

# System-level correlations (Kendall τ)
system_corr: Dict[str, Dict[str, float]] = defaultdict(dict)
for evaluator, sys_dict in a_eval_sys.items():
    for aspect in ASPECTS:
        human_vals = [a_human_sys[sys][aspect] for sys in SYSTEMS]
        eval_vals = [sys_dict[sys][aspect] for sys in SYSTEMS]
        system_corr[evaluator][aspect] = kendall_corr(human_vals, eval_vals)

# Instance-level correlations (Pearson r)
instance_corr: Dict[str, Dict[str, float]] = defaultdict(dict)
for evaluator, inst_dict in a_eval_inst.items():
    for aspect in ASPECTS:
        human_vals: List[float] = []
        eval_vals: List[float] = []
        for meta_id, eval_scores in inst_dict.items():
            if meta_id not in human_inst:
                continue  # skip items without human annotation
            human_vals.append(human_inst[meta_id][aspect])
            eval_vals.append(eval_scores[aspect])
        if human_vals:  # guard against empty lists
            instance_corr[evaluator][aspect] = pearsonr(human_vals, eval_vals)[0]
        else:
            instance_corr[evaluator][aspect] = float("nan")

# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def format_rows(corr_dict: Dict[str, Dict[str, float]]) -> List[List[str]]:
    """Return header + rows formatted to 3 decimal places for *corr_dict*."""
    header = ["Evaluator"] + ASPECTS
    rows = [
        [ev] + [f"{corr_dict[ev][a]:.3f}" for a in ASPECTS]
        for ev in sorted(corr_dict.keys())
    ]
    return [header] + rows


def print_table(title: str, corr_dict: Dict[str, Dict[str, float]]):
    """Pretty-print *corr_dict* as a table with *title*."""
    print(f"\n{title}\n")
    print(tabulate(format_rows(corr_dict), headers="firstrow"))


def save_csv(corr_dict: Dict[str, Dict[str, float]], filepath: str):
    """Save *corr_dict* as a CSV at *filepath* (creates parent dirs)."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in format_rows(corr_dict):
            writer.writerow(row)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Print nicely formatted tables to stdout
    print_table("System-level Kendall τ", system_corr)
    print_table("Instance-level Pearson r", instance_corr)

    # Save the same tables to CSV files
    save_csv(system_corr, SYSTEM_CSV)
    save_csv(instance_corr, INSTANCE_CSV)

    print(f"\nCSV files saved to: {RESULTS_DIR}\n")
