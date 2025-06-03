"""
eval_only.py
Re-compute LiveCodeBench metrics from an already-generated result file.
Usage:
  python eval_only.py \
      --results_file path/to/Scenario.codegeneration_10_0.2_eval_all.json \
      --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
      --scenario codegeneration
"""

from __future__ import annotations
import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
from types import SimpleNamespace

from lcb_runner.utils.scenarios import Scenario
from lcb_runner.lm_styles import LanguageModelStore
from lcb_runner.runner.scenario_router import (
    build_prompt_benchmark,
    get_metrics,
)

# ---------------------------------------------------------------------------


def build_stub_args(cli: Namespace, scenario: Scenario) -> SimpleNamespace:
    """
    Build a *minimal* args namespace that satisfies every attribute access
    inside build_prompt_benchmark()  and  get_metrics().
    """
    stub = SimpleNamespace(
        # --- general --------------------------------------------------------
        scenario=scenario,
        debug=False,
        not_fast=False,
        few_shot_count=0,
        shuffle_few_shot=False,
        input_path=None,
        release_version="release_v6",
        start_date="2024-08-01",
        end_date="2025-02-01",
        cot_code_execution=False,
        # --- generation / evaluation ---------------------------------------
        n=1,
        max_tokens=None,
        timeout=cli.timeout,                # <- used by the grader
        num_process_evaluate=100,           # sensible default
    )
    return stub


# ---------------------------------------------------------------------------


def reorder_lists(entry):
    """
    Move pairs with an empty code snippet to the tail,
    preserving the 1-to-1 correspondence between output_list
    (raw model text) and code_list (extracted code).
    """
    pairs = list(zip(entry["output_list"], entry["code_list"]))
    pairs.sort(key=lambda p: not p[1].strip())
    entry["output_list"], entry["code_list"] = map(list, zip(*pairs))

def main() -> None:
    ap = ArgumentParser()
    ap.add_argument("--results_file", required=True,
                    help="Path to *_eval_all.json  or plain generation .json")
    ap.add_argument("--scenario", default="codegeneration",
                    choices=[s.value for s in Scenario])
    ap.add_argument("--model", required=True,
                    help="Model name (only for metadata header)")
    ap.add_argument("--timeout", type=int, default=6,
                    help="Per-test-case timeout used by the grader")
    cli = ap.parse_args()

    scenario = Scenario(cli.scenario)
    _ = LanguageModelStore[cli.model]      # only to confirm the model key exists

    stub_args = build_stub_args(cli, scenario)
    benchmark, _ = build_prompt_benchmark(stub_args)
    print(f"Loaded {len(benchmark)} problems")

    results_file = Path(cli.results_file)
    with results_file.open() as fp:
        saved = json.load(fp)
        
    for entry in saved:
        reorder_lists(entry)
    # each entry already contains `output_list` and `code_list`
    combined_results = [
        (entry["output_list"], entry["code_list"]) for entry in saved
    ]
    print(saved[0].keys())

    metrics, raw_results, metadata = get_metrics(
        scenario,
        stub_args,            # provides .timeout & .num_process_evaluate
        benchmark,
        combined_results,
    )
    print(json.dumps(float(metrics["pass@1"]), indent=2))
    print(json.dumps(float(metrics["pass@5"]), indent=2))
    print(json.dumps(float(metrics["pass@10"]), indent=2))
    
    out_path = results_file.with_name(results_file.stem + "_metrics.json")
    with out_path.open("w") as fp:
        json.dump(metrics, fp, indent=2)
    print(f"Saved detailed metrics to {out_path}")

if __name__ == "__main__":
    main()
