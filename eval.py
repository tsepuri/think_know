"""score pairs.csv (from create_pairs.py) with one or more causal LMs (gpt2 as a sanity
check that the pipeline works, plus Llama-3.1-8B base and instruct for a real read on
the phenomenon).

pairs.csv has one row per pair (sentence1, sentence2), grouped into assertions
via `assertion_id`, each row tagged with a `rank` (1 = the genuine
grammatical contrast, 2 = no contrast expected):
  - comparison "direct":      1 pair, rank 1. assertion holds if logP(sentence1) > logP(sentence2).
  - comparison "differences": 2 pairs, rank 1 and rank 2. assertion holds if
                                   (logP(rank1.sentence1) - logP(rank1.sentence2))
                                   > (logP(rank2.sentence1) - logP(rank2.sentence2))

by default each assertion is currently scored as 1 (holds) or 0 (fails); results are then
averaged across everything except phenomenon (and subtype/matrix_type in the more
detailed breakdown). "control" assertions (condition_type == "control") are
placed in separate buckets for now but could be merged.

when --model is given more than once, every model is scored against the same pairs and
each gets its own details/summary output (model name slotted into the filename), plus a
combined comparison summary across all models.
"""

# TODO: think about switching to minicons
import argparse
import json
import re

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# TODO: add SLOR

# fields to include in results file and for grouping
METADATA_FIELDS = [
    "phenomenon", "subtype", "comparison_type", "condition_type",
    "tense", "negation", "matrix_subj_category", "matrix_subj", "matrix_type",
    "unique_id",
]

def load_pairs(path):
    """Read pairs.csv (from create_pairs.py) into a DataFrame."""
    return pd.read_csv(path, keep_default_na=False, na_values=[])

def sentence_logprob(text, model, tokenizer, device):
    """Mean log-probability per token, i.e. length-normalized.

    Paired sentences are not guaranteed to tokenize to the same length (e.g.
    "whether" vs "that", "the rumor is about Mary" vs "the rumor about Mary"
    token counts can differ by 1-2 between a pair's two sentences). Comparing raw summed
    log-probability would then favor the shorter sentence regardless of grammaticality,
    so we normalize by token count.
    """
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    if input_ids.shape[1] < 2:
        return 0.0
    with torch.no_grad():
        # GPT2LMHeadModel with labels=input_ids shifts internally and returns the mean
        # cross-entropy over the (seq_len - 1) predicted tokens - i.e. -loss is already
        # the length-normalized log-probability we want.
        outputs = model(input_ids, labels=input_ids)
    return -outputs.loss.item()


def build_logprob_cache(pairs_df, model, tokenizer, device):
    """Score every distinct sentence text once and cache the result.

    The same sentence text can appear in many rows (shared across condition_types, or
    reused as e.g. both a minuend and a subtrahend sentence), so scoring by unique text
    avoids redundant forward passes.
    """
    texts = pd.unique(pd.concat([pairs_df["sentence1"], pairs_df["sentence2"]]))
    return {text: sentence_logprob(text, model, tokenizer, device) for text in texts}


def score_pairs(pairs_df, cache):
    """Attach each pair's sentence-level logprobs and their margin (sentence1 - sentence2)
    using the cached scores."""
    pairs_df = pairs_df.copy()
    pairs_df["sentence1_logprob"] = pairs_df["sentence1"].map(cache)
    pairs_df["sentence2_logprob"] = pairs_df["sentence2"].map(cache)
    pairs_df["margin"] = pairs_df["sentence1_logprob"] - pairs_df["sentence2_logprob"]
    return pairs_df


def score_assertions(pairs_df):
    """Group pairs by assertion_id and evaluate the direct/difference assertion.

    `rank` orders rows by expected contrast magnitude: rank 1's logprob gap
    (sentence1 - sentence2) should be the largest, rank 2's smaller, rank 3's
    smaller still, and so on - the assertion holds only if that whole
    sequence is strictly decreasing, not just rank1 vs rank2. With today's
    data every assertion has exactly 1 or 2 ranks, so this reduces to the
    direct (single margin > 0) or differences (rank1's margin > rank2's)
    cases, but it generalizes cleanly if a future condition ever adds rank 3.
    """
    def combine(group):
        margins = group.sort_values("rank")["margin"]
        row = group.iloc[0][METADATA_FIELDS].copy()
        if len(margins) == 1:
            # direct: a single rank-1 row, no lower rank to compare against
            row["margin"] = margins.iloc[0]
            row["correct"] = int(margins.iloc[0] > 0)
        else:
            row["margin"] = margins.iloc[0] - margins.iloc[-1]
            row["correct"] = int(all(a > b for a, b in zip(margins, margins.iloc[1:])))
        return row

    results = pairs_df.groupby("assertion_id").apply(combine, include_groups=False)
    return results.reset_index()


def summarize_with_controls(results, group_by):
    """Aggregate accuracy per group_by, keeping control-condition assertions (sanity
    checks the model is expected to pass trivially) separate from the substantive
    "critical" assertions that actually probe the phenomenon - averaging the two
    together would dilute the signal from the assertions that matter.
    """
    keys = group_by if group_by else ["_all"]
    df = results.copy()
    if not group_by:
        df["_all"] = ""
    df["bucket"] = np.where(df["condition_type"] == "control", "control", "critical")

    stats = df.groupby(keys + ["bucket"])["correct"].agg(n="count", accuracy="mean")
    stats = stats.unstack("bucket")

    summary = pd.DataFrame(index=stats.index)
    summary["n"] = stats.get(("n", "critical"), 0).fillna(0).astype(int)
    summary["accuracy"] = stats.get(("accuracy", "critical"), np.nan)
    summary["control_n"] = stats.get(("n", "control"), 0).fillna(0).astype(int)
    summary["control_accuracy"] = stats.get(("accuracy", "control"), np.nan)
    summary = summary.reset_index()
    if not group_by:
        summary = summary.drop(columns="_all")
    return summary


def round_floats(value, ndigits=3):
    """Recursively round every float in a JSON-able structure (dict/list/scalar) to
    ndigits, so summary files read cleanly without float noise past the precision the
    underlying accuracy counts actually support."""
    if isinstance(value, float):
        return round(value, ndigits)
    if isinstance(value, dict):
        return {key: round_floats(v, ndigits) for key, v in value.items()}
    if isinstance(value, list):
        return [round_floats(v, ndigits) for v in value]
    return value


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def slugify_model_name(model_name):
    """Turn a HF model id (e.g. 'meta-llama/Llama-3.1-8B') into a filename-safe slug."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)


def insert_suffix(path, suffix):
    """Insert '-{suffix}' before a path's extension, e.g. ('eval_results.csv', 'gpt2')
    -> 'eval_results-gpt2.csv'."""
    if "." in path.rsplit("/", 1)[-1]:
        base, ext = path.rsplit(".", 1)
        return f"{base}-{suffix}.{ext}"
    return f"{path}-{suffix}"


def load_model(model_name, device, cache_dir):
    """Load any causal LM (gpt2, Llama, etc.) and its tokenizer via the Auto* classes."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # bigger models won't fit in fp32 on typical hardware; use bf16/fp16 off of CPU.
    # (mps has had spotty bf16 support, so use fp16 there; cuda gets bf16.)
    if device == "cpu":
        dtype = torch.float32
    elif device == "mps":
        dtype = torch.float16
    else:
        dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_name, cache_dir=cache_dir, torch_dtype=dtype
    ).to(device)
    model.eval()
    return model, tokenizer


def run_eval_for_model(model_name, pairs_df, args, suffix_outputs):
    """Score pairs_df with one model and write its details/summary outputs. Returns the
    overall summary row (used to build the cross-model comparison)."""
    print(f"\n=== {model_name} ===")
    print(f"Loading {model_name} on {args.device} (cache: {args.cache_dir or 'huggingface_hub default'})...")
    model, tokenizer = load_model(model_name, args.device, args.cache_dir)

    cache = build_logprob_cache(pairs_df, model, tokenizer, args.device)
    scored_pairs = score_pairs(pairs_df, cache)
    results = score_assertions(scored_pairs)

    del model
    if args.device == "cuda":
        torch.cuda.empty_cache()

    if suffix_outputs:
        slug = slugify_model_name(model_name)
        details_output = insert_suffix(args.details_output, slug)
        summary_output = insert_suffix(args.summary_output, slug)
        pairs_output = insert_suffix(args.pairs_output, slug)
    else:
        details_output = args.details_output
        summary_output = args.summary_output
        pairs_output = args.pairs_output

    scored_pairs.to_csv(pairs_output, index=False)
    print(f"Wrote {len(scored_pairs)} scored pairs to {pairs_output}")

    results.to_csv(details_output, index=False)
    print(f"Wrote {len(results)} scored assertions to {details_output}")

    # by_phenomenon and by_phenomenon_subtype_matrix_type are each computed directly
    # from the raw per-assertion `results`, not from one another, so neither is a
    # (potentially skewed) average-of-averages of the other.
    overall = summarize_with_controls(results, []).iloc[0]
    by_phenomenon = summarize_with_controls(results, ["phenomenon"])
    by_phenomenon_subtype_matrix_type = summarize_with_controls(results, ["phenomenon", "subtype", "matrix_type"])

    print(f"Overall accuracy: {overall['accuracy']:.3f} ({int(overall['n'])} test assertions)")
    print(f"Overall control accuracy: {overall['control_accuracy']:.3f} ({int(overall['control_n'])} control assertions)")
    summary = {
        "model": model_name,
        "n_assertions": len(results),
        "overall_accuracy": overall["accuracy"],
        "overall_control_accuracy": overall["control_accuracy"],
        "by_phenomenon": by_phenomenon.replace({np.nan: None}).to_dict(orient="records"),
        "by_phenomenon_subtype_matrix_type": by_phenomenon_subtype_matrix_type.replace({np.nan: None}).to_dict(orient="records"),
    }
    with open(summary_output, "w", encoding="utf-8") as f:
        json.dump(round_floats(summary), f, indent=2)
    print(f"Wrote summary to {summary_output}")

    return summary


def main():
    """Load pairs.csv, score every sentence with each LM, evaluate each
    assertion, and write both per-assertion details and an aggregate summary per model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default="benchmark_data/pairs.csv")
    parser.add_argument(
        "--model", dest="models", action="append",
        default=None,
        help="HF model id to score with; repeat to score multiple models "
             "(default: gpt2, meta-llama/Llama-3.1-8B, and meta-llama/Llama-3.1-8B-Instruct).",
    )
    parser.add_argument("--pairs-output", default="eval_output/pairs_scored.csv")
    parser.add_argument("--details-output", default="eval_output/eval_results.csv")
    parser.add_argument("--summary-output", default="eval_output/eval_summary.json")
    parser.add_argument("--device", default=default_device())
    parser.add_argument(
        "--cache-dir", default=None,
        help="Where downloaded model weights are stored.",
    )
    args = parser.parse_args()
    models = args.models or [
        "gpt2", "mistralai/Mistral-7B-Instruct-v0.3", "meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct",
    ]

    pairs_df = load_pairs(args.pairs)
    if pairs_df.empty:
        raise SystemExit(f"No rows found in {args.pairs}")

    suffix_outputs = len(models) > 1
    comparison = [
        run_eval_for_model(model_name, pairs_df, args, suffix_outputs) for model_name in models
    ]

    if len(comparison) > 1:
        comparison_output = insert_suffix(args.summary_output, "comparison")
        with open(comparison_output, "w", encoding="utf-8") as f:
            json.dump(round_floats(comparison), f, indent=2)
        print(f"\n=== Comparison ({comparison_output}) ===")
        for summary in comparison:
            print(
                f"{summary['model']}: accuracy={summary['overall_accuracy']:.3f} "
                f"control_accuracy={summary['overall_control_accuracy']:.3f}"
            )


if __name__ == "__main__":
    main()
