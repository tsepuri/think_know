"""score pairs.csv (from create_pairs.py) with gpt2 as a sanity check that the pipeline works.

pairs.csv has one row per pair (sentence1, sentence2), grouped into assertions
via `assertion_id`:
  - comparison_type "direct":     1 pair.  assertion holds if logP(sentence1) > logP(sentence2).
  - comparison_type "difference": 2 pairs (minuend, subtrahend). assertion holds if
                                   (logP(minuend.sentence1) - logP(minuend.sentence2))
                                   > (logP(subtrahend.sentence1) - logP(subtrahend.sentence2))

by default each assertion is currently scored as 1 (holds) or 0 (fails); results are then
averaged across everything except phenomenon (and subtype/matrix_type in the more
detailed breakdown). "control" assertions (prediction_type == "control") are
placed in separate buckets for now but could be merged.
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# fields to include in results file and for grouping
METADATA_FIELDS = [
    "phenomenon", "category", "subtype", "comparison_type", "condition_key",
    "tense", "negation", "matrix_subj", "matrix_type", "prediction_type",
    "lexical_variant",
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

    The same sentence text can appear in many rows (shared across prediction_types, or
    reused as e.g. both a minuend and a subtrahend sentence), so scoring by unique text
    avoids redundant forward passes.
    """
    texts = pd.unique(pd.concat([pairs_df["sentence1_text"], pairs_df["sentence2_text"]]))
    return {text: sentence_logprob(text, model, tokenizer, device) for text in texts}


def score_pairs(pairs_df, cache):
    """Attach each pair's logprob margin (sentence1 - sentence2) using the cached scores."""
    pairs_df = pairs_df.copy()
    pairs_df["margin"] = (
        pairs_df["sentence1_text"].map(cache) - pairs_df["sentence2_text"].map(cache)
    )
    return pairs_df


def score_assertions(pairs_df):
    """Group pairs by assertion_id and evaluate the direct/difference assertion."""
    def combine(group):
        by_role = group.set_index("pair_role")["margin"]
        if "comparison" in by_role.index:
            margin = by_role["comparison"]
        else:
            margin = by_role["minuend"] - by_role["subtrahend"]
        row = group.iloc[0][METADATA_FIELDS].copy()
        row["margin"] = margin
        row["correct"] = int(margin > 0)
        return row

    results = pairs_df.groupby("assertion_id").apply(combine)
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
    df["bucket"] = np.where(df["prediction_type"] == "control", "control", "critical")

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


def main():
    """Load pairs.csv, score every sentence with the LM, evaluate each
    assertion, and write both per-assertion details and an aggregate summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default="pairs.csv")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--details-output", default="eval_results.csv")
    parser.add_argument("--summary-output", default="eval_summary.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    pairs_df = load_pairs(args.pairs)
    if pairs_df.empty:
        raise SystemExit(f"No rows found in {args.pairs}")

    # TODO: using gpt2tokenizer and gpt2model by default
    print(f"Loading {args.model} on {args.device}...")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained(args.model).to(args.device)
    model.eval()

    cache = build_logprob_cache(pairs_df, model, tokenizer, args.device)
    # TODO: save pair log probs as well?
    pairs_df = score_pairs(pairs_df, cache)
    results = score_assertions(pairs_df)

    results.to_csv(args.details_output, index=False)
    print(f"Wrote {len(results)} scored assertions to {args.details_output}")

    # TODO: maybe first summarize by phenomenon/subtype/matrix type and then summarize by phenomenon?
    overall = summarize_with_controls(results, []).iloc[0]
    by_phenomenon = summarize_with_controls(results, ["phenomenon"])
    by_phenomenon_subtype_matrix_type = summarize_with_controls(results, ["phenomenon", "subtype", "matrix_type"])

    print(f"\nOverall accuracy: {overall['accuracy']:.3f} ({int(overall['n'])} test assertions)")
    print(f"Overall control accuracy: {overall['control_accuracy']:.3f} ({int(overall['control_n'])} control assertions)")
    summary = {
        "model": args.model,
        "n_assertions": len(results),
        "overall_accuracy": overall["accuracy"],
        "overall_control_accuracy": overall["control_accuracy"],
        "by_phenomenon": by_phenomenon.replace({np.nan: None}).to_dict(orient="records"),
        "by_phenomenon_subtype_matrix_type": by_phenomenon_subtype_matrix_type.replace({np.nan: None}).to_dict(orient="records"),
    }
    with open(args.summary_output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote summary to {args.summary_output}")


if __name__ == "__main__":
    main()
