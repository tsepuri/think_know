"""pairs.csv / pairs_pairwise.csv -> baby_pairs.csv / baby_pairs_pairwise.csv: both sentences in
CHILDES formatting, without the assertions that use a vocabulary.csv entry whose
childes_lemma_count (filled in by notebooks/childes_think_know.py) is below --threshold.

An assertion is dropped whole when any of its rows has a rare entry, so a differences
assertion never loses only one of its rank-1/rank-2 pairs. Words that aren't vocabulary.csv
entries (e.g. authored seed content) aren't checked.
"""
import argparse
import os
import re

import pandas as pd

import utils


def rare_entries(pairs, entries):
    """{assertion_id: sorted entries found} for assertions whose sentences contain any of
    `entries` (whole-word, case-insensitive)."""
    patterns = {e: re.compile(r"\b" + re.escape(e) + r"\b", re.IGNORECASE) for e in entries}
    rare = {}
    for assertion_id, group in pairs.groupby("assertion_id"):
        text = " ".join(pd.concat([group.sentence1, group.sentence2]))
        found = sorted(e for e, p in patterns.items() if p.search(text))
        if found:
            rare[assertion_id] = found
    return rare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", nargs="+", default=["benchmark/data/pairs.csv", "benchmark/data/pairs_pairwise.csv"])
    parser.add_argument("--vocabulary", default="benchmark/data/vocabulary.csv")
    parser.add_argument("--threshold", type=int, default=10)
    args = parser.parse_args()

    vocab = pd.read_csv(args.vocabulary, keep_default_na=False, dtype=str)
    vocab = vocab[vocab.childes_lemma_count != ""]
    below = vocab[vocab.childes_lemma_count.astype(int) < args.threshold]
    entries = sorted({utils.normative_to_childes_formatting(w) for w in below.word})
    print(f"{len(entries)} vocabulary entries with childes_lemma_count < {args.threshold}: {', '.join(entries)}")

    for path in args.pairs:
        df = pd.read_csv(path, keep_default_na=False, na_values=[])
        for col in ("sentence1", "sentence2"):
            df[col] = df[col].map(utils.normative_to_childes_formatting)
        rare = rare_entries(df, entries)
        kept = df[~df.assertion_id.isin(rare)]
        output = os.path.join(os.path.dirname(path), "baby_" + os.path.basename(path))
        kept.to_csv(output, index=False)
        print(f"{output}: kept {kept.assertion_id.nunique()} of {df.assertion_id.nunique()} assertions ({len(kept)} pairs)")
        dropped = df[df.assertion_id.isin(rare)].drop_duplicates("assertion_id")
        for (phenomenon, subtype), g in dropped.groupby(["phenomenon", "subtype"]):
            print(f"  dropped {len(g)}: {phenomenon} {subtype}".rstrip())


if __name__ == "__main__":
    main()
