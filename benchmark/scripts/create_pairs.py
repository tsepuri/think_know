"""Expand config.json + sentences.csv into ranked sentence pairs.

Each row is a pair of sentences, where sentence1 is expected to rank above
sentence2. A "direct" comparison (logP(a) > logP(b)) is a single row, with
`rank` 1. A "differences" comparison (logP(a) - logP(b) > logP(c) - logP(d))
is two rows - (a, b) at rank 1 (the genuine grammatical contrast) and (c, d)
at rank 2 (no contrast expected) - sharing the same integer `assertion_id`.
`rank` generalizes past exactly one rank-2 row if a future condition ever
needs more than a single minuend/subtrahend pair.

config.json lists one entry per phenomenon/subtype ("conditions"). Each
condition is matched against sentences.csv (phenomenon + subtype), then split
into "groups" by group_id.

- "direct" conditions: rows are taken two at a time, in file order, and paired
  as (grammatical, ungrammatical).
- "differences" conditions: within the group's critical rows (condition ==
  "critical"), exactly one sentence is grammatical or exactly one sentence is
  ungrammatical. Difference assertions are built from those 4 sentences per the condition's comparison_types.
  When the condition has control: true and a control_comparison_verb, the same is 
  repeated using the group's control rows (condition == "control") as the 
  target and the critical rows for control_comparison_verb as the reference.

see generate_variants.py's make_unique_id to see how variant sentences are kept from mixing
"""

import argparse
import json
import pandas as pd

OUTPUT_COLUMNS = [
    "phenomenon", "subtype", "group_id", "comparison", "condition_type",
    "comparison_type", "tense", "negation", "matrix_subj_category", "matrix_subj", "matrix_type",
    "verb_target", "verb_reference", "unique_id", "assertion_id", "rank",
    "sentence1", "sentence2"
]

def load_sentences(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def condition_matches(df, condition):
    """Rows of sentences.csv matching one config.json condition's phenomenon/subtype."""
    mask = df["phenomenon"] == condition["phenomenon"]
    subtype = condition.get("subtype")
    mask &= df["subtype"] == (subtype if subtype else "")
    return df[mask]


def iter_groups(rows):
    """Split a condition's matched rows into groups, one per distinct
    `unique_id` - generate_variants.py already builds that to be exactly the
    right grouping key (see make_unique_id there)."""
    for _, group in rows.groupby("unique_id", sort=False):
        yield group


def comparison_grouping(rows, comparison_type, comparison):
    """The row pairs for one comparison_type within a set: group `rows` by the
    comparison_type column, and for each group of 2, pair the grammatical row
    with the ungrammatical one.

    Each pair gets a `rank`: rank 1 is the group expected to show the larger
    log-probability gap (a genuine grammatical/ungrammatical contrast), rank 2
    the group expected to show little to none (both rows share the same
    grammaticality) - the same rank>1 group is repeated for however many
    "no genuine contrast" groups a comparison_type produces, so this
    generalizes past exactly 2 groups if a future condition ever needs it.

    For "direct" comparisons, every group is independently rank 1 - there's
    no second group to subtract against.

    For "differences" comparisons: a group with exactly 1 grammatical row
    (mixed) is rank 1, ordered (grammatical, ungrammatical); a group that's
    uniformly grammatical or ungrammatical is rank 2, in its original row
    order (build_comparison then reorders it to line up with rank 1). This works for both a (1 ungrammatical, 3 grammatical) split and a
    (3 ungrammatical, 1 grammatical) split across the two groups, since it's
    driven by each group's own grammaticality count rather than assuming
    which group is which.
    """
    all_variants = rows[comparison_type].unique()
    sorted_groups = []
    for variant in all_variants:
        variant_group = rows[rows[comparison_type] == variant]
        if len(variant_group) != 2:
            print(rows)
            raise ValueError(f"expected 2 rows for {comparison_type}={variant!r}, got {len(variant_group)}")
        gram = variant_group[variant_group["grammaticality"] == "grammatical"]
        ungram = variant_group[variant_group["grammaticality"] == "ungrammatical"]
        if comparison == "direct":
            sorted_groups.append((1, gram.iloc[0], ungram.iloc[0]))
        else:
            if len(gram) == 1:
                sorted_groups.append((1, gram.iloc[0], ungram.iloc[0]))
            else:
                sorted_groups.append((2, variant_group.iloc[0], variant_group.iloc[1]))
    return sorted_groups

def _difference_signature(r1a, r1b, r2a, r2b):
    """The signed sentence weights a differences assertion reduces to:
    (r1a - r1b) - (r2a - r2b). Two assertions with the same signature test the
    exact same inequality, just grouped differently."""
    weights = {}
    for row, sign in ((r1a, 1), (r1b, -1), (r2a, -1), (r2b, 1)):
        weights[row["sentence"]] = weights.get(row["sentence"], 0) + sign
    return frozenset((k, v) for k, v in weights.items() if v)


def _difference_assertions(group_rows, comparison_type, comparison_types):
    """The (r1a, r1b, r2a, r2b) tuples for one comparison_type's differences
    assertions, rank 2 aligned to rank 1 on the 2x2's other factor."""
    assertions = comparison_grouping(group_rows, comparison_type, "differences")
    # a differences assertion needs one rank-1 (genuine contrast)
    # group paired with one rank-2 (no contrast) group; pair them up
    # in order rather than assuming there's only ever one of each,
    # so this still works if a comparison_type ever produces more
    by_rank = {}
    for rank, row1, row2 in assertions:
        by_rank.setdefault(rank, []).append((row1, row2))
    if by_rank.keys() != {1, 2} or len(by_rank[1]) != len(by_rank[2]):
        raise ValueError(
            f"expected equal rank-1/rank-2 counts for comparison_type="
            f"{comparison_type!r}, got { {k: len(v) for k, v in by_rank.items()} }"
        )
    # the 2x2's other factor: what varies within each group when
    # grouping by comparison_type
    (other,) = [c for c in comparison_types if c != comparison_type]
    out = []
    for (r1a, r1b), (r2a, r2b) in zip(by_rank[1], by_rank[2]):
        # order rank 2 like rank 1, so both pairs make the same swap in the
        # same direction and the subtraction isolates the interaction
        # (otherwise the assertion reduces to a main effect of comparison_type)
        if r2a[other] != r1a[other]:
            r2a, r2b = r2b, r2a
        if (r2a[other], r2b[other]) != (r1a[other], r1b[other]):
            raise ValueError(
                f"rank-2 pair can't be aligned with rank 1 on {other!r}: "
                f"{r1a['sentence']!r} / {r2a['sentence']!r}"
            )
        out.append((r1a, r1b, r2a, r2b))
    return out


def _pairwise_assertions(group_rows, comparison_types):
    """The minimal (grammatical, ungrammatical) pairs among a 2x2's rows: those
    differing on exactly one comparison_types column, tagged with that column.
    With exactly one grammatical or exactly one ungrammatical row there are 3
    gram/ungram pairs, of which 2 are minimal - the third differs on both
    factors, so it isn't a minimal pair and is skipped."""
    gram = group_rows[group_rows["grammaticality"] == "grammatical"]
    ungram = group_rows[group_rows["grammaticality"] == "ungrammatical"]
    out = []
    for _, g in gram.iterrows():
        for _, u in ungram.iterrows():
            differs = [c for c in comparison_types if g[c] != u[c]]
            if len(differs) == 1:
                out.append((differs[0], g, u))
    if len(out) != 2:
        print(group_rows)
        raise ValueError(
            f"expected 2 minimal grammatical > ungrammatical pairs, got {len(out)} "
            f"({len(gram)} grammatical, {len(ungram)} ungrammatical)"
        )
    return out


def build_comparison(set_rows, meta, condition, assertion_id, mode="differences"):
    """Build every pair row for one condition's one set, returning
    (rows, next_assertion_id).

    Builds a "critical" group (set_rows where condition == "critical") and,
    when the condition has control: true, an additional "control" group
    (the group's control rows plus optionally the critical rows for
    control_comparison_verb.

    "direct" conditions get one comparison_grouping assertion per entry in
    comparison_types regardless of mode. "differences" conditions depend on mode:
    - "differences": one difference-of-differences assertion per
      comparison_type, except that one equivalent to an already-built
      assertion (same signed sentences - always the case for a clean 2x2
      interaction) is dropped. matrix_verb goes first, so it's the one kept.
    - "pairwise": each minimal grammatical > ungrammatical pair among the 4
      sentences (differing on one comparison_type) as its own single-pair
      assertion (always 2).
    """
    comparison = condition["comparison"]
    comparison_types = condition.get("comparison_types", [])
    control_on = str(condition.get("control", "")).lower() == "true"
    control_verb = condition.get("control_comparison_verb")
    critical_rows = set_rows[set_rows["condition"] == "critical"]
    rows_out = []

    groups = [("critical", critical_rows)]
    if control_on:
        control_rows = set_rows[set_rows["condition"] == "control"]
        if control_verb:
            reference_rows = critical_rows[critical_rows["matrix_verb"] == control_verb]
            combined = pd.concat([control_rows, reference_rows])
        else:
            combined = control_rows
        groups.append(("control", combined))

    for condition_type, group_rows in groups:
        base_row = {**meta, "condition_type": condition_type}

        if comparison == "differences" and mode == "pairwise":
            for differs, g, u in _pairwise_assertions(group_rows, comparison_types):
                rows_out.append({
                    **base_row, "comparison": "pairwise", "comparison_type": differs,
                    "assertion_id": assertion_id, "rank": 1,
                    "sentence1": g["sentence"], "sentence2": u["sentence"],
                })
                assertion_id += 1
            continue

        if comparison == "direct":
            for comparison_type in comparison_types:
                # every rank-1 tuple is a complete, independent assertion on its
                # own - one comparison_type can produce more than one (e.g.
                # Factive island's matrix_type has 2 values), so each needs its
                # own assertion_id rather than sharing one across all of them
                for rank, row1, row2 in comparison_grouping(group_rows, comparison_type, comparison):
                    rows_out.append({
                        **base_row, "comparison_type": comparison_type,
                        "assertion_id": assertion_id, "rank": rank,
                        "sentence1": row1["sentence"], "sentence2": row2["sentence"],
                    })
                    assertion_id += 1
            continue

        seen = set()
        ordered_types = sorted(comparison_types, key=lambda c: c != "matrix_verb")
        for comparison_type in ordered_types:
            for r1a, r1b, r2a, r2b in _difference_assertions(group_rows, comparison_type, comparison_types):
                signature = _difference_signature(r1a, r1b, r2a, r2b)
                if signature in seen:
                    continue
                seen.add(signature)
                row = {**base_row, "comparison_type": comparison_type, "assertion_id": assertion_id}
                rows_out.append({**row, "rank": 1, "sentence1": r1a["sentence"], "sentence2": r1b["sentence"]})
                rows_out.append({**row, "rank": 2, "sentence1": r2a["sentence"], "sentence2": r2b["sentence"]})
                assertion_id += 1
    return rows_out, assertion_id

def _invalid_phenomena_or_subtype(condition, phenomena, subtypes):
    return (phenomena and condition["phenomenon"] not in phenomena) or subtypes and condition.get("subtype") not in subtypes

def build_pairs(config, sentences_df, phenomena=None, subtypes=None, mode="differences"):
    """Build the full list of pair rows for every matching condition/set
    (see build_comparison for `mode`)."""
    rows_out = []
    assertion_id = 0

    for condition in config["conditions"]:
        if _invalid_phenomena_or_subtype(condition, phenomena, subtypes):
            continue
        matched_sentences = condition_matches(sentences_df, condition)
        if matched_sentences.empty:
            raise ValueError(f"no sentences matched condition {condition.get('group_id')}")

        comparison = condition["comparison"]
        for set_rows in iter_groups(matched_sentences):
            meta = {
                "phenomenon": condition["phenomenon"],
                "subtype": condition.get("subtype") or "",
                "group_id": condition.get("group_id"),
                "comparison": comparison,
                "tense": set_rows["tense"].iloc[0],
                "negation": set_rows["negation"].iloc[0],
                "matrix_subj_category": set_rows["matrix_subj_category"].iloc[0],
                "matrix_subj": set_rows["matrix_subj"].iloc[0],
                "matrix_type": set_rows["matrix_type"].iloc[0],
                "unique_id": set_rows["unique_id"].iloc[0],
            }
            new_rows, assertion_id = build_comparison(set_rows, meta, condition, assertion_id, mode)
            rows_out.extend(new_rows)

    return rows_out


def write_pairs(rows, output_path):
    """Write pair rows to a CSV, using OUTPUT_COLUMNS as the column order."""
    if not rows:
        raise ValueError("No pairs generated - check --phenomena/--subtypes filters")
    pd.DataFrame(rows).reindex(columns=OUTPUT_COLUMNS).to_csv(output_path, index=False)


def main():
    """CLI entry point: parse args, load config.json + sentences.csv, write pairs.csv."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="benchmark/data/config.json")
    parser.add_argument("--sentences", default="benchmark/data/sentences_with_variants.csv")
    parser.add_argument("--output", default="benchmark/data/pairs.csv",
                        help="differences-of-differences version")
    parser.add_argument("--pairwise-output", default="benchmark/data/pairs_pairwise.csv",
                        help="version with each minimal grammatical > ungrammatical pair per 2x2")
    parser.add_argument("--phenomena", nargs="*", default=None, help="restrict to these phenomenon names")
    parser.add_argument("--subtypes", nargs="*", default=None, help="restrict to these subtypes")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)
    sentences_df = load_sentences(args.sentences)

    for mode, output in (("differences", args.output), ("pairwise", args.pairwise_output)):
        rows = build_pairs(
            config,
            sentences_df,
            phenomena=args.phenomena,
            subtypes=args.subtypes,
            mode=mode,
        )
        write_pairs(rows, output)
        n_assertions = len({row["assertion_id"] for row in rows})
        print(f"[{mode}] Wrote {len(rows)} pairs ({n_assertions} assertions) to {output}")

if __name__ == "__main__":
    main()
