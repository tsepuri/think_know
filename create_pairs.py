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

def build_comparison(set_rows, meta, condition, assertion_id):
    """Build every pair row for one condition's one set, returning
    (rows, next_assertion_id).

    Builds a "critical" group (set_rows where condition == "critical") and,
    when the condition has control: true, an additional "control" group
    (the group's control rows plus optionally the critical rows for
    control_comparison_verb. Each group gets one comparison_grouping
    assertion per entry in the condition's comparison_types.
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
        for comparison_type in comparison_types:
            assertions = comparison_grouping(group_rows, comparison_type, comparison)
            base_row = {**meta, "comparison_type": comparison_type, "condition_type": condition_type}

            if comparison == "direct":
                # every rank-1 tuple is a complete, independent assertion on its
                # own - one comparison_type can produce more than one (e.g.
                # Factive island's matrix_type has 2 values), so each needs its
                # own assertion_id rather than sharing one across all of them
                for rank, row1, row2 in assertions:
                    rows_out.append({
                        **base_row, "assertion_id": assertion_id, "rank": rank,
                        "sentence1": row1["sentence"], "sentence2": row2["sentence"],
                    })
                    assertion_id += 1
            else:
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
                    rows_out.append({
                        **base_row, "assertion_id": assertion_id, "rank": 1,
                        "sentence1": r1a["sentence"], "sentence2": r1b["sentence"],
                    })
                    rows_out.append({
                        **base_row, "assertion_id": assertion_id, "rank": 2,
                        "sentence1": r2a["sentence"], "sentence2": r2b["sentence"],
                    })
                    assertion_id += 1
    return rows_out, assertion_id

def _invalid_phenomena_or_subtype(condition, phenomena, subtypes):
    return (phenomena and condition["phenomenon"] not in phenomena) or subtypes and condition.get("subtype") not in subtypes

def build_pairs(config, sentences_df, phenomena=None, subtypes=None):
    """Build the full list of pair rows for every matching condition/set."""
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
            new_rows, assertion_id = build_comparison(set_rows, meta, condition, assertion_id)
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
    parser.add_argument("--config", default="benchmark_data/config.json")
    parser.add_argument("--sentences", default="benchmark_data/sentences_with_variants.csv")
    parser.add_argument("--output", default="benchmark_data/pairs.csv")
    parser.add_argument("--phenomena", nargs="*", default=None, help="restrict to these phenomenon names")
    parser.add_argument("--subtypes", nargs="*", default=None, help="restrict to these subtypes")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)
    sentences_df = load_sentences(args.sentences)

    rows = build_pairs(
        config,
        sentences_df,
        phenomena=args.phenomena,
        subtypes=args.subtypes,
    )
    write_pairs(rows, args.output)
    n_assertions = len({row["assertion_id"] for row in rows})
    print(f"Wrote {len(rows)} pairs ({n_assertions} assertions) to {args.output}")


if __name__ == "__main__":
    main()
