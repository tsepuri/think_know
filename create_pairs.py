"""Expand config.json + sentences.csv into ranked sentence pairs.

Each row is a pair of sentences, where sentence1 is expected to rank above
sentence2. A "direct" comparison (logP(a) > logP(b)) is a single row, with
pair_role "comparison". A "differences" comparison (logP(a) - logP(b) >
logP(c) - logP(d)) is two rows - (a, b) as "minuend" and (c, d) as
"subtrahend" - sharing the same integer `assertion_id`.

config.json lists one entry per phenomenon/subtype ("conditions"). Each
condition is matched against sentences.csv (phenomenon + subtype), then split
into "groups" by group_id.

- "direct" conditions: rows are taken two at a time, in file order, and paired
  as (grammatical, ungrammatical).
- "differences" conditions: within the group's critical rows (condition ==
  "critical"), exactly one sentence is grammatical. Difference assertions 
  are built from those 4 sentences per the condition's comparison_types. 
  When the condition has control: true and a control_comparison_verb, 
  the same is repeated using the group's control rows (matrix_verb == "control") 
  as the target and the critical rows for control_comparison_verb as the reference.

By default, every assertion is duplicated with a few lexical substitutions
applied to all of its sentences (skipped when the substitution has no effect):
  - rumor -> message
  - rain -> snow (and inflections: rains/raining/rained -> snows/snowing/snowed)
  - Mary -> Christopher, John -> Theresa   (proper-name swap)
  - Mary -> the student, John -> the teacher   (role-noun swap)
"""

import argparse
import json
import re
import pandas as pd

BASE_VARIANT = "base"
LEXICAL_VARIANTS = {
    "rumor_message": {"rumor": "message"},
    "rain_snow": {"rain": "snow"},
    "name_swap_proper": {"Mary": "Christopher", "John": "Theresa"},
    "name_swap_role": {"Mary": "the student", "John": "the teacher"},
}
ALL_VARIANTS = [BASE_VARIANT] + list(LEXICAL_VARIANTS)
INFLECTIONAL_SUFFIXES = ("s", "ing", "ed")  # rain -> rain/rains/raining/rained

OUTPUT_COLUMNS = [
    "phenomenon", "subtype", "group_id", "comparison", "condition_type",
    "comparison_type", "tense", "negation", "matrix_subj", "matrix_type",
    "verb_target", "verb_reference", "lexical_variant", "assertion_id", "pair_role",
    "sentence1", "sentence2"
]

# Descriptive columns that identify one instantiation of a condition (e.g. one
# of CP + Q's 8 present/past x person x declarative/polar_question combos).
SET_KEY_COLUMNS = ["tense", "negation", "matrix_subj"]


def load_sentences(path):
    """Load sentences.csv, tagging every row with its file order (row_num)."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["row_num"] = range(len(df))
    return df

def apply_substitution(text, mapping):
    """Whole-word find/replace for every (old, new) pair in mapping.

    Each old word also matches with an inflectional suffix tacked on - "rain"
    matches "rains"/"raining"/"rained" too - and that suffix carries over to
    the replacement (rain -> snow, raining -> snowing), so this only needs to
    be spelled once per stem. \\b...\\b avoids matching "rain" inside an
    unrelated longer word like "train"; case-sensitive since these words only
    ever appear capitalized (proper nouns) or lowercase (common nouns).
    """
    suffix_group = "|".join(INFLECTIONAL_SUFFIXES)
    for old, new in mapping.items():
        pattern = rf"\b{re.escape(old)}({suffix_group})?\b"
        text = re.sub(pattern, lambda m, new=new: new + (m.group(1) or ""), text)
    return text

def variant_texts(row_nums, text_lookup, variant):
    """Apply one lexical variant to a set of sentences, or None if it's a no-op.

    Substitution is applied to the whole sentence set at once (not sentence by
    sentence) so that a difference assertion's four sentences stay consistent
    with each other under the same variant.
    """
    base_texts = [text_lookup[r] for r in row_nums]
    if variant == BASE_VARIANT:
        return base_texts
    mapping = LEXICAL_VARIANTS[variant]
    substituted = [apply_substitution(t, mapping) for t in base_texts]
    # Re-capitalize the sentence-initial word: a substitution like John -> "the teacher"
    # would otherwise leave a lowercase sentence opener when John was the first word.
    substituted = [t[:1].upper() + t[1:] for t in substituted]
    return substituted if substituted != base_texts else None

def condition_matches(df, condition):
    """Rows of sentences.csv matching one config.json condition's phenomenon/subtype."""
    mask = df["phenomenon"] == condition["phenomenon"]
    subtype = condition.get("subtype")
    mask &= df["subtype"] == (subtype if subtype else "")
    return df[mask]


def iter_groups(rows, comparison_types):
    """Split a condition's matched rows into groups, one per distinct
    combination of the descriptive columns (SET_KEY_COLUMNS) that aren't
    themselves one of this condition's comparison_types.
    """
    key_cols = [c for c in SET_KEY_COLUMNS if c not in comparison_types]
    if not key_cols:
        yield rows
        return
    for _, group in rows.groupby(key_cols, sort=False):
        yield group


def comparison_grouping(rows, comparison_type, comparison):
    """The row pairs for one comparison_type within a set: group `rows` by the
    comparison_type column, and for each group of 2, pair the grammatical row
    with the ungrammatical one.

    For "direct" comparisons, every group becomes its own ("comparison",
    gram, ungram) pair.

    For "differences" comparisons: the group with exactly 1 grammatical row
    (mixed) is the "minuend", ordered (grammatical, ungrammatical); the other
    group - both grammatical or both ungrammatical - is the "subtrahend", in
    its original row order. This works for both a (1 ungrammatical, 3
    grammatical) split and a (3 ungrammatical, 1 grammatical) split across the
    two groups, since it's driven by each group's own grammaticality count
    rather than assuming which group is which.
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
            sorted_groups.append(("comparison", gram.iloc[0], ungram.iloc[0]))
        else:
            if len(gram) == 1:
                sorted_groups.append(("minuend", gram.iloc[0], ungram.iloc[0]))
            else:
                sorted_groups.append(("subtrahend", variant_group.iloc[0], variant_group.iloc[1]))
    return sorted_groups

def build_comparison(set_rows, meta, condition, variants, text_lookup, assertion_id):
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
            row_nums = sorted({row["row_num"] for _, row1, row2 in assertions for row in (row1, row2)})
            for variant in variants:
                texts = variant_texts(row_nums, text_lookup, variant)
                if texts is None:
                    continue
                text_by_row = dict(zip(row_nums, texts))
                base_row = {
                    **meta,
                    "comparison_type": comparison_type,
                    "condition_type": condition_type,
                    "lexical_variant": variant,
                    "assertion_id": assertion_id,
                }
                assertion_id += 1
                for pair_role, row1, row2 in assertions:
                    rows_out.append({
                        **base_row,
                        "pair_role": pair_role,
                        "sentence1": text_by_row[row1["row_num"]],
                        "sentence2": text_by_row[row2["row_num"]],
                    })
    return rows_out, assertion_id

def _invalid_phenomena_or_subtype(condition, phenomena, subtypes):
    return (phenomena and condition["phenomenon"] not in phenomena) or subtypes and condition.get("subtype") not in subtypes

def build_pairs(config, sentences_df, phenomena=None, subtypes=None, variants=None):
    """Build the full list of pair rows for every matching condition/set/variant."""
    variants = variants if variants is not None else ALL_VARIANTS
    text_lookup = dict(zip(sentences_df["row_num"], sentences_df["sentence"]))
    rows_out = []
    assertion_id = 0

    for condition in config["conditions"]:
        if _invalid_phenomena_or_subtype(condition, phenomena, subtypes):
            continue
        matched_sentences = condition_matches(sentences_df, condition)
        if matched_sentences.empty:
            raise ValueError(f"no sentences matched condition {condition.get('group_id')}")

        comparison = condition["comparison"]
        for set_rows in iter_groups(matched_sentences, condition["comparison_types"]):
            meta = {
                "phenomenon": condition["phenomenon"],
                "subtype": condition.get("subtype") or "",
                "group_id": condition.get("group_id"),
                "comparison": comparison,
                "tense": set_rows["tense"].iloc[0],
                "negation": set_rows["negation"].iloc[0],
                "matrix_subj": set_rows["matrix_subj"].iloc[0],
                "matrix_type": set_rows["matrix_type"].iloc[0],
            }
            new_rows, assertion_id = build_comparison(set_rows, meta, condition, variants, text_lookup, assertion_id)
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
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--sentences", default="sentences.csv")
    parser.add_argument("--output", default="pairs.csv")
    parser.add_argument("--phenomena", nargs="*", default=None, help="restrict to these phenomenon names")
    parser.add_argument("--subtypes", nargs="*", default=None, help="restrict to these subtypes")
    parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
        choices=ALL_VARIANTS,
        help="lexical variants to include (default: all)",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)
    sentences_df = load_sentences(args.sentences)

    rows = build_pairs(
        config,
        sentences_df,
        phenomena=args.phenomena,
        subtypes=args.subtypes,
        variants=args.variants,
    )
    write_pairs(rows, args.output)
    n_assertions = len({row["assertion_id"] for row in rows})
    print(f"Wrote {len(rows)} pairs ({n_assertions} assertions) to {args.output}")


if __name__ == "__main__":
    main()
