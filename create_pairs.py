"""Expand config.json + sentences.csv into ranked sentence pairs.

Each row is a pair of sentences, where sentence1 is expected to rank above sentence2. 
A "direct" assertion (logP(a) > logP(b)) is a single row, with pair_role
"comparison". A "difference" assertion (logP(a) - logP(b) > logP(c) - logP(d))
is two rows - (a, b) as "minuend" and (c, d) as "subtrahend" - sharing the same
integer `assertion_id`. While a "direct" assertion inherently involves one ungrammatical sentence,
only (b) in a "difference" assertion is necessarily ungrammatical. 

A "difference" assertion takes the minuend row's logprob margin
minus the subtrahend row's logprob margin.

By default, every base assertion is duplicated with a few lexical substitutions
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


def load_sentences(path):
    """sentence_id -> sentence text, as referenced by config.json's condition sents/predictions."""
    df = pd.read_csv(path, dtype=str)
    return df.set_index("sentence_id")["sentence"].to_dict()

def parse_condition_key(condition_key):
    """Split a "<tense>_<negation>_<matrix_subj>_<matrix_type>" key (e.g. "pres_aff_3p_decl")
    into its four dimensions, per config.json's schema, so eval.py can group/report on them.
    """
    tense, negation, matrix_subj, matrix_type = condition_key.split("_")
    return {"tense": tense, "negation": negation, "matrix_subj": matrix_subj, "matrix_type": matrix_type}

def apply_substitution(text, mapping):
    """Whole-word find/replace for every (old, new) pair in mapping.

    Each old word also matches with an inflectional suffix tacked on - "rain" matches
    "rains"/"raining"/"rained" too - and that suffix carries over to the replacement
    (rain -> snow, raining -> snowing), so this only needs to be spelled once per stem.
    \\b...\\b avoids matching "rain" inside an unrelated longer word like "train"; case
    -sensitive since these words only ever appear capitalized (proper nouns) or
    lowercase (common nouns) in one fixed position, so an exact match is all that's needed.
    """
    suffix_group = "|".join(INFLECTIONAL_SUFFIXES)
    for old, new in mapping.items():
        pattern = rf"\b{re.escape(old)}({suffix_group})?\b"
        text = re.sub(pattern, lambda m, new=new: new + (m.group(1) or ""), text)
    return text


def variant_texts(sentence_ids, sentences, variant):
    """Apply one lexical variant to every sentence in an assertion, or None if it's a no-op.

    Substitution is applied to the whole assertion's sentence set at once (not sentence by
    sentence) so that, e.g., a difference assertion's minuend and subtrahend pairs stay
    consistent with each other under the same variant.
    """
    base_texts = [sentences[sid] for sid in sentence_ids]
    if variant == BASE_VARIANT:
        return base_texts
    mapping = LEXICAL_VARIANTS[variant]
    substituted = [apply_substitution(t, mapping) for t in base_texts]
    # Re-capitalize the sentence-initial word: a substitution like John -> "the teacher" would
    # otherwise leave a lowercase sentence opener when John was the first word.
    substituted = [t[:1].upper() + t[1:] for t in substituted]
    # If nothing changed (e.g. this condition never mentions "rumor"), skip it - otherwise
    # every variant would emit an identical duplicate of the base assertion.
    return substituted if substituted != base_texts else None


def iter_conditions(config, phenomena=None, categories=None, subtypes=None):
    """Yield (phenomenon_dict, condition_key, condition_dict) for every condition that
    passes the optional phenomenon/category/subtype filters."""
    for phen in config["phenomena"]:
        if phenomena and phen["phenomenon"] not in phenomena:
            continue
        if categories and phen.get("category") not in categories:
            continue
        if subtypes and phen.get("subtype") not in subtypes:
            continue
        for condition_key, condition in phen["conditions"].items():
            yield phen, condition_key, condition

def pairs_for_assertion(sentence_ids, comparison_type):
    """Split one assertion's sentence ids into (pair_role, sentence1_id, sentence2_id) tuples.

    Per config.json's schema:
      - assertion_direct     [a, b]       means logP(a) > logP(b)                     -> 1 pair
      - assertion_difference [a, b, c, d] means logP(a) - logP(b) > logP(c) - logP(d) -> 2 pairs
    """
    if comparison_type == "direct":
        a, b = sentence_ids
        return [("comparison", a, b)]
    if comparison_type == "difference":
        a, b, c, d = sentence_ids
        return [("minuend", a, b), ("subtrahend", c, d)]
    raise ValueError(f"Unknown comparison_type: {comparison_type}")

def build_pairs(config, sentences, phenomena=None, categories=None, subtypes=None, variants=None):
    """Build the full list of pair rows: every assertion in every matching condition,
    times every requested lexical variant (skipping variants that are no-ops).

    Returns a flat list of row dicts, ready to pass to write_pairs.
    """
    variants = variants if variants is not None else ALL_VARIANTS
    rows = []
    assertion_id = 0  # one id per (condition, prediction_type, assertion, lexical_variant) instance
    for phen, condition_key, condition in iter_conditions(config, phenomena, categories, subtypes):
        cond_parts = parse_condition_key(condition_key)
        for prediction_type, assertions in condition["predictions"].items():
            for sentence_ids in assertions:
                for variant in variants:
                    texts = variant_texts(sentence_ids, sentences, variant)
                    if texts is None:
                        continue  # substitution had no effect on this assertion; skip the duplicate
                    text_by_id = dict(zip(sentence_ids, texts))
                    base_row = {
                        "phenomenon": phen["phenomenon"],
                        "category": phen.get("category"),
                        "subtype": phen.get("subtype", ""),
                        "comparison_type": phen["comparison_type"],
                        "condition_key": condition_key,
                        **cond_parts,
                        "prediction_type": prediction_type,
                        "lexical_variant": variant,
                        "assertion_id": assertion_id,
                    }
                    assertion_id += 1
                    for pair_role, id1, id2 in pairs_for_assertion(sentence_ids, phen["comparison_type"]):
                        rows.append({
                            **base_row,
                            "pair_role": pair_role,
                            "sentence1_text": text_by_id[id1],
                            "sentence2_text": text_by_id[id2],
                        })
    return rows

def write_pairs(rows, output_path):
    """Write pair rows to a CSV, using the first row's keys as the column order."""
    if not rows:
        raise ValueError("No pairs generated - check --phenomena/--categories/--subtypes filters")
    pd.DataFrame(rows, columns=list(rows[0].keys())).to_csv(output_path, index=False)

def main():
    """CLI entry point: parse args, load config.json + sentences.csv, write pairs.csv."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--sentences", default="sentences.csv")
    parser.add_argument("--output", default="pairs.csv")
    parser.add_argument("--phenomena", nargs="*", default=None, help="restrict to these phenomenon names")
    parser.add_argument("--categories", nargs="*", default=None, help="restrict to these categories")
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
    sentences = load_sentences(args.sentences)

    rows = build_pairs(
        config,
        sentences,
        phenomena=args.phenomena,
        categories=args.categories,
        subtypes=args.subtypes,
        variants=args.variants,
    )
    write_pairs(rows, args.output)
    n_assertions = len({row["assertion_id"] for row in rows})
    print(f"Wrote {len(rows)} pairs ({n_assertions} assertions) to {args.output}")


if __name__ == "__main__":
    main()
