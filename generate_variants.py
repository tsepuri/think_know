"""Expand sentences.csv into sentences_with_variants.csv, used by create_pairs.py.

Two independent expansion passes, per condition in config.json:

1. Structural expansion (utils.py conjugation tables): for each dimension in
   a condition's `manipulated_types` that isn't already one of its
   `comparison_types` (those already exist as separate authored rows), add
   sibling rows - tense (present->past), matrix_subj_category (the other
   persons), matrix_type (declarative->polar_question), negation, and
   complementizer (that-deletion). Every helper returns None when the
   expected literal text isn't found (e.g. a matrix verb already in bare
   do-support form), and the row is just left as-is rather than guessed at.

2. Lexical variation (vocabulary.csv): applied once per assertion-forming set
   (the same 4/6-sentence unit comparison_grouping pairs in create_pairs.py),
   not per row and not per whole group_id, so a "differences" assertion's
   sentences stay about the same rumor/weather/mishap instance while
   different tense/person/etc. combinations within one group_id still get
   independently varied. On top of the unchanged base copy, --n-variants (2
   by default) more full copies are generated per set, each stacking a
   random 1-3 substitutions drawn from whichever vocabulary slots actually
   match text present in that set's rows.
"""
import argparse
import json
import random
import re

import pandas as pd

import utils

# second person conditions are not expanded
PERSONS = ["first_person", "third_person"]
SIMPLE_SLOTS = ["person", "weather_predicate", "content_noun", "judgment", "object", "hiding_verb"]

def load_config(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_sentences(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return df


def load_vocabulary(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return df


def condition_for_row(conditions, row):
    for c in conditions:
        if c["phenomenon"] == row["phenomenon"] and (c.get("subtype") or "") == (row["subtype"] or ""):
            return c
    return None


# ---------------------------------------------------------------------------
# Structural expansion
# ---------------------------------------------------------------------------

def _wants(dimension, manipulated_types, comparison_types):
    return dimension in manipulated_types and dimension not in comparison_types

# TODO: abstract expand method to get rid of duplicate code
def expand_person(row, manipulated_types, comparison_types):
    if not _wants("matrix_subj_category", manipulated_types, comparison_types):
        return [row]
    rows = [row]
    base_person = row["matrix_subj_category"]
    for person in PERSONS:
        if person == base_person or base_person not in PERSONS:
            continue
        result = utils.change_person(row["sentence"], row["matrix_subj"], row["matrix_verb"], base_person, person, row["tense"])
        if result is None:
            continue
        new_sentence, new_subj = result
        rows.append({**row, "sentence": new_sentence, "matrix_subj_category": person, "matrix_subj": new_subj})
    return rows


def expand_matrix_type(row, manipulated_types, comparison_types):
    if not _wants("matrix_type", manipulated_types, comparison_types) or row["matrix_type"] != "declarative":
        return [row]
    new_sentence = utils.make_polar_question(row["sentence"], row["matrix_subj"], row["matrix_verb"], row["matrix_subj_category"], row["tense"])
    if new_sentence is None:
        return [row]
    return [row, {**row, "sentence": new_sentence, "matrix_type": "polar_question"}]


def expand_tense(row, manipulated_types, comparison_types):
    if not _wants("tense", manipulated_types, comparison_types) or row["tense"] != "present":
        return [row]
    new_sentence = utils.change_tense(row["sentence"], row["matrix_subj"], row["matrix_verb"], row["matrix_subj_category"], "present", "past", row["matrix_type"])
    if new_sentence is None:
        return [row]
    return [row, {**row, "sentence": new_sentence, "tense": "past"}]


def expand_negation(row, manipulated_types, comparison_types):
    if not _wants("negation", manipulated_types, comparison_types) or row["negation"] != "no":
        return [row]
    new_sentence = utils.negate_clause(row["sentence"], row["matrix_subj"], row["matrix_verb"], row["matrix_subj_category"], row["tense"], row["matrix_type"])
    if new_sentence is None:
        return [row]
    return [row, {**row, "sentence": new_sentence, "negation": "yes"}]


def expand_structural(row, manipulated_types, comparison_types):
    """Tense/person/matrix_type/negation each add a sibling row within the
    same assertion set - every comparison_type value gets the same treatment,
    so pairing stays intact. Complementizer and preposition are deliberately
    not included here for now since they only vary one side of its paired
    comparison_type (complementizer: only the declarative side, never
    'whether'; preposition: only the PP side, never NP's constant 'NA'), so
    neither can be a per-row sibling without orphaning the never-varying side
    into its own group - see duplicate_for_axis, which instead duplicates the set."""
    rows = [row]
    for step in (expand_person, expand_matrix_type, expand_tense, expand_negation):
        next_rows = []
        for r in rows:
            next_rows.extend(step(r, manipulated_types, comparison_types))
        rows = next_rows
    return rows


def _toggle_complementizer_row(r):
    new_sentence = utils.toggle_complementizer(r["sentence"], r["complementizer"])
    return None if new_sentence is None else (new_sentence, {"complementizer": "none"})


def _toggle_preposition_row(r):
    result = utils.toggle_preposition(r["sentence"], r["matrix_verb"], r["matrix_subj_category"], r["tense"], r["preposition"])
    return None if result is None else (result[0], {"preposition": result[1]})


# (dimension, unmodified-state tag, toggled-state tag, per-row toggle fn) - tags are the
# literal value that column takes (matches "complementizer"/"preposition" column values),
# so they're meaningful on their own in unique_id, not arbitrary labels like "orig".
AXES = [
    ("complementizer", "that", "none", _toggle_complementizer_row),
    ("preposition", "about", "of", _toggle_preposition_row),
]


def duplicate_for_axis(rows, dimension, unmodified_tag, toggled_tag, manipulated_types, comparison_types, toggle_row):
    """Yield (tag, rows) pairs for one axis. If `dimension` isn't a wanted
    manipulated_type, yields a single (None, rows) - axis not applicable,
    no tag needed. Otherwise yields (unmodified_tag, rows) - the unchanged
    set - and, if any row actually changes, (toggled_tag, rows) - a second
    full copy with `toggle_row` applied wherever it succeeds (rows where it
    doesn't - e.g. the comparison_type side that never varies along this
    axis - pass through unchanged, so they end up correctly present in
    *both* copies instead of orphaned into their own group)."""
    if not _wants(dimension, manipulated_types, comparison_types):
        yield None, rows
        return
    yield unmodified_tag, rows
    toggled = []
    changed = False
    for r in rows:
        result = toggle_row(r)
        if result is None:
            toggled.append(r)
        else:
            new_sentence, updates = result
            toggled.append({**r, "sentence": new_sentence, **updates})
            changed = True
    if changed:
        yield toggled_tag, toggled


def duplicate_for_axes(structural_rows, manipulated_types, comparison_types):
    """Chain duplicate_for_axis over every axis (complementizer, preposition),
    yielding (axis_tags, rows) - axis_tags is a tuple of the tags that apply
    to this copy (empty if neither axis is manipulated for this condition)."""
    copies = [((), structural_rows)]
    for dimension, unmod_tag, tog_tag, toggle_row in AXES:
        next_copies = []
        for existing_tags, rows in copies:
            for axis_tag, axis_rows in duplicate_for_axis(rows, dimension, unmod_tag, tog_tag, manipulated_types, comparison_types, toggle_row):
                new_tags = existing_tags if axis_tag is None else existing_tags + (axis_tag,)
                next_copies.append((new_tags, axis_rows))
        copies = next_copies
    return copies


# ---------------------------------------------------------------------------
# Lexical variation
# ---------------------------------------------------------------------------

def build_slot_pool(vocab_df, slot):
    return vocab_df[vocab_df["slot"] == slot].to_dict("records")


def _contains(sentence, word):
    return re.search(r"\b" + re.escape(word) + r"\b", sentence) is not None


def _replace_all(sentence, old, new):
    return re.sub(r"\b" + re.escape(old) + r"\b", new, sentence)


def _verb_compatible(entry, rows):
    """True if `entry` has no `requires` list, or every row in the partition
    has a matrix_verb on it - e.g. content_noun "message" requires
    "think,know": a partition mixing critical (think/know) and control
    (start) rows for the same tense/person/type must not offer "message" as
    an alternative, since "John starts the message" is not felicitious.
    TODO: fix this so that it instead just does 
    not include controls if message is chosen at the replacement?"""
    requires = [v.strip() for v in (entry.get("requires") or "").split(",") if v.strip()]
    return not requires or all(r["matrix_verb"] in requires for r in rows)


def find_simple_candidates(rows, pool):
    """Words from `pool` that appear in at least one row's sentence, each
    mapped to alternatives from the same pool (all members, minus itself,
    and minus any not verb_compatible with every row in this partition)."""
    found = {}
    for entry in pool:
        word = entry["word"]
        if not any(_contains(r["sentence"], word) for r in rows):
            continue
        alternatives = [e["word"] for e in pool if e["word"] != word and _verb_compatible(e, rows)]
        if alternatives:
            found[word] = alternatives
    return found


def find_feature_matched_candidates(rows, pool, feature_cols):
    """Like find_simple_candidates, but alternatives are restricted to pool
    entries sharing the same feature values (so a swap never breaks a
    dependent slot - e.g. keeping 'broke' compatible with the still-present
    'breakable' object, or keeping a swapped object compatible with the
    still-present verb)."""
    by_word = {entry["word"]: entry for entry in pool}
    found = {}
    for word, entry in by_word.items():
        if not any(_contains(r["sentence"], word) for r in rows):
            continue
        alternatives = [
            w for w, other in by_word.items()
            if w != word and all(other.get(c) == entry.get(c) for c in feature_cols)
        ]
        if alternatives:
            found[word] = alternatives
    return found


def find_person_rename_candidates(rows, person_pool):
    words = [entry["word"] for entry in person_pool]
    present = [w for w in words if any(_contains(r["sentence"], w) for r in rows)]
    return present


def find_pronoun_candidate(rows, person_pool, pronoun_pool):
    """Only works for the matrix subject (not e.g. a content-referenced
    name like "Mary" in "about Mary") - that's the one position we can
    reconjugate correctly; TODO: work for object-position pronoun as well. 
    Matches are by number and animacy, not gender. "they" is only offered when every row
    sharing that subject is a shape utils.repronoun_subject knows how to
    reconjugate (declarative, negated or not, or polar_question). in the future "they"
    should work with all sentence forms """
    by_word = {entry["word"]: entry for entry in person_pool}
    subj_rows = {}
    for r in rows:
        if r["matrix_subj"] in by_word and _contains(r["sentence"], r["matrix_subj"]):
            subj_rows.setdefault(r["matrix_subj"], []).append(r)
    if not subj_rows:
        return None
    word = random.choice(list(subj_rows))
    entry = by_word[word]
    matches = [p for p in pronoun_pool if p["number"] == entry["number"] and p["animate"] == entry["animate"]]
    if not all(r["matrix_type"] in ("declarative", "polar_question") for r in subj_rows[word]):
        matches = [p for p in matches if p["word"] != "they"]
    if not matches:
        return None
    return word, random.choice(matches)["word"]


def _present_tensed(row):
    """`row`'s sentence normalized to present tense, via utils.change_tense,
    so embedded-clause detection/substitution can work against
    vocabulary.csv's present-tense-canonical clause forms (e.g. 'it is
    raining', 'the rumor is true') - there's no separate past-tense entry.
    No-op when the row is already present tense."""
    if row["tense"] == "present":
        return row["sentence"]
    reverted = utils.change_tense(row["sentence"], row["matrix_subj"], row["matrix_verb"], row["matrix_subj_category"], row["tense"], "present", row["matrix_type"])
    return reverted if reverted is not None else row["sentence"]


def find_embedded_clause_candidate(rows, clause_pool):
    present_word = None
    # needs to contain an embedded clause to be able to replace
    for entry in clause_pool:
        if any(_contains(_present_tensed(r), entry["word"]) for r in rows):
            present_word = entry["word"]
            break
    if present_word is None:
        return None
    alternatives = [e["word"] for e in clause_pool if e["word"] != present_word]
    if not alternatives:
        return None
    return present_word, alternatives


def gather_candidates(rows, vocab_df, condition):
    """All applicable substitution candidates for this group's rows, keyed
    by an arbitrary candidate id -> a zero-arg apply function."""
    candidates = {}

    for slot in SIMPLE_SLOTS:
        pool = build_slot_pool(vocab_df, slot)
        if slot == "person":
            continue  # person has its own rename/pronoun handling below
        for word, alternatives in find_simple_candidates(rows, pool).items():
            new_word = random.choice(alternatives)
            candidates[f"{slot}:{word}"] = (
                lambda rs, w=word, nw=new_word: _apply_simple(rs, w, nw),
                f"{slot}: {word}->{new_word}",
            )
    # TODO: simplify these to work more straightforward like simple slots
    mishap_object_pool = build_slot_pool(vocab_df, "mishap_object")
    mishap_verb_pool = build_slot_pool(vocab_df, "mishap_verb")
    for word, alternatives in find_feature_matched_candidates(rows, mishap_object_pool, ["liquid", "breakable"]).items():
        new_word = random.choice(alternatives)
        candidates[f"mishap_object:{word}"] = (
            lambda rs, w=word, nw=new_word: _apply_simple(rs, w, nw),
            f"mishap_object: {word}->{new_word}",
        )
    for word, alternatives in find_feature_matched_candidates(rows, mishap_verb_pool, ["requires"]).items():
        new_word = random.choice(alternatives)
        candidates[f"mishap_verb:{word}"] = (
            lambda rs, w=word, nw=new_word: _apply_simple(rs, w, nw),
            f"mishap_verb: {word}->{new_word}",
        )

    person_pool = build_slot_pool(vocab_df, "person")
    pronoun_pool = build_slot_pool(vocab_df, "pronoun")
    rename_present = find_person_rename_candidates(rows, person_pool)
    if rename_present:
        pool_words = [e["word"] for e in person_pool]
        available = [w for w in pool_words if w not in rename_present]
        if available:
            n = min(len(rename_present), len(available))
            new_words = random.sample(available, n)
            mapping = dict(zip(rename_present[:n], new_words))
            candidates["person:rename"] = (
                lambda rs, m=mapping: _apply_mapping(rs, m),
                "person: " + "; ".join(f"{k}->{v}" for k, v in mapping.items()),
            )
    pronoun_choice = find_pronoun_candidate(rows, person_pool, pronoun_pool)
    if pronoun_choice:
        word, pronoun = pronoun_choice
        candidates["person:pronoun"] = (
            lambda rs, w=word, p=pronoun: _apply_pronoun(rs, w, p),
            f"person: {word}->{pronoun}",
        )

    if condition.get("free_embedded_clause"):
        clause_pool = build_slot_pool(vocab_df, "embedded_clause")
        clause_choice = find_embedded_clause_candidate(rows, clause_pool)
        if clause_choice:
            word, alternatives = clause_choice
            new_word = random.choice(alternatives)
            candidates["embedded_clause"] = (
                lambda rs, w=word, nw=new_word: _apply_embedded_clause(rs, w, nw),
                f"embedded_clause: {word}->{new_word}",
            )

    return candidates


def _apply_simple(rows, old, new):
    return [{**r, "sentence": _replace_all(r["sentence"], old, new)} for r in rows]


def _apply_embedded_clause(rows, old, new):
    """Substitute the embedded clause on the present-tensed form of each row
    (matching vocabulary.csv's present-tense-canonical entries), then reapply change_tense."""
    new_rows = []
    for r in rows:
        substituted = _replace_all(_present_tensed(r), old, new)
        if r["tense"] == "present":
            new_sentence = substituted
        else:
            new_sentence = utils.change_tense(substituted, r["matrix_subj"], r["matrix_verb"], r["matrix_subj_category"], "present", r["tense"], r["matrix_type"])
            if new_sentence is None:
                new_sentence = substituted
        new_rows.append({**r, "sentence": new_sentence})
    return new_rows


def _apply_pronoun(rows, word, pronoun):
    new_rows = []
    for r in rows:
        if r["matrix_subj"] == word:
            new_sentence = utils.repronoun_subject(r["sentence"], word, pronoun, r["matrix_verb"], r["tense"], r["matrix_type"], r["negation"])
            new_rows.append({**r, "sentence": new_sentence if new_sentence is not None else r["sentence"]})
        else:
            new_rows.append({**r, "sentence": _replace_all(r["sentence"], word, pronoun)})
    return new_rows


def _apply_mapping(rows, mapping):
    """Renames a person word wherever it appears in the sentence text, and
    also updates `matrix_subj` when the renamed word was a row's own
    matrix_subj - otherwise that column goes stale (still says "John") even
    though the sentence now says "Christopher", which would then confuse
    find_pronoun_candidate/_apply_pronoun on a later pass."""
    new_rows = []
    for r in rows:
        sentence = r["sentence"]
        for old, new in mapping.items():
            sentence = _replace_all(sentence, old, new)
        matrix_subj = mapping.get(r["matrix_subj"], r["matrix_subj"])
        new_rows.append({**r, "sentence": sentence, "matrix_subj": matrix_subj})
    return new_rows


def clean_sentence(sentence):
    """Collapse runs of whitespace and drop leading/trailing space."""
    return re.sub(r" +", " ", sentence).strip()


def _capitalize_first(rows):
    cleaned = [clean_sentence(r["sentence"]) for r in rows]
    return [{**r, "sentence": s[:1].upper() + s[1:]} for r, s in zip(rows, cleaned)]


def make_lexical_variant(rows, vocab_df, condition, min_n=1, max_n=3):
    candidates = gather_candidates(rows, vocab_df, condition)
    if not candidates:
        return _capitalize_first(rows)
    n = random.randint(min_n, min(max_n, len(candidates)))
    chosen = random.sample(list(candidates.items()), n)
    for _, (apply_fn, _label) in chosen:
        rows = apply_fn(rows)
    return _capitalize_first(rows)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

STRUCTURAL_DIMENSIONS = ["tense", "negation", "matrix_subj_category", "matrix_type"]


def already_varies(group_df, dimension):
    """True if the group's own base rows already differ along `dimension`
    (e.g. GRP010's Slifting rows separately author both matrix_subj_category
    values). Auto-generating that dimension on top would duplicate what's
    already there, so it's treated the same as a comparison_type - already
    covered, skip."""
    return group_df[dimension].nunique() > 1


def partition_by_dimensions(rows, dims):
    """Split `rows` into the actual 4/6-sentence assertion-forming sets - one
    per distinct combination of `dims` (the structural dimensions this
    condition auto-generates, e.g. tense/person/matrix_type). Lexical
    variation is then chosen fresh per partition, not once for the whole
    group_id, so different tense/person/etc. combinations get independently
    varied instead of all sharing one substitution."""
    partitions = {}
    for r in rows:
        key = tuple(r.get(d) for d in dims)
        partitions.setdefault(key, []).append(r)
    return list(partitions.values())


# Abbreviations for unique_id - only the human-readability layer
PERSON_ABBREV = {"first_person": "1p", "second_person": "2p", "third_person": "3p"}
TENSE_ABBREV = {"present": "pres", "past": "past"}
NEGATION_ABBREV = {"no": "aff", "yes": "neg"}
MATRIX_TYPE_ABBREV = {"declarative": "decl", "polar_question": "pol", "wh_question": "wh"}
DIMENSION_ABBREV = {
    "matrix_subj_category": PERSON_ABBREV,
    "tense": TENSE_ABBREV,
    "negation": NEGATION_ABBREV,
    "matrix_type": MATRIX_TYPE_ABBREV,
}


def make_unique_id(row, code, partition_dims, axis_tags, variant_id):
    """A human-readable identifier for the assertion-forming set (partition +
    axis copy + lexical variant) this row belongs to, e.g.
    "cpq_3p_pres_aff_decl_that_base". Doubles as create_pairs.py's entire
    grouping key: it already excludes comparison_types (matrix_verb,
    embedded_type still vary *within* one unique_id's rows, which is exactly
    what pairing needs) and includes everything that must stay aligned
    across a differences assertion's 4/6 sentences."""
    parts = [code]
    for dim in partition_dims:
        abbrev = DIMENSION_ABBREV.get(dim, {})
        parts.append(abbrev.get(row.get(dim), row.get(dim)))
    parts.extend(axis_tags)
    parts.append(variant_id)
    return "_".join(str(p) for p in parts)


def build_variants(sentences_df, config, vocab_df, n_lexical_variants=2):
    conditions = config["conditions"]
    out_rows = []
    for group_id, group_df in sentences_df.groupby("group_id", sort=False):
        group_rows = group_df.to_dict("records")
        condition = None
        for row in group_rows:
            condition = condition_for_row(conditions, row) or condition
        condition = condition or {}
        code = condition.get("code", group_id)
        manipulated_types = condition.get("manipulated_types", [])
        comparison_types = condition.get("comparison_types", [])
        effective_manipulated_types = [
            d for d in manipulated_types
            if d not in STRUCTURAL_DIMENSIONS or not already_varies(group_df, d)
        ]
        partition_dims = [d for d in effective_manipulated_types if d in STRUCTURAL_DIMENSIONS]

        structural_rows = []
        for row in group_rows:
            structural_rows.extend(expand_structural(row, effective_manipulated_types, comparison_types))

        for axis_tags, axis_rows in duplicate_for_axes(structural_rows, manipulated_types, comparison_types):
            for part_rows in partition_by_dimensions(axis_rows, partition_dims):
                for r in part_rows:
                    uid = make_unique_id(r, code, partition_dims, axis_tags, "base")
                    out_rows.append({**r, "unique_id": uid, "variant_id": "base"})
                for i in range(1, n_lexical_variants + 1):
                    variant_id = f"variant_{i}"
                    variant_rows = make_lexical_variant(part_rows, vocab_df, condition)
                    for r in variant_rows:
                        uid = make_unique_id(r, code, partition_dims, axis_tags, variant_id)
                        out_rows.append({**r, "unique_id": uid, "variant_id": variant_id})

    for i, row in enumerate(out_rows, start=1):
        row["sent_id"] = i
    return out_rows


OUTPUT_COLUMNS = [
    "sent_id", "group_id", "item_in_set", "category", "phenomenon", "subtype", "paradigm",
    "condition", "tense", "negation", "matrix_subj_category", "matrix_subj", "matrix_type",
    "matrix_verb", "embedded_type", "complementizer", "preposition", "grammaticality",
    "sentence", "unique_id", "variant_id", "notes",
]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="benchmark_data/config.json")
    parser.add_argument("--sentences", default="benchmark_data/sentences.csv")
    parser.add_argument("--vocabulary", default="benchmark_data/vocabulary.csv")
    parser.add_argument("--output", default="benchmark_data/sentences_with_variants.csv")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-variants", type=int, default=2, help="lexically-varied copies generated per assertion set, on top of the base copy")
    args = parser.parse_args()

    random.seed(args.seed)
    config = load_config(args.config)
    sentences_df = load_sentences(args.sentences)
    vocab_df = load_vocabulary(args.vocabulary)

    rows = build_variants(sentences_df, config, vocab_df, n_lexical_variants=args.n_variants)
    pd.DataFrame(rows).reindex(columns=OUTPUT_COLUMNS).to_csv(args.output, index=False)
    n_ids = len({row["variant_id"] for row in rows}) if rows else 0
    print(f"Wrote {len(rows)} rows ({args.n_variants + 1} variant_ids per set, {n_ids} distinct variant_id labels) to {args.output}")


if __name__ == "__main__":
    main()
