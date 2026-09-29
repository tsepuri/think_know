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
   by default) more full copies are generated per set. In a condition with
   free_embedded_clause, a copy first gets a different embedded clause
   (with probability embedded_clause_swap_prob); then a random
   min_slots_modified..max_slots_modified substitutions are stacked, drawn from
   whichever vocabulary slots actually match text present in that set's rows
   (now including the swapped-in clause). Both are set in variant_config.json,
   for every condition or overridden per condition code.

   A vocabulary entry's `requires` (<column>:<value>[,<value>..], see
   parse_requires) restricts where it may be swapped in; a substitution is
   never allowed to newly break one, whichever word it's on.
"""
import argparse
import collections
import json
import random
import re
import sys

import pandas as pd

import utils

# second person conditions are not expanded
PERSONS = ["first_person", "third_person"]
# slots whose words are swapped for one another wherever they appear. person has its own
# rename/pronoun handling, and embedded_clause/complementizer/determiner their own logic
SIMPLE_SLOTS = ["weather_predicate", "content_noun", "judgment", "object", "hiding_verb", "mishap_object", "mishap_verb"]
# the "content" a free_embedded_clause condition swaps first, before any of the above: a clause
# ("John knows where Mary hid the keys") or, in a mini-discourse, its wh-question ("A: Where are my keys?")
CLAUSE_SLOTS = ["embedded_clause", "wh_question"]

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

# (group_id, dimension, matrix_type) -> times a manipulation the config asks for couldn't be applied
SKIPPED = collections.Counter()


def _skip(row, dimension):
    SKIPPED[(row["group_id"], dimension, row["matrix_type"])] += 1


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
        result = utils.change_person(row["sentence"], row["matrix_subj"], row["matrix_verb"], base_person, person, row["tense"], row["matrix_type"])
        if result is None:
            _skip(row, "matrix_subj_category")
            continue
        new_sentence, new_subj = result
        rows.append({**row, "sentence": new_sentence, "matrix_subj_category": person, "matrix_subj": new_subj})
    return rows


def expand_matrix_type(row, manipulated_types, comparison_types):
    if not _wants("matrix_type", manipulated_types, comparison_types) or row["matrix_type"] != "declarative":
        return [row]
    new_sentence = utils.make_polar_question(row["sentence"], row["matrix_subj"], row["matrix_verb"], row["matrix_subj_category"], row["tense"])
    if new_sentence is None:
        _skip(row, "matrix_type")
        return [row]
    return [row, {**row, "sentence": new_sentence, "matrix_type": "polar_question"}]


def expand_tense(row, manipulated_types, comparison_types):
    if not _wants("tense", manipulated_types, comparison_types) or row["tense"] != "present":
        return [row]
    new_sentence = utils.change_tense(row["sentence"], row["matrix_subj"], row["matrix_verb"], row["matrix_subj_category"], "present", "past", row["matrix_type"])
    if new_sentence is None:
        _skip(row, "tense")
        return [row]
    return [row, {**row, "sentence": new_sentence, "tense": "past"}]


def expand_negation(row, manipulated_types, comparison_types):
    if not _wants("negation", manipulated_types, comparison_types) or row["negation"] != "no":
        return [row]
    new_sentence = utils.negate_clause(row["sentence"], row["matrix_subj"], row["matrix_verb"], row["matrix_subj_category"], row["tense"], row["matrix_type"])
    if new_sentence is None:
        _skip(row, "negation")
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
# Vocabulary + `requires`
# ---------------------------------------------------------------------------

DETERMINER_SLOT = "determiner" 
NOUN_SLOTS = ("content_noun", "object", "mishap_object")
NON_WH_COMPLEMENTIZERS = {"that", "whether"}
NON_FEATURE_COLUMNS = {"word", "slot", "requires", "notes"}


def parse_requires(text):
    """vocabulary.csv `requires` -> {column: {values}}. Syntax is
    <column>:<value>[,<value>...], with ';' between several requirements
    (all must hold). <column> is either a sentences.csv column - the row
    being varied must have one of the values, e.g. matrix_verb:think,know,
    paradigm:mini-discourse, complementizer:where,why - or a vocabulary.csv
    feature column - some word in the sentence must have one of the values,
    e.g. breakable:1 on a mishap_verb wants a breakable object present."""
    reqs = {}
    for part in (text or "").split(";"):
        part = part.strip()
        if not part:
            continue
        column, sep, values = part.partition(":")
        values = {v.strip() for v in values.split(",") if v.strip()}
        if not sep or not column.strip() or not values:
            raise ValueError(f"bad requires {text!r}: expected <column>:<value>[,<value>...]")
        reqs[column.strip()] = values
    return reqs


class Vocabulary:
    """vocabulary.csv plus the lookups `requires` checking needs. Raises on a
    malformed `requires` or one naming a column that's in neither
    sentences.csv nor vocabulary.csv; a value that never occurs in that
    sentences.csv column (or in the vocabulary slot of the same name, for
    e.g. complementizer) is only recorded in `warnings`, since it may just
    be a value no base sentence uses yet."""

    def __init__(self, vocab_df, sentences_df):
        self.records = vocab_df.to_dict("records")
        self.sentence_columns = set(sentences_df.columns)
        self.feature_columns = set(vocab_df.columns) - NON_FEATURE_COLUMNS
        self.warnings = []
        self._pools = {}
        for entry in self.records:
            self._pools.setdefault(entry["slot"], []).append(entry)
        self.wh_complementizers = {e["word"] for e in self.pool("complementizer")} - NON_WH_COMPLEMENTIZERS
        for slot in SIMPLE_SLOTS + CLAUSE_SLOTS + [DETERMINER_SLOT, "person", "pronoun", "complementizer"]:
            if not self.pool(slot):
                self.warnings.append(f"vocabulary.csv has no {slot!r} entries (a renamed slot?) - that slot won't be varied")
        for entry in self.pool(DETERMINER_SLOT):
            if entry["number"] not in ("sg", "pl", ""):
                self.warnings.append(f"determiner {entry['word']!r}: number {entry['number']!r} should be sg, pl, or blank for either")
        self.constrained = []  # (entry, parsed requires) for entries that have one
        for entry in self.records:
            try:
                reqs = parse_requires(entry["requires"])
            except ValueError as err:
                raise ValueError(f"vocabulary.csv, {entry['word']!r}: {err}") from None
            for column, values in reqs.items():
                self._check_requirement(entry, column, values, sentences_df)
            if reqs:
                self.constrained.append((entry, reqs))
        # a clause that lists no complementizers is for that/whether/none only
        # - a wh-word needs a clause that names it (see requires_violations)
        self.unlicensed_clauses = [e for e in self.pool("embedded_clause") if "complementizer" not in parse_requires(e["requires"])]
        self._words_in_cache = {}

    def _check_requirement(self, entry, column, values, sentences_df):
        if column in self.sentence_columns:
            known = set(sentences_df[column]) | {e["word"] for e in self.pool(column)}
            for value in sorted(values - known):
                self.warnings.append(f"{entry['word']!r} requires {column}:{value}, but no {column} in sentences.csv or vocabulary.csv is {value!r}")
        elif column not in self.feature_columns:
            raise ValueError(f"vocabulary.csv, {entry['word']!r}: requires column {column!r} is in neither sentences.csv nor vocabulary.csv")

    def pool(self, slot):
        return self._pools.get(slot, [])

    def noun(self, word):
        for slot in NOUN_SLOTS:
            for entry in self.pool(slot):
                if entry["word"].lower() == word.lower():
                    return entry
        return None

    def words_in(self, text):
        """Vocabulary entries whose word appears in `text`."""
        if text not in self._words_in_cache:
            self._words_in_cache[text] = [e for e in self.records if "<" not in e["word"] and _contains(text, e["word"])]
        return self._words_in_cache[text]


def _row_value(row, column, vocab):
    """row[column], except a wh_question's complementizer is the wh-word it
    fronts: 'Where does John know Mary hid the keys?' is 'none' in
    sentences.csv, but for `requires` purposes (and for varying it in step with
    its embedded-wh siblings) it's 'where'."""
    value = row[column]
    if column == "complementizer" and value not in vocab.wh_complementizers and row["matrix_type"] == "wh_question":
        first = re.match(r"\w+", row["sentence"])
        if first and first.group(0).lower() in vocab.wh_complementizers:
            return first.group(0).lower()
    return value


def requirement_met(column, values, row, text, vocab, context=None):
    """One parsed `requires` entry. A sentences.csv column is checked against
    the row; a vocabulary feature column against the vocabulary words present
    in `text` (or in `context`, when the requirement is about one specific
    word - a determiner's noun - rather than anything in the sentence)."""
    if column in vocab.sentence_columns:
        return _row_value(row, column, vocab) in values
    words = context if context is not None else vocab.words_in(text)
    return any(e.get(column) in values for e in words)


def requires_violations(rows, vocab):
    """(row index, slot, column) for every unmet `requires` among the
    vocabulary words present in `rows`. A substitution is safe when it leaves
    this no bigger than before - which grandfathers the unmet requirements
    that are the point of a control row (e.g. 'about' with 'John starts
    about the rumor') while catching a swap that newly breaks one: 'message'
    into a partition that has start rows, or 'vase' -> 'coffee' under 'broke'.
    Determiners are left out - their requirement is about their own noun, see
    find_determiner_candidates.

    Also counted: a wh-complementizer on a clause that doesn't list any
    ('Where did John know the message was true?') - only clauses whose
    `requires` names complementizers (e.g. 'Mary hid the keys') license one."""
    found = set()
    for i, row in enumerate(rows):
        present = None
        for entry, reqs in vocab.constrained:
            if entry["slot"] == DETERMINER_SLOT:
                continue
            if entry["slot"] == "embedded_clause":
                present = present if present is not None else _present_tensed(row)
                text = present
            else:
                text = row["sentence"]
            if not _contains(text, entry["word"]):
                continue
            for column, values in reqs.items():
                if not requirement_met(column, values, row, text, vocab):
                    found.add((i, entry["slot"], column))
        if _row_value(row, "complementizer", vocab) in vocab.wh_complementizers:
            present = present if present is not None else _present_tensed(row)
            if any(_contains(present, e["word"]) for e in vocab.unlicensed_clauses):
                found.add((i, "embedded_clause", "complementizer"))
    return found


# ---------------------------------------------------------------------------
# Lexical variation
# ---------------------------------------------------------------------------

# sentence-initial, or right after a discourse turn label ("A: Where are my keys?")
_SENTENCE_START = re.compile(r"(?:^|[.?!:]\s+)$")

DEFAULT_VARIATION = {
    "embedded_clause_swap_prob": 0.5,  # chance a variant first gets a different embedded clause
    "min_slots_modified": 1,           # then this many..
    "max_slots_modified": 2,           # ..to this many other slots modified
}


def variation_settings(variant_config, condition):
    """DEFAULT_VARIATION, overridden by variant_config.json's top-level
    settings, overridden in turn by its "conditions" entry for this
    condition's code (e.g. {"conditions": {"cpq": {"max_slots_modified": 3}}})."""
    top_level = {k: v for k, v in variant_config.items() if k != "conditions"}
    per_condition = variant_config.get("conditions", {}).get(condition.get("code"), {})
    settings = {**DEFAULT_VARIATION, **top_level, **per_condition}
    unknown = settings.keys() - DEFAULT_VARIATION.keys()
    if unknown:
        raise ValueError(f"variant_config.json: unknown setting(s) {sorted(unknown)}")
    if not 0 <= settings["min_slots_modified"] <= settings["max_slots_modified"]:
        raise ValueError(f"variant_config.json needs 0 <= min_slots_modified <= max_slots_modified, got {settings}")
    return settings


def _contains(sentence, word):
    return re.search(r"\b" + re.escape(word) + r"\b", sentence, re.IGNORECASE) is not None


def _replace_all(sentence, old, new):
    """Whole-word, case-insensitive replace; `new` is capitalized when it
    lands at the start of a sentence or discourse turn ('Where are my keys?'
    -> 'When did Mary eat the cookies?')."""
    def sub(match):
        at_start = _SENTENCE_START.search(sentence[:match.start()]) is not None
        return new[:1].upper() + new[1:] if at_start else new
    return re.sub(r"\b" + re.escape(old) + r"\b", sub, sentence, flags=re.IGNORECASE)


def _substitutable(rows, old, alternatives, vocab):
    """The `alternatives` to `old` that don't leave any `requires` newly unmet."""
    before = requires_violations(rows, vocab)
    return [a for a in alternatives if requires_violations(_apply_simple(rows, old, a), vocab) <= before]


def find_simple_candidates(rows, pool, vocab):
    """Words from `pool` that appear in at least one row's sentence, each
    mapped to alternatives from the same pool (all members, minus itself,
    and minus any that would newly break a `requires`)."""
    found = {}
    for entry in pool:
        word = entry["word"]
        if not any(_contains(r["sentence"], word) for r in rows):
            continue
        alternatives = _substitutable(rows, word, [e["word"] for e in pool if e["word"] != word], vocab)
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


def _determiner_pattern(dets, names):
    """One regex over every determiner (longest first, so 'some of the' wins
    over 'the'), each a named group d<i>, followed by its noun. '<person>'s'
    matches any proper name plus 's."""
    name_alt = "|".join(re.escape(n) for n in names)
    branches = []
    for i, det in sorted(enumerate(dets), key=lambda item: -len(item[1]["word"])):
        body = re.escape(det["word"]).replace(re.escape("<person>"), f"(?:{name_alt})")
        branches.append(f"(?P<d{i}>{body})")
    return re.compile(r"\b(?:" + "|".join(branches) + r")\s+(?P<noun>[A-Za-z]+)", re.IGNORECASE)


def _determiner_fits(det, noun_entry, rows, vocab):
    """A determiner's `number` (sg/pl) must match its noun's - 'a' only with
    singular nouns, 'all of the' only with plural ones; blank takes either.
    The determiner's own `requires` (e.g. countable:1) is checked against its
    noun's vocabulary entry. A noun with no entry (e.g. 'cookies') has no known
    number and can't satisfy a vocabulary-column requirement, so it only takes
    determiners with a blank number and no such requirement."""
    if det["number"] in ("sg", "pl") and not (noun_entry and noun_entry["number"] == det["number"]):
        return False
    context = [noun_entry] if noun_entry else []
    return all(
        requirement_met(column, values, r, r["sentence"], vocab, context)
        for r in rows for column, values in parse_requires(det["requires"]).items()
    )


def find_determiner_candidates(rows, vocab):
    """{'the keys': ['my keys', "Sarah's keys", ...]} - every determiner+noun
    phrase in `rows` mapped to the phrases the same noun could take instead.
    Persons' role nouns ('the man') are skipped: the determiner is part of
    that person entry, not a slot of its own."""
    dets = vocab.pool(DETERMINER_SLOT)
    persons = vocab.pool("person")
    names = [e["word"] for e in persons if not e["word"].startswith("the ")]
    if not dets:
        return {}
    pattern = _determiner_pattern(dets, names)
    role_nouns = {e["word"].lower() for e in persons}
    found = {}
    for r in rows:
        for m in pattern.finditer(r["sentence"]):
            if m.group(0).lower() in role_nouns or m.group(0) in found:
                continue
            det = next(d for i, d in enumerate(dets) if m.group(f"d{i}") is not None)
            noun = m.group("noun")
            affected = [row for row in rows if _contains(row["sentence"], m.group(0))]
            unused_names = [n for n in names if not any(_contains(row["sentence"], n) for row in rows)]
            alternatives = []
            for other in dets:
                if other is det or not _determiner_fits(other, vocab.noun(noun), affected, vocab):
                    continue
                if "<person>" in other["word"]:
                    alternatives.extend(other["word"].replace("<person>", n) + " " + noun for n in unused_names)
                else:
                    alternatives.append(other["word"] + " " + noun)
            if alternatives:
                found[m.group(0)] = alternatives
    return found


def _present_tensed(row):
    """`row`'s sentence normalized to present tense, via utils.change_tense,
    so embedded-clause detection/substitution can work against
    vocabulary.csv's present-tense-canonical clause forms (e.g. 'it is
    raining', 'the rumor is true'). No-op when the row is already present tense."""
    if row["tense"] == "present":
        return row["sentence"]
    reverted = utils.change_tense(row["sentence"], row["matrix_subj"], row["matrix_verb"], row["matrix_subj_category"], row["tense"], "present", row["matrix_type"])
    return reverted if reverted is not None else row["sentence"]


def _present_clauses(rows, vocab):
    """embedded_clause / wh_question entries whose text appears in at least one row."""
    return [e for slot in CLAUSE_SLOTS for e in vocab.pool(slot) if any(_contains(_present_tensed(r), e["word"]) for r in rows)]


def _wh_complementizer(rows, vocab):
    """The one wh-complementizer (where/when/why) `rows` share, else None."""
    current = {_row_value(r, "complementizer", vocab) for r in rows} & vocab.wh_complementizers
    return next(iter(current)) if len(current) == 1 else None


def find_complementizer_candidate(rows, vocab):
    """(current wh-word, alternatives): a wh-complementizer can change when
    an embedded clause in the rows lists several via its `requires`
    (e.g. 'Mary hid the keys' requires complementizer:where,when,why). The
    alternatives are the ones every such clause allows and no other
    `requires` objects to."""
    allowed = None
    for entry in _present_clauses(rows, vocab):
        values = parse_requires(entry["requires"]).get("complementizer")
        if values:
            allowed = values if allowed is None else allowed & values
    current = _wh_complementizer(rows, vocab)
    if not allowed or current not in allowed:
        return None
    before = requires_violations(rows, vocab)
    alternatives = [
        c for c in sorted((allowed - {current}) & vocab.wh_complementizers)
        if requires_violations(_apply_complementizer(rows, current, c, vocab), vocab) <= before
    ]
    return (current, alternatives) if alternatives else None


def find_embedded_clause_swap(rows, vocab):
    """(old clause, new clause, (old wh, new wh) or None) for a random
    swappable clause, or None - an embedded_clause is swapped for another
    embedded_clause, a mini-discourse's wh_question for another wh_question.
    A clause whose `requires` the rows
    already satisfy is swapped as-is; one that lists other complementizers
    (e.g. 'Mary ate the cookies' can't ask 'where') is also swapped in if
    the rows' wh-complementizer can change to one it allows."""
    present = _present_clauses(rows, vocab)
    if not present:
        return None
    old = present[0]["word"]
    before = requires_violations(rows, vocab)
    current = _wh_complementizer(rows, vocab)
    options = {}
    for entry in vocab.pool(present[0]["slot"]):
        new = entry["word"]
        if new == old:
            continue
        swapped = _apply_embedded_clause(rows, old, new)
        if requires_violations(swapped, vocab) <= before:
            options.setdefault(new, []).append(None)
            continue
        allowed = parse_requires(entry["requires"]).get("complementizer") or set()
        for comp in sorted((allowed - {current}) & vocab.wh_complementizers) if current else []:
            adapted = _apply_complementizer(swapped, current, comp, vocab)
            if requires_violations(adapted, vocab) <= before:
                options.setdefault(new, []).append((current, comp))
    if not options:
        return None
    new = random.choice(sorted(options))
    return old, new, random.choice(options[new])


def gather_candidates(rows, vocab):
    """All applicable substitution candidates for this group's rows, keyed
    by an arbitrary candidate id -> (apply function, label)."""
    candidates = {}

    for slot in SIMPLE_SLOTS:
        for word, alternatives in find_simple_candidates(rows, vocab.pool(slot), vocab).items():
            new_word = random.choice(alternatives)
            candidates[f"{slot}:{word}"] = (
                lambda rs, w=word, nw=new_word: _apply_simple(rs, w, nw),
                f"{slot}: {word}->{new_word}",
            )

    for phrase, alternatives in find_determiner_candidates(rows, vocab).items():
        new_phrase = random.choice(alternatives)
        candidates[f"determiner:{phrase}"] = (
            lambda rs, p=phrase, np=new_phrase: _apply_simple(rs, p, np),
            f"determiner: {phrase}->{new_phrase}",
        )

    complementizer_choice = find_complementizer_candidate(rows, vocab)
    if complementizer_choice:
        old, alternatives = complementizer_choice
        new = random.choice(alternatives)
        candidates["complementizer"] = (
            lambda rs, o=old, n=new: _apply_complementizer(rs, o, n, vocab),
            f"complementizer: {old}->{new}",
        )

    person_pool = vocab.pool("person")
    pronoun_pool = vocab.pool("pronoun")
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


def _apply_complementizer(rows, old, new, vocab):
    """Swap wh-complementizer `old` for `new` in every row it's the wh-word
    of - embedded ('Does John know where...') and fronted ('Where does John
    know...') alike, so a factive island's whole assertion set keeps asking
    about the same thing."""
    new_rows = []
    for r in rows:
        if _row_value(r, "complementizer", vocab) != old:
            new_rows.append(r)
            continue
        updates = {"sentence": _replace_all(r["sentence"], old, new)}
        if r["complementizer"] == old:
            updates["complementizer"] = new
        new_rows.append({**r, **updates})
    return new_rows


def _apply_clause_swap(rows, swap, vocab):
    old, new, complementizer_change = swap
    rows = _apply_embedded_clause(rows, old, new)
    if complementizer_change:
        rows = _apply_complementizer(rows, *complementizer_change, vocab)
    return rows


def _apply_pronoun(rows, word, pronoun):
    """Like _apply_mapping, also updates `matrix_subj` for the rows whose
    subject was actually swapped"""
    new_rows = []
    for r in rows:
        if r["matrix_subj"] == word:
            new_sentence = utils.repronoun_subject(r["sentence"], word, pronoun, r["matrix_verb"], r["tense"], r["matrix_type"], r["negation"])
            if new_sentence is None:
                new_rows.append(r)
            else:
                new_rows.append({**r, "sentence": new_sentence, "matrix_subj": pronoun})
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


def make_lexical_variant(rows, vocab, condition, settings):
    """One lexically-varied copy of an assertion set. The embedded clause
    (conditions with free_embedded_clause) goes first, with probability
    `embedded_clause_swap_prob`, so that everything after it - determiners,
    wh-words, mishap verbs.. - varies the clause that's actually there, the
    original or the new one. Then a random min_slots_modified..
    max_slots_modified of the other slots, determiners before the nouns
    they're attached to (a noun swapped first would leave the determiner's
    phrase unfindable). Each is re-checked against `requires` as it's applied,
    since the choices were made independently of one another."""
    if condition.get("free_embedded_clause") and random.random() < settings["embedded_clause_swap_prob"]:
        swap = find_embedded_clause_swap(rows, vocab)
        if swap:
            rows = _apply_clause_swap(rows, swap, vocab)
    candidates = gather_candidates(rows, vocab)
    if candidates:
        lo = min(settings["min_slots_modified"], len(candidates))
        hi = min(settings["max_slots_modified"], len(candidates))
        chosen = random.sample(list(candidates.items()), random.randint(lo, hi))
        chosen.sort(key=lambda item: not item[0].startswith("determiner:"))
        for _, (apply_fn, _label) in chosen:
            varied = apply_fn(rows)
            if requires_violations(varied, vocab) <= requires_violations(rows, vocab):
                rows = varied
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


# unique_id -> rows dropped because it matches a condition's bad_substitutions
EXCLUDED = collections.Counter()


def is_bad_substitution(unique_id, bad_substitutions):
    """True if every underscore-separated word of some config.json
    `bad_substitutions` entry is also a whole underscore-separated word of
    `unique_id` (in any order) - "pres_neg_1p" excludes
    "npcop_pres_neg_1p_variant_1" but not "npcop_pres_neg_3p_base"."""
    id_words = set(unique_id.split("_"))
    return any(set(bad.split("_")) <= id_words for bad in bad_substitutions)


def build_variants(sentences_df, config, vocab, variant_config, n_lexical_variants=2):
    conditions = config["conditions"]
    SKIPPED.clear()
    EXCLUDED.clear()
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
        settings = variation_settings(variant_config, condition)
        bad_substitutions = condition.get("bad_substitutions", [])
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
                # variants are still generated for excluded sets so the random draws,
                # and so every other set's output, match a run without bad_substitutions
                variant_sets = [("base", part_rows)] + [
                    (f"variant_{i}", make_lexical_variant(part_rows, vocab, condition, settings))
                    for i in range(1, n_lexical_variants + 1)
                ]
                for variant_id, variant_rows in variant_sets:
                    for r in variant_rows:
                        uid = make_unique_id(r, code, partition_dims, axis_tags, variant_id)
                        if is_bad_substitution(uid, bad_substitutions):
                            EXCLUDED[uid] += 1
                            continue
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
    parser.add_argument("--variant-config", default="benchmark_data/variant_config.json", help="lexical variation settings")
    parser.add_argument("--sentences", default="benchmark_data/sentences.csv")
    parser.add_argument("--vocabulary", default="benchmark_data/vocabulary.csv")
    parser.add_argument("--output", default="benchmark_data/sentences_with_variants.csv")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-variants", type=int, default=2, help="lexically-varied copies generated per assertion set, on top of the base copy")
    args = parser.parse_args()

    random.seed(args.seed)
    config = load_config(args.config)
    variant_config = load_config(args.variant_config)
    sentences_df = load_sentences(args.sentences)
    vocab = Vocabulary(load_vocabulary(args.vocabulary), sentences_df)
    for warning in vocab.warnings:
        print(f"warning: {warning}", file=sys.stderr)

    rows = build_variants(sentences_df, config, vocab, variant_config, n_lexical_variants=args.n_variants)
    for (group_id, dimension, matrix_type), n in sorted(SKIPPED.items()):
        print(f"warning: {group_id}: couldn't apply {dimension} to {matrix_type} rows ({n}x), left unchanged", file=sys.stderr)
    if EXCLUDED:
        print(f"Excluded {sum(EXCLUDED.values())} rows in {len(EXCLUDED)} sets matching bad_substitutions: {', '.join(sorted(EXCLUDED))}", file=sys.stderr)
    pd.DataFrame(rows).reindex(columns=OUTPUT_COLUMNS).to_csv(args.output, index=False)
    n_ids = len({row["variant_id"] for row in rows}) if rows else 0
    print(f"Wrote {len(rows)} rows ({args.n_variants + 1} variant_ids per set, {n_ids} distinct variant_id labels) to {args.output}")


if __name__ == "__main__":
    main()
