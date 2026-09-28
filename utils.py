"""Small lookup-table conjugation helpers for the closed set of matrix verbs
used across sentences.csv (think, know, wonder, start, guess) and subjects
(I, you, John). Not a general grammar engine - each function locates a known
literal substring (the subject word, a conjugated verb form) in `sentence`
and splices in the replacement, returning None if that substring isn't found
so callers can skip a variant rather than emit a corrupted sentence.
"""
import csv
import functools
import os
import re

SUBJECT_WORD = {"first_person": "I", "second_person": "you", "third_person": "John"}

# (non-3sg present, 3sg present), past (no person agreement), bare (used after do-support)
# TODO: maybe move to vocabulary.csv
VERB_FORMS = {
    "think": {"present": ("think", "thinks"), "past": "thought", "bare": "think"},
    "know": {"present": ("know", "knows"), "past": "knew", "bare": "know"},
    "wonder": {"present": ("wonder", "wonders"), "past": "wondered", "bare": "wonder"},
    "start": {"present": ("start", "starts"), "past": "started", "bare": "start"},
    "guess": {"present": ("guess", "guesses"), "past": "guessed", "bare": "guess"},
}

COPULA = {"present": "is", "past": "was"}
COPULA_SWAP = {
    ("present", "past"): {"is": "was", "are": "were"},
    ("past", "present"): {"was": "is", "were": "are"},
}
DO_PAST = "did"
QUESTION_TYPES = ("wh_question", "polar_question")


def conjugate(verb, person, tense):
    """Finite form of `verb` for `person`/`tense`, e.g. conjugate("think", "third_person", "present") -> "thinks".
    Any person other than the literal "third_person" gets the non-3sg form -
    this is also how a "they" agreement class is represented (see
    repronoun_subject): pass any person string besides "third_person"."""
    forms = VERB_FORMS[verb]
    if tense == "past":
        return forms["past"]
    non3sg, third = forms["present"]
    return third if person == "third_person" else non3sg


def do_form(person, tense):
    """do/does/did for a given person/tense - "does" only for third_person,
    "do" for everything else (I/you/they all take non-3sg do-support)."""
    if tense == "past":
        return DO_PAST
    return "does" if person == "third_person" else "do"


def _splice(sentence, old, new, count=1):
    """Whole-word/phrase substring replace; re-capitalizes if the match was sentence-initial.
    Returns None if `old` isn't found (callers should skip the variant, not guess)."""
    pattern = r"\b" + re.escape(old) + r"\b"
    if not re.search(pattern, sentence):
        return None
    was_initial = re.match(pattern, sentence) is not None
    result = re.sub(pattern, lambda m: new, sentence, count=count)
    if was_initial and result:
        result = result[0].upper() + result[1:]
    return result

def _splice_aux(sentence, old, new):
    """Swap a do-support auxiliary, which is capitalized when sentence-initial
    ('Does John know...?'). Returns None if neither casing is found."""
    return _splice(sentence, old, new) or _splice(sentence, old.capitalize(), new.capitalize())


def _swap_embedded_copula(sentence, from_tense, to_tense):
    """Swap the first is/are (or was/were) for the other tense; unchanged if there is none."""
    mapping = COPULA_SWAP[(from_tense, to_tense)]
    pattern = r"\b(" + "|".join(mapping) + r")\b"
    return re.sub(pattern, lambda m: mapping[m.group(1)], sentence, count=1)


# TODO: refactor change methods to avoid duplicate code
def change_person(sentence, matrix_subj, matrix_verb, from_person, to_person, tense, matrix_type):
    """Swap the matrix subject and re-agree the verb. In a declarative that
    means reconjugating the verb ('John thinks' -> 'I think'); in a wh/polar
    question the verb stays bare and agreement lives on the fronted do-support
    auxiliary ('Does John know' -> 'Do I know').
    Returns (new_sentence, new_matrix_subj), or None if the expected literal
    subject/verb/aux text isn't found."""
    new_subj = SUBJECT_WORD[to_person]
    result = _splice(sentence, matrix_subj, new_subj)
    if result is None:
        return None
    if matrix_type in QUESTION_TYPES:
        result = _splice_aux(result, do_form(from_person, tense), do_form(to_person, tense))
    else:
        result = _splice(result, conjugate(matrix_verb, from_person, tense), conjugate(matrix_verb, to_person, tense))
    if result is None:
        return None
    return result, new_subj


def change_tense(sentence, matrix_subj, matrix_verb, person, from_tense, to_tense, matrix_type):
    """Swap matrix verb tense and the embedded copula (is/was). Returns None
    if the matrix verb's conjugated form for `from_tense` isn't found.
    Wh/polar questions take a separate path that swaps the do-support
    auxiliary and the embedded copula ('Does John know where the keys are?'
    -> 'Did John know where the keys were?') and never touches the bare verb:
    for first/second person the present-tense conjugated form and the bare
    form are the same string ("I think"), so matching on the verb would
    produce 'Do I thought...'."""
    if matrix_type in QUESTION_TYPES:
        result = _splice_aux(sentence, do_form(person, from_tense), do_form(person, to_tense))
        return None if result is None else _swap_embedded_copula(result, from_tense, to_tense)
    if matrix_type != "declarative":
        return None
    old_verb_form = conjugate(matrix_verb, person, from_tense)
    new_verb_form = conjugate(matrix_verb, person, to_tense)
    result = _splice(sentence, old_verb_form, new_verb_form)
    if result is None:
        return None
    old_cop, new_cop = COPULA[from_tense], COPULA[to_tense]
    spliced = _splice(result, old_cop, new_cop)
    if spliced is None and from_tense == "present" and to_tense == "past":
        # the plain " is "/" was " splice misses a contracted copula
        # ("It's raining" has no space before "'s"), which would otherwise
        # leave the embedded clause's tense stuck in present ("It's
        # raining, I thought.") while the matrix verb moves to past
        spliced = _splice(result, "It's", "It was") or _splice(result, "it's", "it was")
    return spliced if spliced is not None else result


def negate_clause(sentence, matrix_subj, matrix_verb, person, tense, matrix_type):
    """Insert do-support negation at the matrix verb: 'John thinks' -> 'John
    doesn't think'. Only safe for a plain declarative matrix clause - a
    matrix_type that's already a question (do-support already fronted) is
    refused rather than double-inserting an auxiliary."""
    if matrix_type != "declarative":
        return None
    old_verb_form = conjugate(matrix_verb, person, tense)
    aux = do_form(person, tense)
    bare = VERB_FORMS[matrix_verb]["bare"]
    return _splice(sentence, old_verb_form, f"{aux}n't {bare}")


def make_polar_question(sentence, matrix_subj, matrix_verb, person, tense):
    """Front do-support: 'John thinks that X.' -> 'Does John think that X?'.
    Only valid (and only attempted) when matrix_subj + the conjugated verb
    are the sentence's first two words, since that's a structural
    requirement of subject-aux fronting, not a phenomenon-specific choice."""
    if not sentence.startswith(matrix_subj):
        return None
    old_verb_form = conjugate(matrix_verb, person, tense)
    rest = sentence[len(matrix_subj):].lstrip()
    if not rest.startswith(old_verb_form):
        return None
    rest = rest[len(old_verb_form):].lstrip().rstrip(".")
    aux = do_form(person, tense).capitalize()
    bare = VERB_FORMS[matrix_verb]["bare"]
    return f"{aux} {matrix_subj} {bare} {rest}?"


def repronoun_subject(sentence, matrix_subj, pronoun, matrix_verb, tense, matrix_type, negation):
    """Swap matrix_subj for a pronoun, reconjugating the verb/aux if needed.
    This method is really only used to help with the conjugate for reconjugating "they"
    "he"/"she"/"it" stay 3sg-agreeing like any name, so a plain swap is safe.
    "they" takes non-3sg agreement instead ("they think", "they don't know",
    "do they know") even when referring to one person, handled for plain declarative, 
    negated declarative (do-support "doesn't"/"don't"), and 
    polar_question (fronted "does"/"do"). Other matrix_types (e.g. wh_question) 
    aren't modeled here and are skipped (None) rather than guessed - 
    see find_pronoun_candidate in generate_variants.py, which only offers "they" where this will work.
    TODO: support wh questions """
    result = _splice(sentence, matrix_subj, pronoun)
    if result is None or pronoun != "they":
        return result
    old_person, new_person = "third_person", "they"
    if matrix_type == "declarative" and negation == "yes":
        bare = VERB_FORMS[matrix_verb]["bare"]
        old_form, new_form = f"{do_form(old_person, tense)}n't {bare}", f"{do_form(new_person, tense)}n't {bare}"
    elif matrix_type == "polar_question":
        # the fronted aux is sentence-initial and thus capitalized ("Does"),
        # unlike the lowercase do-support forms elsewhere
        old_form, new_form = do_form(old_person, tense).capitalize(), do_form(new_person, tense).capitalize()
    elif matrix_type == "declarative":
        old_form, new_form = conjugate(matrix_verb, old_person, tense), conjugate(matrix_verb, new_person, tense)
    else:
        return None
    return _splice(result, old_form, new_form)


def toggle_complementizer(sentence, complementizer):
    """Remove an optional declarative 'that': 'John thinks that X' -> 'John
    thinks X'. 'whether' is obligatory for interrogative complements and is
    never touched; 'none' rows have nothing to add back, so this only ever
    removes. Returns None if not applicable."""
    if complementizer != "that":
        return None
    result = _splice(sentence, " that ", " ")
    return result


PREPOSITION_VERBS = ("think", "know")  # matches vocabulary.csv's preposition rows' `requires` column 
PREPOSITIONS = ("about", "of")

def toggle_preposition(sentence, matrix_verb, person, tense, preposition):
    """Swap 'about' <-> 'of' immediately after the matrix verb ('John thinks
    about the rumor' <-> 'John thinks of the rumor') - only meaningful for
    the PP-complement side of an NP-vs-PP comparison; the NP side has no
    preposition at all (always 'NA'), so this returns None for it rather
    than guessing. Only applies for think/know (see PREPOSITION_VERBS) -
    e.g. a control row's "start" doesn't take "of" naturally. Returns
    (new_sentence, new_preposition), or None."""
    if preposition not in PREPOSITIONS or matrix_verb not in PREPOSITION_VERBS:
        return None
    new_prep = "of" if preposition == "about" else "about"
    verb_form = conjugate(matrix_verb, person, tense)
    result = _splice(sentence, f"{verb_form} {preposition}", f"{verb_form} {new_prep}")
    return (result, new_prep) if result is not None else None

# CHILDES-style transcripts are all lowercase with punctuation split off as its own
# token ("john thinks that it is raining ."). Apostrophes inside contractions stay put.
PUNCTUATION = ".,?!;"
VOCABULARY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_data", "vocabulary.csv")


@functools.lru_cache(maxsize=None)
def proper_nouns(vocabulary_path=VOCABULARY_PATH):
    """Capitalized `person` entries in vocabulary.csv (John, Mary, ...), whose
    capitalization lowercasing loses."""
    with open(vocabulary_path, newline="", encoding="utf-8") as f:
        return frozenset(
            row["word"] for row in csv.DictReader(f)
            if row["slot"] == "person" and row["word"][:1].isupper()
        )


def normative_to_childes_formatting(sentence):
    """'John thinks that it's raining.' -> 'john thinks that it's raining .'
    Dialogue speaker labels stay uppercase: 'A: Is it raining?' -> 'A: is it raining ?'"""
    sentence = re.sub(rf"\s*([{re.escape(PUNCTUATION)}])", r" \1", sentence.lower())
    sentence = re.sub(r"\b([ab]):", lambda m: m.group(1).upper() + ":", sentence)
    return re.sub(r"\s+", " ", sentence).strip()


def childes_to_normative_formatting(sentence):
    """Best-effort inverse of normative_to_childes_formatting: reattach punctuation,
    capitalize sentence starts (and after dialogue speaker labels like 'A:'), 'I', and
    vocabulary proper nouns. Any other capitalization in the original can't be recovered."""
    sentence = re.sub(rf"\s+([{re.escape(PUNCTUATION)}])", r"\1", sentence.strip())
    sentence = re.sub(r"\bi\b", "I", sentence)
    for name in proper_nouns():
        sentence = re.sub(rf"\b{name.lower()}\b", name, sentence)
    return re.sub(r"(^|[.?!:]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), sentence)