import pytest
import spacy

from scripts import childes as ch


@pytest.fixture(scope="module")
def nlp():
    return spacy.load("en_core_web_md", exclude=["ner"])


@pytest.mark.slow
@pytest.mark.parametrize("text, category", [
    ("I don't know if it's raining .", "interrogative"),
    ("I'm trying to think whether we have a clothesline .", "interrogative"),
    ("I can't think what it's called .", "interrogative"),
    ("I think if you push it it will go .", "declarative"),
    ("I think when we're done we'll eat some biscuits .", "declarative"),
    ("I think what we should do is play outside .", "declarative"),
    ("I think so .", "so"),
    ("what do you think ?", "matrix_wh"),
    ("I don't know .", "none"),
])
def test_classify_complement(nlp, text, category):
    """Known answers, including declaratives whose complement opens with an adjunct or pseudo-cleft."""
    verbs, _ = ch.classify_texts([text], nlp)
    assert verbs.category.iloc[0] == category


def _gra(rows):
    import pandas as pd
    return pd.DataFrame(rows, columns=["utterance_id", "token_order", "gloss", "stem", "part_of_speech", "gra_index", "gra_head", "gra_relation"])


@pytest.mark.parametrize("rows, category, detail, negated", [
    # real %gra parse: i don't know whether it would work though
    ([(1, 1, "i", "i", "pron", 1, 4, "NSUBJ"), (1, 2, "don't", "do", "aux", 2, 4, "AUX"), (1, 3, "know", "know", "verb", 4, 0, "ROOT"),
      (1, 4, "whether", "whether", "sconj", 5, 8, "MARK"), (1, 5, "it", "it", "pron", 6, 8, "NSUBJ"), (1, 6, "would", "would", "aux", 7, 8, "AUX"),
      (1, 7, "work", "work", "verb", 8, 4, "CCOMP"), (1, 8, "though", "though", "adv", 9, 8, "ADVMOD")], "interrogative", "whether", True),
    # it's heads on raining across the clitic's index (4, no token row)
    ([(2, 1, "i", "i", "pron", 1, 2, "NSUBJ"), (2, 2, "think", "think", "verb", 2, 0, "ROOT"),
      (2, 3, "it's", "it", "pron", 3, 5, "NSUBJ"), (2, 4, "raining", "rain", "verb", 5, 2, "CCOMP")], "declarative", "zero", False),
    ([(3, 1, "i", "i", "pron", 1, 2, "NSUBJ"), (3, 2, "know", "know", "verb", 2, 0, "ROOT"),
      (3, 3, "about", "about", "adp", 3, 4, "CASE"), (3, 4, "it", "it", "pron", 4, 2, "OBL")], "PP", "about", False),
    ([(4, 1, "i", "i", "pron", 1, 2, "NSUBJ"), (4, 2, "think", "think", "verb", 2, 0, "ROOT"), (4, 3, "so", "so", "adv", 3, 2, "ADVMOD")], "so", "so", False),
])
def test_classify_gra(rows, category, detail, negated):
    """Known answers on %gra parses: wh/whether CCOMP, zero-that CCOMP, obl+case PP, so; negation on a don't aux."""
    verbs = ch.classify_gra(_gra(rows))
    assert (verbs.category.iloc[0], verbs.detail.iloc[0], verbs.negated.iloc[0]) == (category, detail, negated)


def _parse(uid, words):
    """%gra token rows from 'gloss/index>head:RELATION' words; the first word containing a TARGET_FORMS stem is tagged verb."""
    rows, tagged = [], False
    for n, w in enumerate(words.split(), 1):
        gloss, rest = w.split("/")
        index, rest = rest.split(">")
        head, relation = rest.split(":")
        is_verb = not tagged and gloss in {f for fs in ch.TARGET_FORMS.values() for f in fs}
        tagged |= is_verb
        stem = next((s for s, fs in ch.TARGET_FORMS.items() if gloss in fs), gloss) if is_verb else gloss
        rows.append((uid, n, gloss, stem, "verb" if is_verb else "x", int(index), int(head), relation))
    return _gra(rows)


@pytest.mark.parametrize("words, category, detail, polar", [
    # a wh-word deep inside a declarative complement doesn't make it interrogative
    ("i/1>2:NSUBJ think/2>0:ROOT you/3>4:NSUBJ know/4>2:CCOMP what/5>4:OBJ this/6>7:NSUBJ is/7>5:ACL-RELCL", "declarative", "zero", False),
    # nor does a relative pronoun inside the object
    ("i/1>2:NSUBJ know/2>0:ROOT a/3>4:DET lady/4>2:OBJ who/5>6:NSUBJ swallowed/6>4:ACL-RELCL", "NP", "noun", False),
    ("do/1>3:AUX you/2>3:NSUBJ know/3>0:ROOT what/4>5:DET color/5>3:OBJ", "interrogative", "wh_noun_phrase", True),
    # the 's clitic (index 5) has no row: its dependents still reach think as a complement
    ("do/1>3:AUX you/2>3:NSUBJ think/3>0:ROOT there's/4>5:EXPL something/6>5:NSUBJ", "declarative", "zero", True),
    # what's keeps its clitic in %gra
    ("i/1>4:NSUBJ don't/2>4:AUX know/4>0:ROOT what's/5>7:NSUBJ next/7>4:CCOMP", "interrogative", "wh_clause", False),
    # a sluiced wh-adverb
    ("i/1>4:NSUBJ don't/2>4:AUX know/4>0:ROOT why/5>4:ADVMOD", "interrogative", "wh_bare", False),
    # an if-clause %gra calls advcl is a conditional, not an embedded question
    ("i/1>2:NSUBJ think/2>0:ROOT if/3>5:MARK you/4>5:NSUBJ turn/5>2:ADVCL it/6>5:OBJ", "none", "root", False),
    ("what/1>6:OBJ do/2>4:AUX you/3>4:NSUBJ think/4>0:ROOT that/5>6:NSUBJ is/6>4:CCOMP", "matrix_wh", "what", False),
])
def test_classify_gra_heuristics(words, category, detail, polar):
    """Known answers on (real) %gra parses for the opener, clitic, sluice, conditional and polar rules."""
    verbs = ch.classify_gra(_parse(9, words))
    assert (verbs.category.iloc[0], verbs.detail.iloc[0], verbs.polar.iloc[0]) == (category, detail, polar)


def test_classify_gra_skips_unparsed():
    """An utterance with a token missing its %gra annotation yields no verb rows."""
    rows = [(5, 1, "i", "i", "pron", 1, 2, "NSUBJ"), (5, 2, "know", "know", "verb", None, None, None)]
    assert ch.classify_gra(_gra(rows)).empty


def test_augmentation_crossing():
    """At the closed-form crossing, augmented think's ratio equals know's observed ratio."""
    import pandas as pd
    table = pd.DataFrame(
        {"declarative": [100, 900], "interrogative": [300, 10], "total": [500, 1000]}, index=["know", "think"]
    )
    m = ch.augmentation_crossing(table)
    ratio = ch.augmentation_sweep(table, [m]).think_decl_per_interrog.iloc[0]
    assert ratio == pytest.approx(100 / 300)
    assert m == pytest.approx((900 * 3 - 10) / 300)
