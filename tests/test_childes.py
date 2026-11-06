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
