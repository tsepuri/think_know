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
