import pandas as pd

from benchmark.scripts import generate_variants as gv

VOCAB = pd.DataFrame({
    "word": ["start", "investigate", "like", "rumor", "secret", "where", "when"],
    "slot": ["control_verb", "control_verb", "control_verb", "content_noun", "content_noun", "complementizer", "complementizer"],
    "investigable": ["", "", "", "1", "1", "", ""],
    "requires": ["tense:past", "investigable:1", "tense:present", "", "matrix_verb:think,know,investigate,like", "", ""],
    "notes": [""] * 7,
})


def _row(sentence, **kw):
    return {"condition": "control", "tense": "present", "matrix_type": "declarative", "complementizer": "none",
            "matrix_subj_category": "third_person", "matrix_verb": "investigate", "sentence": sentence, **kw}


def _vocab(rows):
    return gv.Vocabulary(VOCAB, pd.DataFrame(rows))


def test_fit_control_verbs():
    """Past takes start unless the noun can't be started (secret); present takes
    investigate for an investigable noun, else like."""
    rows = [_row("John investigated the rumor.", tense="past"), _row("John investigated the secret.", tense="past"),
            _row("John investigates the rumor."), _row("John investigates the song.")]
    fitted = gv.fit_control_verbs(rows, _vocab(rows))
    assert [r["sentence"] for r in fitted] == [
        "John started the rumor.", "John investigated the secret.", "John investigates the rumor.", "John likes the song."]


def test_matching_exclusion_counts_fronted_wh_word():
    """A fronted 'When' is the row's complementizer, so the whole set is excluded."""
    rows = [_row("When does John know Mary hid the keys?", matrix_type="wh_question", matrix_verb="know"),
            _row("Does John know when Mary hid the keys?", matrix_type="polar_question", matrix_verb="know", complementizer="when")]
    exclusion = {"match": {"complementizer": "when"}, "reason": "matrix attachment"}
    vocab = _vocab(rows)
    assert gv.matching_exclusion(rows[:1], [exclusion], vocab) is exclusion
    assert gv.matching_exclusion([_row("Where does John know Mary hid the keys?", matrix_type="wh_question")], [exclusion], vocab) is None
