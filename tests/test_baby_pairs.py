import pandas as pd

from benchmark.scripts import make_baby_pairs as mb


def test_rare_entry_drops_whole_assertion():
    """A rare entry in only the rank-2 row of a differences assertion flags the whole assertion;
    matches are whole-word, so 'rumor' doesn't match inside 'rumors'."""
    pairs = pd.DataFrame({
        "assertion_id": [0, 0, 1],
        "sentence1": ["john knows that it's raining .", "john knows the rumor .", "john knows the rumors ."],
        "sentence2": ["john thinks that it's raining .", "john thinks the rumor .", "john thinks the rumors ."],
    })
    assert mb.rare_entries(pairs, ["rumor"]) == {0: ["rumor"]}
