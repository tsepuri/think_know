# think and know

## setup

```bash
conda create -n think_know python=3.12
conda activate think_know
pip install -r requirements.txt
```

## usage

```bash
# expand config.json + sentences.csv into pairs.csv
python create_pairs.py
```

## generated files

- **`pairs.csv`** — one row per pair (`sentence1`, `sentence2`), which
  `sentence1` is predicted to outscore. A `direct` assertion is a single row
  (`pair_role = comparison`); a `difference` assertion is two rows sharing an
  `assertion_id` (`pair_role = minuend` , `subtrahend`) — checking if the
  minuend row's logprob margin is greater than the subtrahend row's. TODO: use rank number instead of `pair_role`

## lexical variants

every base assertion is optionally duplicated with a word substituted throughout
(skipped when the substitution has no effect on that assertion):

| variant | substitution |
|---|---|
| `rumor_message` | rumor -> message |
| `rain_snow` | raining -> snowing ; rain -> snow ; rained -> snowed|
| `name_swap_proper` | Mary -> Christopher, John -> Theresa |
| `name_swap_role` | Mary -> the student, John -> the teacher |
