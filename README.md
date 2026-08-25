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

# score pairs.csv with GPT-2 and write results/summary
python eval.py
```

`create_pairs.py` flags:
- `--phenomena`, `--categories`, `--subtypes`: restrict to matching conditions (default: all)
- `--variants`: which lexical variants to include, e.g. `--variants base rain_snow`
  (default: all — see "lexical variants" below)
- `--config`, `--sentences`, `--output`: override the default file paths

`eval.py` flags:
- `--pairs`: input CSV from create_pairs.py (default `pairs.csv`)
- `--model`: any causal LM supported by `transformers` (default `gpt2`); not sure if this works
- `--details-output`, `--summary-output`: override the default output paths
- `--device`: default `cuda` if available, else `cpu`

## data files

- **`config.json`** — the phenomena/conditions/predictions being tested. this file is still a work in progress. each
  condition lists up to 6 sentence IDs with different predictions per `prediction_type`
  (e.g. `verb_contrast`, `type_contrast`, `control`)::
  - `comparison_type: "direct"`: a prediction `[a, b]` means `logP(a) > logP(b)`.
  - `comparison_type: "difference"`: a prediction `[a, b, c, d]` means
    `logP(a) - logP(b) > logP(c) - logP(d)`.
  see the `schema` block at the top of the file for the full key with all the metadata included.
- **`sentences.csv`** — sentence lookup table with two columns: `sentence_id,sentence` and is referenced by `config.json`. maybe need to add grammaticality, verb type, and other metadata here.

## generated files

- **`pairs.csv`** — one row per pair (`sentence1_text`, `sentence2_text`), which
  `sentence1_text` is predicted to outscore. A `direct` assertion is a single row
  (`pair_role = comparison`); a `difference` assertion is two rows sharing an
  `assertion_id` (`pair_role = minuend` , `subtrahend`) — checking if the
  minuend row's logprob margin is greater than the subtrahend row's.
- **`eval_results.csv`** — one row per assertion, with the LM's `margin` and
  binary `correct` (1 if the assertion held).
- **`eval_summary.json`** — aggregate accuracy, broken down `by_phenomenon` and
  `by_phenomenon_subtype_matrix_type`. can be adjusted to include other breakdowns. breakdown reports `n`/`accuracy` for both critical and control conditions.

## lexical variants

every base assertion is optionally duplicated with a word substituted throughout
(skipped when the substitution has no effect on that assertion):

| variant | substitution |
|---|---|
| `rumor_message` | rumor -> message |
| `rain_snow` | raining -> snowing ; rain -> snow ; rained -> snowed|
| `name_swap_proper` | Mary -> Christopher, John -> Theresa |
| `name_swap_role` | Mary -> the student, John -> the teacher |
