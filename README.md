# think and know

## setup

```bash
conda create -n think_know python=3.12
conda activate think_know
pip install -r requirements.txt
```

## usage

```bash
# expand sentences.csv into sentences_with_variants.csv (tense/person/negation/
# matrix_type/complementizer/preposition siblings, plus 2 lexically-varied
# copies per assertion set)
python -m benchmark.scripts.generate_variants

# expand config.json + sentences_with_variants.csv into benchmark/data/pairs.csv
python -m benchmark.scripts.create_pairs

# score pairs.csv with gpt2, Llama-3.1-8B, Llama-3.1-8B-Instruct, and Mistral 8B (the
# defaults) and compare all three
python -m scripts.eval

# score with specific model(s) instead; repeat --model to compare several
python -m scripts.eval --model gpt2
python -m scripts.eval --model meta-llama/Llama-3.1-8B --model meta-llama/Llama-3.1-8B-Instruct

# Llama weights are gated on HF - log in first or add HF_TOKEN; weights land in
# huggingface_hub's usual cache unless you point --cache-dir elsewhere
huggingface-cli login
# OR
HF_TOKEN=xxxx
# THEN
python -m scripts.eval --cache-dir /path/to/scratch/hf_cache
```

Each model gets its own folder, `eval_output/<model>/` (`--output-dir` to change the
parent), holding `pairs_scored.csv`, `eval_results.csv` and `eval_summary.json`.
`pairs_scored.csv` is `pairs.csv` with each sentence's logprob and the pair's margin
attached, one row per pair, for inspecting individual sentence scores rather than only
assertion-level results.

File names get a prefix for the run's inputs, so runs on the same model never collide:
`childes_formatted_` for `--childes-format`, then the pairs file's tag (`--pairs
benchmark/data/pairs_pairwise.csv` gives `pairwise_`):

```
eval_output/
  gpt2/
    eval_results.csv  pairs_scored.csv  eval_summary.json
    pairwise_eval_results.csv  ...             # --pairs benchmark/data/pairs_pairwise.csv
    childes_formatted_eval_results.csv  ...    # --childes-format
  gpt2-slor/  ...                              # eval_slor_manual.py
  unigram/                                     # cached unigram counts for SLOR
```

### SLOR scoring

`eval_slor_manual.py` scores the same pairs with SLOR (model logprob minus a unigram model's
logprob, per token) instead of mean logprob, so token frequency alone can't decide a
pair. The unigram model is estimated per model with its own tokenizer; pass a corpus
close to what that model was trained on (`hf:` corpora are streamed and only a ~1 MB
`.npy` of counts is cached but this can be changed):

```
python -m scripts.eval_slor_manual \
  --model gpt2=hf:Skylion007/openwebtext
```

Outputs go in `eval_output/<model>-slor/`, so they never overwrite the plain-logprob
results, and `plot_eval.py` shows them as `<model>-slor`. Unigram counts are cached in
`eval_output/unigram/`.

## layout

Run everything from the repo root as modules (`python -m scripts.eval`), so `utils.py`
and `scripts/` import.

- `benchmark/data/`: benchmark inputs and generated pairs; `benchmark/scripts/`: their generators
- `scripts/`: scoring, plotting, and `childes.py` (behind the CHILDES notebook); `figures/`: plots from `plot_eval.py`
- `notebooks/`: marimo notebooks
- `cache/` (gitignored): downloaded CHILDES tables, parses, and notebook figures

## CHILDES think/know analysis

`notebooks/childes_think_know.py` pulls every Eng-NA and Eng-UK utterance from
childes-db 2026.1 on Redivis, checks benchmark vocabulary frequency (threshold 10)
with same-tag alternatives for rare words, and classifies think/know complements
with spaCy (declarative vs. interrogative ratios, categorical check, ablation sweep).

```bash
python -m spacy download en_core_web_md   # parsing
python -m spacy download en_core_web_lg   # word vectors for alternatives
marimo edit notebooks/childes_think_know.py   # first run opens a Redivis login in the browser
```

## generated files

- **`sentences_with_variants.csv`** — `sentences.csv` expanded by
  `generate_variants.py`: every base row gets tense/negation/person/matrix_type
  siblings per its condition's `manipulated_types` in `config.json`, plus a
  complementizer- or preposition-toggled copy where applicable, plus 2
  lexically-varied copies (see below). Each row's `unique_id` identifies which
  assertion-forming set it belongs to (e.g. `cpq_pres_aff_3p_decl_that_base`).
- **`pairs.csv`** — one row per pair (`sentence1`, `sentence2`), where
  `sentence1` is predicted to outscore `sentence2`, grouped into assertions by
  `assertion_id`. Each row's `rank` orders it by expected contrast size: a
  `direct` assertion is a single `rank = 1` row; a `differences` assertion is
  two or more rows (`rank = 1`, `rank = 2`, ..) sharing an `assertion_id`, checking that rank 1's logprob margin exceeds rank 2's which exceeds the next one etc. Currently we only have pair of sentences so rank 1 and rank 2.

## lexical variation

`generate_variants.py` draws on `vocabulary.csv` (word pools with number/
gender/animacy/liquid/breakable/etc. features, so substitutions respect agreement
and collocation) to build 2 lexically-varied copies of each assertion set, on
top of the unchanged base copy. We can use
`--n-variants` to specify how many lexically-varied copies of each assertion set.

Each copy is built in two steps. For conditions with `free_embedded_clause`, it
first gets a different `embedded_clause` - or, in a mini-discourse, a different `wh_question` -
(with probability `embedded_clause_swap_prob`). Then a random `min_slots_modified` to
`max_slots_modified` of the other slots that match text in the set are modified:
person/pronoun swaps (with reconjugation when needed), weather predicate, rumor
content, judgment adjectives, objects, hiding verbs, mishap object/verb pairs, the
control rows' matrix verb (start/investigate, reinflected),
determiners (a determiner's `number` must match its noun's: `a` needs a singular noun, `all of the`
a plural one, blank takes either), and the wh-complementizer (where/when/why). Because the clause goes first, those
slots vary the clause that's actually there, including a swapped-in one. Set the
options in `benchmark/data/variant_config.json` (`--variant-config` to use another),
for every condition or overridden per condition `code` under `"conditions"`:

```json
{
  "embedded_clause_swap_prob": 0.5,
  "min_slots_modified": 1,
  "max_slots_modified": 2,
  "conditions": {"cpq": {"max_slots_modified": 3}}
}
```

### `requires`

A vocabulary entry's `requires` is `<column>:<value>[,<value>...]` (`;` between
several requirements, all must hold) and limits where it can be swapped in.
`<column>` is either

- a `sentences.csv` column - the row being varied must have one of the values:
  `matrix_verb:think,know` (message), `paradigm:mini-discourse` (wh_question),
  `complementizer:where,when,why` (an embedded_clause), or
- a `vocabulary.csv` feature column - a word with one of the values must be in
  the sentence: `breakable:1` (a mishap_verb needs a breakable object present).
  For a determiner it's checked against its own noun: `countable:1`.

A substitution is never allowed to newly break a `requires` of any word in the
set, so swapping `vase` for `coffee` is refused under `broke`, and `message` is
refused wherever the set has `start` rows. Requirements that are already unmet in
the base rows (control rows like `John starts about the rumor`) are left alone.
Malformed `requires`, or one naming an unknown column, stops the script; a value
that occurs in no matching `sentences.csv` column is printed as a warning.

The complementizer is only varied for an embedded_clause that lists several in
its `requires`, and is changed together with every sentence it appears in,
fronted ("Where does John know Mary hid the keys?") or embedded ("Does John know
where Mary hid the keys?"). A clause that lists no complementizers is only used
with that/whether/none, never a wh-word. See `vocabulary.csv` for the full word
lists and `generate_variants.py`'s `gather_candidates` for how they're combined.
