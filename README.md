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
python generate_variants.py

# expand config.json + sentences_with_variants.csv into benchmark_data/pairs.csv
python create_pairs.py

# score pairs.csv with gpt2, Llama-3.1-8B, Llama-3.1-8B-Instruct, and Mistral 8B (the
# defaults) and compare all three
python eval.py

# score with specific model(s) instead; repeat --model to compare several
python eval.py --model gpt2
python eval.py --model meta-llama/Llama-3.1-8B --model meta-llama/Llama-3.1-8B-Instruct

# Llama weights are gated on HF - log in first or add HF_TOKEN; weights land in
# huggingface_hub's usual cache unless you point --cache-dir elsewhere
huggingface-cli login
# OR
HF_TOKEN=xxxx
# THEN
python eval.py --cache-dir /path/to/scratch/hf_cache
```

Running more than one `--model` writes separate `pairs_scored-<model>.csv` /
`eval_results-<model>.csv` / `eval_summary-<model>.json` files per model (instead
of the single `pairs_scored.csv` / `eval_results.csv` / `eval_summary.json`
produced for one model), plus an `eval_summary-comparison.json` with each
model's overall accuracy side by side. `pairs_scored.csv` is `pairs.csv` with
each sentence's logprob and the pair's margin attached, one row per pair, for
inspecting individual sentence scores rather than only assertion-level results.

### SLOR scoring

`eval_slor_manual.py` scores the same pairs with SLOR (model logprob minus a unigram model's
logprob, per token) instead of mean logprob, so token frequency alone can't decide a
pair. The unigram model is estimated per model with its own tokenizer; pass a corpus
close to what that model was trained on (`hf:` corpora are streamed and only a ~1 MB
`.npy` of counts is cached but this can be changed):

```
python eval_slor_manual.py \
  --model gpt2=hf:Skylion007/openwebtext
```

Outputs are `*-<model>-slor.*` in `eval_output/`, so they sit beside (and never
overwrite) the plain-logprob results, and `plot_eval.py` shows them as `<model>-slor`.

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
content, judgment adjectives, objects, hiding verbs, mishap object/verb pairs,
determiners (a determiner's `number` must match its noun's: `a` needs a singular noun, `all of the`
a plural one, blank takes either), and the wh-complementizer (where/when/why). Because the clause goes first, those
slots vary the clause that's actually there, including a swapped-in one. Set the
options in `benchmark_data/variant_config.json` (`--variant-config` to use another),
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
