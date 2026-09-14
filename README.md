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
`--n-variants` to specify how many lexically-varied copies of each assertion set. Each copy stacks a random 1-3 substitutions drawn from whatever slots actually match text in that set - person/pronoun swaps (with reconjugation when needed), weather predicate, rumor content, judgment adjectives, objects, and matched mishap object/verb pairs. See `vocabulary.csv` for the full word lists and `generate_variants.py`'s
`gather_candidates` for how they're combined.
