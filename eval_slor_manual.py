"""score pairs.csv like eval.py, but with SLOR instead of mean per-token logprob.

SLOR (Pauls & Klein 2012; Lau et al. 2017) = (log p_model(text) - log p_unigram(text)) / n_tokens,
so a sentence isn't rewarded just for being made of frequent tokens (e.g. "that" vs
"whether"). As in Misra & Mahowald's aannalysis setup, the unigram model is estimated
with the same tokenizer as the LM and, ideally, on the corpus that LM was trained on.

Pretrained models (gpt2, Llama, Mistral) don't ship their training data, so pass the
closest open proxy per model as `--model NAME=CORPUS`, e.g.

    --model gpt2=hf:Skylion007/openwebtext              (open replica of gpt2's WebText)
    --model meta-llama/Llama-3.1-8B=hf:HuggingFaceFW/fineweb   (generic web-text proxy)
    --model gpt2=hf:wikitext@wikitext-103-raw-v1        (@CONFIG picks a dataset config)
    --model gpt2=corpus.txt                             (plain text, one document/line per row)

`hf:` corpora are streamed (first --unigram-docs documents, never saved to disk) and only
the resulting token log-probs are cached, as a small .npy in --unigram-cache-dir, so each
tokenizer/corpus pair is counted once. `--unigram-corpus` is the fallback for any --model
given without its own `=CORPUS`.

everything downstream of the sentence score (assertions, summaries, file layout) is reused
from eval.py. Outputs get a `-slor` suffix (eval_results-<model>-slor.csv, ...) so they
never overwrite eval.py's, and plot_eval.py picks them up as separate "<model>-slor" entries.
"""

import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd
import torch

from eval import (
    apply_childes_format, default_device, insert_suffix, load_model, load_pairs, round_floats,
    score_assertions, score_pairs, slugify_model_name, summarize_with_controls, token_logprobs,
)


def corpus_lines(corpus, text_field, max_docs):
    """Yield the corpus's documents: streamed from the HF hub for "hf:NAME[@CONFIG]"
    (nothing is written to disk), else read from a plain-text file, one document per line."""
    if corpus.startswith("hf:"):
        from datasets import load_dataset  # only needed for hf: corpora

        name, _, config = corpus[len("hf:"):].partition("@")
        dataset = load_dataset(name, config or None, split="train", streaming=True)
        for example in itertools.islice(dataset, max_docs):
            yield example[text_field]
    else:
        with open(corpus, encoding="utf-8") as f:
            yield from f


def build_unigram_logprobs(lines, tokenizer, vocab_size, batch_size=1_000):
    """Log-probability of each token id under a unigram model estimated from `lines`
    (documents) tokenized with `tokenizer`. Add-one smoothing so tokens the corpus never
    contains still get a finite log-prob."""
    counts = np.ones(vocab_size)
    lines = (line.strip() for line in lines)
    lines = (line for line in lines if line)
    while batch := list(itertools.islice(lines, batch_size)):
        ids = tokenizer(batch, add_special_tokens=False).input_ids
        ids = np.fromiter(itertools.chain.from_iterable(ids), dtype=np.int64)
        counts += np.bincount(ids, minlength=vocab_size)
    return torch.from_numpy(np.log(counts / counts.sum()))


def load_unigram_logprobs(model_name, corpus, tokenizer, vocab_size, args):
    """build_unigram_logprobs for a model, cached as a small .npy per (model, corpus, doc
    count) so a corpus is only streamed and tokenized once. Plain-text files aren't cached
    (they're local and may change)."""
    cache_path = None
    if corpus.startswith("hf:"):
        key = "-".join(slugify_model_name(part) for part in (model_name, corpus, str(args.unigram_docs)))
        cache_path = os.path.join(args.unigram_cache_dir, f"unigram-{key}.npy")
        if os.path.exists(cache_path):
            print(f"Loading cached unigram model from {cache_path}")
            return torch.from_numpy(np.load(cache_path))

    lines = corpus_lines(corpus, args.text_field, args.unigram_docs)
    unigram_logprobs = build_unigram_logprobs(lines, tokenizer, vocab_size)
    if cache_path:
        os.makedirs(args.unigram_cache_dir, exist_ok=True)
        np.save(cache_path, unigram_logprobs.numpy())
        print(f"Cached unigram model to {cache_path}")
    return unigram_logprobs


def sentence_slor(text, model, tokenizer, device, unigram_logprobs):
    """(log p_model - log p_unigram) / n_tokens for the whole sentence.

    Both terms are summed over the same tokens - every token of the sentence, conditioned
    on BOS (see eval.py's token_logprobs) - matching how eval.py's mean logprob is computed.
    """
    targets, logprobs = token_logprobs(text, model, tokenizer, device)
    unigram_logprob = unigram_logprobs[targets.cpu()].sum().item()
    return (logprobs.sum().item() - unigram_logprob) / len(targets)


def build_slor_cache(pairs_df, model, tokenizer, device, unigram_logprobs):
    """SLOR for every distinct sentence text, scored once (same reasoning as eval.py's
    build_logprob_cache)."""
    texts = pd.unique(pd.concat([pairs_df["sentence1"], pairs_df["sentence2"]]))
    return {
        text: sentence_slor(text, model, tokenizer, device, unigram_logprobs)
        for text in texts
    }


def run_slor_for_model(model_name, corpus, pairs_df, args):
    """Score pairs_df with one model's SLOR and write its details/summary outputs. Returns
    the summary (used to build the cross-model comparison)."""
    print(f"\n=== {model_name} (SLOR, unigram corpus: {corpus}) ===")
    print(f"Loading {model_name} on {args.device} (cache: {args.cache_dir or 'huggingface_hub default'})...")
    model, tokenizer = load_model(model_name, args.device, args.cache_dir)

    print("Estimating unigram model...")
    unigram_logprobs = load_unigram_logprobs(
        model_name, corpus, tokenizer, max(model.config.vocab_size, len(tokenizer)), args
    )

    cache = build_slor_cache(pairs_df, model, tokenizer, args.device, unigram_logprobs)
    # score_pairs (from eval.py) names its columns *_logprob; these hold SLOR
    scored_pairs = score_pairs(pairs_df, cache).rename(columns={
        "sentence1_logprob": "sentence1_slor", "sentence2_logprob": "sentence2_slor",
    })
    results = score_assertions(scored_pairs)

    del model
    if args.device == "cuda":
        torch.cuda.empty_cache()

    slug = f"{slugify_model_name(model_name)}-slor"
    pairs_output = insert_suffix(args.pairs_output, slug)
    details_output = insert_suffix(args.details_output, slug)
    summary_output = insert_suffix(args.summary_output, slug)

    scored_pairs.to_csv(pairs_output, index=False)
    print(f"Wrote {len(scored_pairs)} scored pairs to {pairs_output}")

    results.to_csv(details_output, index=False)
    print(f"Wrote {len(results)} scored assertions to {details_output}")

    overall = summarize_with_controls(results, []).iloc[0]
    by_phenomenon = summarize_with_controls(results, ["phenomenon"])
    by_phenomenon_subtype_matrix_type = summarize_with_controls(results, ["phenomenon", "subtype", "matrix_type"])

    print(f"Overall accuracy: {overall['accuracy']:.3f} ({int(overall['n'])} test assertions)")
    print(f"Overall control accuracy: {overall['control_accuracy']:.3f} ({int(overall['control_n'])} control assertions)")
    summary = {
        "model": model_name,
        "score": "slor",
        "unigram_corpus": corpus,
        "n_assertions": len(results),
        "overall_accuracy": overall["accuracy"],
        "overall_control_accuracy": overall["control_accuracy"],
        "by_phenomenon": by_phenomenon.replace({np.nan: None}).to_dict(orient="records"),
        "by_phenomenon_subtype_matrix_type": by_phenomenon_subtype_matrix_type.replace({np.nan: None}).to_dict(orient="records"),
    }
    with open(summary_output, "w", encoding="utf-8") as f:
        json.dump(round_floats(summary), f, indent=2)
    print(f"Wrote summary to {summary_output}")

    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", default="benchmark_data/pairs.csv")
    parser.add_argument(
        "--model", dest="models", action="append", required=True,
        help="HF model id, optionally with its own unigram corpus as NAME=CORPUS "
             "(hf:DATASET[@CONFIG] or a text file); "
             "repeat to score multiple models.",
    )
    parser.add_argument(
        "--unigram-corpus", default=None,
        help="corpus for any --model given without its own =CORPUS.",
    )
    parser.add_argument(
        "--unigram-docs", type=int, default=300_000,
        help="how many documents to stream from an hf: corpus (default: 300000).",
    )
    parser.add_argument("--text-field", default="text", help="text column of hf: corpora.")
    parser.add_argument(
        "--unigram-cache-dir", default="eval_output",
        help="where the cached unigram log-probs (.npy) for hf: corpora are stored.",
    )
    parser.add_argument("--pairs-output", default="eval_output/pairs_scored.csv")
    parser.add_argument("--details-output", default="eval_output/eval_results.csv")
    parser.add_argument("--summary-output", default="eval_output/eval_summary.json")
    parser.add_argument("--device", default=default_device())
    parser.add_argument(
        "--cache-dir", default=None,
        help="Where downloaded model weights are stored.",
    )
    parser.add_argument(
        "--childes-format", action="store_true",
        help="Score sentences in CHILDES transcript style (lowercase, space before punctuation).",
    )
    args = parser.parse_args()

    model_corpora = []
    for spec in args.models:
        model_name, _, corpus = spec.partition("=")
        corpus = corpus or args.unigram_corpus
        if not corpus:
            parser.error(f"no unigram corpus for {model_name}: use --model {model_name}=CORPUS or --unigram-corpus")
        model_corpora.append((model_name, corpus))

    pairs_df = load_pairs(args.pairs)
    if pairs_df.empty:
        raise SystemExit(f"No rows found in {args.pairs}")
    if args.childes_format:
        pairs_df = apply_childes_format(pairs_df, args)

    comparison = [
        run_slor_for_model(model_name, corpus, pairs_df, args)
        for model_name, corpus in model_corpora
    ]

    if len(comparison) > 1:
        comparison_output = insert_suffix(args.summary_output, "slor-comparison")
        with open(comparison_output, "w", encoding="utf-8") as f:
            json.dump(round_floats(comparison), f, indent=2)
        print(f"\n=== Comparison ({comparison_output}) ===")
        for summary in comparison:
            print(
                f"{summary['model']}: accuracy={summary['overall_accuracy']:.3f} "
                f"control_accuracy={summary['overall_control_accuracy']:.3f}"
            )


if __name__ == "__main__":
    main()
