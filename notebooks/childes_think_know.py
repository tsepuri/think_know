import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import re
    import sys

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import spacy
    from better_profanity import profanity

    ROOT = mo.notebook_dir().parent
    sys.path.insert(0, str(ROOT))
    import utils
    from scripts import childes as ch

    return ROOT, ch, mo, np, pd, plt, profanity, re, spacy, utils


@app.cell
def _(mo):
    mo.md(r"""
    # think vs. know in CHILDES

    The syntactic-only hypothesis: a verb that occurs with both declarative and interrogative
    complements is veridical (responsive); one that occurs with only one is not. This notebook
    measures that input for *think* and *know* in Eng-NA + Eng-UK (childes-db 2026.1 on Redivis),
    and checks that every benchmark word is frequent enough for a CHILDES-trained model to learn.
    """)
    return


@app.cell
def _(ROOT, ch, spacy):
    CACHE = ROOT / "cache"
    THRESHOLD = 10
    utterances = ch.load_utterances(CACHE)
    token_tags = ch.load_token_tags(CACHE)
    nlp = spacy.load("en_core_web_md", exclude=["ner"])
    vectors = spacy.load("en_core_web_lg", exclude=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner", "senter"])
    return CACHE, THRESHOLD, nlp, token_tags, utterances, vectors


@app.cell
def _(utterances):
    utterances.groupby("collection_name").agg(
        utterances=("id", "size"), words=("num_tokens", "sum"), transcripts=("transcript_id", "nunique")
    ).assign(target_child_share=utterances.groupby("collection_name").speaker_role.apply(lambda r: (r == "Target_Child").mean()))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Vocabulary frequency

    For a single word $w$, $\mathrm{count}(w)$ is its number of whitespace-token occurrences
    across all lowercased glosses (xxx/yyy/www dropped). A multiword entry $p$ gets its
    whole-word phrase count, and $\min_{w \in p} \mathrm{count}(w)$. An entry is **below threshold**
    when that minimum is under 10: some word in it is too rare to learn. Templates like
    `<person>'s` are skipped.
    """)
    return


@app.cell
def _(ROOT, THRESHOLD, ch, pd, utterances):
    counts = ch.word_counts(utterances.gloss)
    vocabulary = pd.read_csv(ROOT / "benchmark" / "data" / "vocabulary.csv")
    vocabulary = vocabulary[~vocabulary.word.str.contains("<")].assign(form=lambda v: v.word.str.lower())
    _multi = [f for f in vocabulary.form.unique() if " " in f]
    _phrases = ch.phrase_counts(utterances.gloss, _multi)
    vocab_freq = vocabulary[["word", "slot", "form"]].assign(
        count=lambda v: [_phrases[f] if f in _phrases else counts[f] for f in v.form],
        min_word_count=lambda v: [min(counts[w] for w in f.split()) for f in v.form],
        rarest_word=lambda v: [min(f.split(), key=counts.__getitem__) for f in v.form],
    ).assign(below_threshold=lambda v: v.min_word_count < THRESHOLD).drop_duplicates(["form", "slot"])
    vocab_freq.sort_values("min_word_count")
    return counts, vocab_freq


@app.cell
def _(mo):
    mo.md(r"""
    Every word in the benchmark itself (CHILDES-formatted `sentences_with_variants.csv`, so
    contractions stay single tokens as in CHILDES glosses), with the same threshold:
    """)
    return


@app.cell
def _(ROOT, THRESHOLD, ch, counts, pd, utils):
    _sentences = pd.read_csv(ROOT / "benchmark" / "data" / "sentences_with_variants.csv").sentence
    _formatted = [utils.normative_to_childes_formatting(s) for s in _sentences]
    benchmark_words = pd.Series(ch.word_counts(_formatted)).rename("benchmark_count").to_frame()
    benchmark_words = benchmark_words[~benchmark_words.index.str.match(r"^[.,?!;]$|^\*")]
    benchmark_words["childes_count"] = [counts[w] for w in benchmark_words.index]
    rare_benchmark_words = benchmark_words[benchmark_words.childes_count < THRESHOLD].sort_values("childes_count")
    rare_benchmark_words
    return (rare_benchmark_words,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Alternatives for rare words

    Each word's tag is its most frequent CHILDES %mor tag: part of speech plus word-level
    features (`noun-Plur`, `verb-Fin Past`), so a replacement keeps number, tense and gender.
    Case, subject agreement and mood vary with the utterance, so they are dropped. An unattested
    word takes the tag of its nearest attested neighbour. Candidates are words with that tag and
    $\mathrm{count} \ge 10$, ranked by cosine similarity of spaCy `en_core_web_lg` vectors (md shares 20k vectors across
    its whole vocabulary, so rare words get their neighbours' vectors):
    $\cos(v_w, v_c) = v_w \cdot v_c / (\lVert v_w \rVert \lVert v_c \rVert)$. Similarity ranks
    plausible swaps; the `requires` features in `vocabulary.csv` (breakable, liquid, ...) still
    need a manual check.
    """)
    return


@app.cell
def _(
    THRESHOLD,
    ch,
    counts,
    pd,
    profanity,
    rare_benchmark_words,
    token_tags,
    vectors,
    vocab_freq,
):
    tags = ch.word_tags(token_tags)
    # Profanity is grammatical but inappropriate in a child-directed benchmark
    profanity.load_censor_words()
    _rare_vocab = {w for f in vocab_freq.form for w in f.split() if counts[w] < THRESHOLD}
    _rare_all = _rare_vocab | set(rare_benchmark_words.index)
    no_vector = sorted(w for w in _rare_all if vectors.vocab[w].vector_norm == 0)
    _rare = sorted(_rare_all - set(no_vector))
    _alts = ch.alternatives(_rare, tags, {w: c for w, c in counts.items() if not profanity.contains_profanity(w)}, vectors, THRESHOLD)
    rare_alternatives = pd.DataFrame(
        [(w, counts[w], tag, ", ".join(f"{a} ({counts[a]:,})" for a in alts)) for w, (tag, alts) in _alts.items()],
        columns=["word", "childes_count", "tag", "alternatives"],
    ).sort_values("childes_count")
    rare_alternatives
    return (no_vector,)


@app.cell
def _(mo, no_vector):
    mo.md(f"""
    Rare words without a spaCy vector (no alternatives computed): {', '.join(no_vector) or 'none'}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Whole-sentence frequency

    How often each benchmark sentence (base and lexical variants) occurs verbatim as a CHILDES
    utterance: lowercased glosses (xxx/yyy/www dropped) against the CHILDES-formatted sentence
    without punctuation. A mini-discourse is matched turn by turn; `count` is its rarest turn's
    count and `turn_counts` lists each turn's.
    """)
    return


@app.cell
def _(ROOT, ch, pd, re, utils, utterances):
    _utterance_counts = pd.Series([" ".join(ch.tokens(g)) for g in utterances.gloss]).value_counts()
    _s = pd.read_csv(ROOT / "benchmark" / "data" / "sentences_with_variants.csv").drop_duplicates("sentence")
    _turns = [
        [" ".join(w for w in t.split() if w not in utils.PUNCTUATION) for t in re.split(r"\*\w+:", utils.normative_to_childes_formatting(x)) if t.strip()]
        for x in _s.sentence
    ]
    _turn_counts = [[int(_utterance_counts.get(t, 0)) for t in ts] for ts in _turns]
    sentence_freq = _s[["sentence", "paradigm", "variant_id", "grammaticality"]].assign(
        count=[min(c) for c in _turn_counts],
        turn_counts=[", ".join(map(str, c)) if len(c) > 1 else "" for c in _turn_counts],
    ).sort_values("count", ascending=False)
    sentence_freq
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Complement classification

    spaCy (`en_core_web_md`) parses each utterance containing a think/know form. CHILDES glosses
    have no punctuation, so the parser input ends in `?` for question-type utterances and `.`
    otherwise, and `i` is capitalized. For each token with lemma think/know tagged VERB, only
    dependents whose whole span is right of the verb count as complements, checked in this order:

    - **interrogative**: a clausal dependent (ccomp/xcomp/advcl) whose first word is *whether*,
      *if*, or a wh-word; or a wh-word object (*you know what?*, `wh_bare`; *I know what you mean*,
      `wh_object`, where free relatives and concealed questions are indistinguishable).
    - **declarative**: a ccomp otherwise (`that` if it starts with complementizer *that*, else `zero`).
      A clause whose first *if*/wh-word sits in an adverbial or subject clause inside it is
      declarative too (`adjunct_first`: *I think [if you push it] it'll go*, *I think [what we'll do] is play*).
    - **infinitival** (*think to*), **so** (*think so*), **NP** (object), **PP** (*about*/*of*).
    - With nothing to the right: **matrix_wh** (*what do you think?*), **left_clause**
      (parentheticals, *it's raining I think*), else **none** (*I don't know*).

    Known-answer check: the benchmark's think/know single sentences carry `embedded_type`
    labels with the same names. Benchmark `none` includes parentheticals, which this classifier
    splits into `left_clause` by design.
    """)
    return


@app.cell
def _(ROOT, ch, nlp, pd, utils):
    _s = pd.read_csv(ROOT / "benchmark" / "data" / "sentences_with_variants.csv")
    _s = _s[_s.matrix_verb.isin(["think", "know"]) & (_s.paradigm == "single_sentence")].drop_duplicates("sentence")
    _texts = [
        ch.parser_input(utils.normative_to_childes_formatting(x).rstrip(" .?"), "question" if x.endswith("?") else "declarative")
        for x in _s.sentence
    ]
    _verbs, _ = ch.classify_texts(_texts, nlp)
    _pred = _verbs.drop_duplicates("text_index").set_index("text_index").category.reindex(range(len(_s))).fillna("no verb found")
    benchmark_check = pd.crosstab(_s.embedded_type.rename("benchmark label").to_numpy(), _pred.rename("classifier").to_numpy())
    benchmark_check.index.name, benchmark_check.columns.name = "benchmark label", "classifier"
    benchmark_check
    return


@app.cell
def _(CACHE, ch, nlp, utterances):
    verbs = ch.classify_utterances(utterances, nlp, CACHE / "childes_think_know_verbs.parquet")
    verbs = verbs.assign(speaker=verbs.speaker_role.eq("Target_Child").map({True: "target child", False: "other speakers"}))
    return (verbs,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Complement distribution

    Per lemma $v$ with category counts $n_c$ and total $N = \sum_c n_c$:

    - $\text{p\_decl\_of\_clausal} = n_\text{decl} / (n_\text{decl} + n_\text{interrog})$, and $\text{p\_interrog\_of\_clausal} = 1 - \text{p\_decl\_of\_clausal}$
    - $\text{p\_other} = 1 - (n_\text{decl} + n_\text{interrog}) / N$, and $\text{p\_none} = n_\text{none} / N$

    `total` counts spaCy verb tokens. The raw form count (whole-word matches of think/thinks/
    thinking/thought, know/knows/knew/known/knowing) also includes nouns (*a thought*), shown
    for comparison.
    """)
    return


@app.cell
def _(ch, counts, pd, verbs):
    category_counts = ch.category_table(verbs)
    category_counts.assign(raw_form_count=pd.Series({l: sum(counts[f] for f in fs) for l, fs in ch.TARGET_FORMS.items()}))
    return (category_counts,)


@app.cell
def _(verbs):
    verbs.groupby(["lemma", "category", "detail"]).size().rename("n").reset_index().query("category in ['declarative', 'interrogative']").pivot_table(index=["category", "detail"], columns="lemma", values="n", fill_value=0)
    return


@app.cell
def _(mo):
    mo.md(r"""
    By speaker (models train on all speakers; child-directed input is `other speakers`), by
    CHILDES speaker role (most frequent first), and by target-child age in years (ages are stored in days):
    """)
    return


@app.cell
def _(ch, verbs):
    ch.category_table(verbs, by=["speaker"]).reorder_levels(["speaker", "lemma"]).sort_index()
    return


@app.cell
def _(ch, pd, verbs):
    _roles = verbs.speaker_role.value_counts().index
    _table = ch.category_table(verbs, by=["speaker_role"]).reorder_levels(["speaker_role", "lemma"])
    # every role gets a know and a think row, zero counts included
    _counts = [c for c in _table.columns if not c.startswith("p_")]
    _table.reindex(pd.MultiIndex.from_product([_roles, sorted(ch.TARGET_FORMS)], names=_table.index.names)).fillna({c: 0 for c in _counts}).astype({c: int for c in _counts})
    return


@app.cell
def _(ch, pd, utterances, verbs):
    _bins, _labels = [0, 2, 3, 4, 5, 6, float("inf")], ["<2", "2", "3", "4", "5", "6+"]
    _utterance_age = pd.cut(utterances.target_child_age / 365.25, _bins, labels=_labels, right=False).rename("age_years")
    _utterance_speaker = utterances.speaker_role.eq("Target_Child").map({True: "target child", False: "other speakers"}).rename("speaker")
    # words spoken in transcripts of a target child that age, by all speakers and by each speaker group
    _words = pd.concat([
        utterances.num_tokens.groupby([_utterance_age, pd.Series("all speakers", utterances.index, name="speaker")], observed=True).sum(),
        utterances.num_tokens.groupby([_utterance_age, _utterance_speaker], observed=True).sum(),
    ]).rename("words_spoken")
    _aged = verbs.assign(age_years=pd.cut(verbs.target_child_age / 365.25, _bins, labels=_labels, right=False)).dropna(subset=["age_years"])
    age_table = pd.concat([
        ch.category_table(_aged.assign(speaker="all speakers"), by=["age_years", "speaker"]),
        ch.category_table(_aged, by=["age_years", "speaker"]),
    ]).join(_words, on=["age_years", "speaker"]).sort_index()
    age_table = age_table.assign(pct_of_words=100 * age_table.total / age_table.words_spoken)
    age_table[["declarative", "interrogative", "total", "words_spoken", "pct_of_words", "p_decl_of_clausal", "p_interrog_of_clausal"]]
    return (age_table,)


@app.cell(hide_code=True)
def _(CACHE, age_table, plt, utils):
    _colors = {"know": "#2a78d6", "think": "#eb6834"}
    _groups = ["target child", "other speakers", "all speakers"]
    _fig, _axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    for _ax, _speaker in zip(_axes, _groups):
        for _lemma, _color in _colors.items():
            _rows = age_table.xs((_lemma, _speaker), level=["lemma", "speaker"])
            _ax.plot(_rows.index.astype(str), _rows.pct_of_words, color=_color, lw=2, marker="o", ms=4, label=_lemma)
        _ax.set(title=_speaker, xlabel="target-child age (years)")
        _ax.title.set_size(10)
        _ax.spines[["top", "right"]].set_visible(False)
    _axes[0].set_ylabel("% of words spoken")
    _axes[0].legend(frameon=False, fontsize=8)
    _axes[0].set_ylim(bottom=0)
    _fig.tight_layout()
    utils.save_figure(_fig, str(CACHE / "figures" / "childes_frequency_by_age.png"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Verb frequency

    The 50 most frequent verbs across all speakers, counting each %mor `verb` token toward its stem
    (*knows*, *knew* → *know*). Auxiliaries and copulas carry their own tags, so they're excluded.
    """)
    return


@app.cell(hide_code=True)
def _(CACHE, ch, plt, utils):
    verb_stems = ch.load_verb_stems(CACHE).nlargest(50, "count").iloc[::-1]
    _colors = {"know": "#2a78d6", "think": "#eb6834"}
    _fig, _ax = plt.subplots(figsize=(6, 10))
    _ax.barh(verb_stems.stem, verb_stems["count"], color=[_colors.get(s, "#b8b6b0") for s in verb_stems.stem], height=0.75)
    _ax.set(xlabel="verb tokens", ylim=(-0.6, len(verb_stems) - 0.4))
    _ax.xaxis.set_major_formatter(lambda x, _: f"{x / 1000:.0f}k")
    _ax.tick_params(axis="y", length=0, labelsize=8)
    _ax.spines[["top", "right", "left"]].set_visible(False)
    _fig.tight_layout()
    utils.save_figure(_fig, str(CACHE / "figures" / "childes_top_verbs.png"))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Is the contrast categorical?

    The hypothesis reads "occurs with both" categorically, so a handful of genuine *think* +
    interrogative tokens would count. Utterances within a transcript are correlated, so the 95% CI
    on $p = n_\text{interrog} / (n_\text{decl} + n_\text{interrog})$ comes from resampling
    transcripts with replacement (2000 draws). Below that: how many corpora have any *think* +
    interrogative token, and a sample of such tokens with the turn before and after (speaker codes
    as in the transcript, the token's own turn in brackets), for checking by hand whether they are
    parse errors.
    """)
    return


@app.cell
def _(ch, verbs):
    ch.bootstrap_p_interrog(verbs)
    return


@app.cell
def _(verbs):
    _clausal = verbs[verbs.category.isin(["declarative", "interrogative"])]
    _by_corpus = _clausal.groupby(["lemma", "corpus_name"]).category.agg(interrog=lambda c: (c == "interrogative").sum(), clausal="size")
    _by_corpus.groupby("lemma").agg(
        corpora=("clausal", "size"),
        corpora_with_interrog=("interrog", lambda n: (n > 0).sum()),
        median_interrog_share=("interrog", lambda n: (n / _by_corpus.loc[n.index, "clausal"]).median()),
    )
    return


@app.cell
def _(ch, utterances, verbs):
    think_interrogatives = ch.with_context(verbs.query("lemma == 'think' and category == 'interrogative'"), utterances)[["detail", "speaker", "speaker_code", "corpus_name", "text", "context"]]
    think_interrogatives.sample(min(60, len(think_interrogatives)), random_state=0).sort_values("detail")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Ablation sweep

    Let $I_k$, $D_k$, $N_k$ be know's interrogative, declarative and total counts, and $I_t$, $D_t$
    think's. Removing a fraction $f$ of know + interrogative uses leaves know with ratio
    $D_k / ((1-f) I_k)$ and removes a share $f I_k / N_k$ of all know uses (the confound).
    Counterfactual 1 relabels those $f I_k$ uses as think, giving think the ratio $D_t / (I_t + f I_k)$.

    **Figure criterion.** If the input supports the syntactic hypothesis, know's ratio at $f = 0$
    sits well below think's (know takes far more interrogatives), and the two curves cross for
    some $f$. The right panel tests the planning claim that full ablation removes over half of know's uses:
    it holds if the curve ends above 0.5 at $f = 1$.
    """)
    return


@app.cell
def _(category_counts, ch, np):
    sweep = ch.ablation_sweep(category_counts, np.linspace(0, 1, 11))
    sweep
    return (sweep,)


@app.cell
def _(CACHE, category_counts, plt, sweep, utils):
    _blue, _orange, _muted = "#2a78d6", "#eb6834", "#52514e"
    _fig, (_left, _right) = plt.subplots(1, 2, figsize=(10, 3.8))
    _left.plot(sweep.fraction, sweep.know_decl_per_interrog, color=_blue, lw=2, marker="o", ms=4, label="know, interrogatives removed")
    _left.plot(sweep.fraction, sweep.think_decl_per_interrog_cf1, color=_orange, lw=2, marker="o", ms=4, label="think, removed uses relabeled as think")
    _left.axhline(category_counts.loc["think", "declarative"] / category_counts.loc["think", "interrogative"], color=_orange, lw=1, ls="--", label="think, observed")
    _left.set(yscale="log", xlabel="fraction f of know + interrogative uses moved", ylabel="declarative per interrogative")
    _left.legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1, 0.9))
    _right.plot(sweep.fraction, sweep.share_of_know_removed, color=_blue, lw=2, marker="o", ms=4)
    _right.axhline(0.5, color=_muted, lw=1, ls=":")
    _right.set(xlabel="fraction f of know + interrogative uses removed", ylabel="share of all know uses removed", ylim=(0, 1))
    for _ax in (_left, _right):
        _ax.spines[["top", "right"]].set_visible(False)
    _fig.tight_layout()
    utils.save_figure(_fig, str(CACHE / "figures" / "childes_ablation_sweep.png"))
    return


@app.cell
def _(mo):
    mo.md("""
    ## Samples for manual review

    Five random utterances per lemma and category
    """)
    return


@app.cell
def _(ch, utterances, verbs):
    ch.with_context(verbs.sample(frac=1, random_state=0).groupby(["lemma", "category"]).head(10), utterances).sort_values(["lemma", "category"])[["lemma", "category", "detail", "speaker_code", "text", "context"]]
    return


if __name__ == "__main__":
    app.run()
