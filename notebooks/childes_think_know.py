import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import re
    import sys

    import marimo as mo
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import pandas as pd
    import spacy
    from better_profanity import profanity

    ROOT = mo.notebook_dir().parent
    sys.path.insert(0, str(ROOT))
    import utils
    from scripts import childes as ch

    return ROOT, ch, mo, np, nx, pd, plt, profanity, re, spacy, utils


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
def _(ROOT, ch, pd, utterances):
    counts = ch.word_counts(utterances.gloss)
    vocabulary = pd.read_csv(ROOT / "benchmark" / "data" / "vocabulary.csv")
    vocabulary = vocabulary[~vocabulary.word.str.contains("<")].assign(form=lambda v: v.word.str.lower())
    _multi = [f for f in vocabulary.form.unique() if " " in f]
    _phrases = ch.phrase_counts(utterances.gloss, _multi)
    vocab_freq = vocabulary[["word", "slot", "form"]].assign(
        count=lambda v: [_phrases[f] if f in _phrases else counts[f] for f in v.form],
        min_word_count=lambda v: [min(counts[w] for w in f.split()) for f in v.form],
        rarest_word=lambda v: [min(f.split(), key=counts.__getitem__) for f in v.form],
    ).assign(below_threshold=lambda v: v.min_word_count < 100).drop_duplicates(["form", "slot"])
    vocab_freq.sort_values("min_word_count")
    return counts, vocab_freq


@app.cell
def _(mo):
    mo.md(r"""
    `vocabulary.csv` keeps two frequency columns, for thresholding its entries later:

    - `childes_count`: $\mathrm{count}(w)$, or the whole-word phrase count for a multiword entry.
    - `childes_lemma_count`: $\mathrm{count}(\ell)$, the number of CHILDES morphemes (`token_morpheme`,
      stems and clitics alike) with lemma $\ell$, where a word's $\ell$ is the lemma CHILDES most often
      gives its stem morpheme (*papers* → *paper*, *hid* → *hide*, *message's* → *message*). A
      contraction unattested as a gloss (*meeting's*) takes its base word's lemma; any other
      unattested word counts 0. A multiword entry gets its rarest word's lemma count,
      $\min_{w \in p} \mathrm{count}(\mathrm{lemma}(w))$. Templates like `<person>'s` stay blank.
    """)
    return


@app.cell
def _(CACHE, ROOT, ch, pd, re, vocab_freq):
    _path = ROOT / "benchmark" / "data" / "vocabulary.csv"
    _vocab = pd.read_csv(_path, dtype=str, keep_default_na=False)
    _forms = _vocab.word.str.lower()
    _entry_counts = dict(zip(vocab_freq.form, vocab_freq["count"]))
    _gloss_lemmas = ch.load_gloss_lemmas(CACHE).sort_values("count", ascending=False).drop_duplicates("gloss")
    _lemma_of = dict(zip(_gloss_lemmas.gloss, _gloss_lemmas.lemma))
    _lemma_counts = ch.load_lemma_counts(CACHE).set_index("lemma")["count"]
    _vocab["childes_count"] = [str(_entry_counts[f]) if f in _entry_counts else "" for f in _forms]
    _vocab["childes_lemma_count"] = [
        str(min(int(_lemma_counts.get(_lemma_of.get(w, _lemma_of.get(re.sub(r"'(s|re|ll|ve|d|m)$|n't$", "", w))), 0)) for w in f.split())) if f in _entry_counts else "" for f in _forms
    ]
    _vocab.to_csv(_path, index=False)
    _vocab[["word", "slot", "childes_count", "childes_lemma_count"]]
    return


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
    alternative_words = ch.alternatives(_rare, tags, {w: c for w, c in counts.items() if not profanity.contains_profanity(w)}, vectors, THRESHOLD)
    rare_alternatives = pd.DataFrame(
        [(w, counts[w], tag, ", ".join(f"{a} ({counts[a]:,})" for a in alts)) for w, (tag, alts) in alternative_words.items()],
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


@app.cell
def _(ROOT, counts, pd):
    # rare_vocabulary.csv is hand-curated (started from alternative_words): one row per word, each
    # rare word ("original") followed by the alternatives kept for it; this only refreshes the counts
    _path = ROOT / "benchmark" / "data" / "rare_vocabulary.csv"
    rare_vocabulary = pd.read_csv(_path, keep_default_na=False).assign(childes_count=lambda v: [counts[w] for w in v.word])
    rare_vocabulary.to_csv(_path, index=False)
    rare_vocabulary
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

    Each mental-verb token (a `ch.TARGET_FORMS` stem tagged verb on the CHILDES %mor tier) is
    classified from its CHILDES %gra dependency parse (UD relations). xxx/yyy/www carry no %gra
    and are dropped; utterances with any other unparsed token are skipped. UD attaches a PP's
    noun as `obl` with the preposition as its `case` dependent, and keeps *n't* on its auxiliary
    (*don't* is one token). %gra gives a clitic (the *'s* of *there's*) an index but no token
    row, so its dependents would be cut off; it is rebuilt and attached to the verb before it
    (see `ch._gra_nodes`). Only dependents whose whole span is right of the verb count as
    complements, checked in this order:

    - **interrogative**: a clausal dependent (ccomp/xcomp/advcl) *opened* by *whether*, *if*, or a
      wh-word (skipping fillers and the clause's own non-wh adverbs: *know [exactly where...]*);
      a wh-word object (*you know what?*, `wh_bare`; *I know what you mean*, `wh_object`, where
      free relatives and concealed questions are indistinguishable); a wh-determiner opening the
      object (*know [which one]*, `wh_noun_phrase`); a bare wh-adverb (*I don't know [why]*,
      `wh_bare`). A wh-word deeper in the clause (*think [you know what this is]*) or in an
      object's relative clause (*know [a lady who...]*) does not count.
    - **declarative**: a ccomp otherwise (`that` if it starts with complementizer *that*, else `zero`).
      A clause whose first *if*/wh-word sits in an adverbial or subject clause inside it is
      declarative too (`adjunct_first`: *I think [if you push it] it'll go*, *I think [what we'll do] is play*),
      as is an *if*-clause followed by the clause it conditions. An *if*-clause %gra labels advcl is a
      conditional adjunct and is skipped.
    - **infinitival** (*think to*), **so** (*think so*), **NP** (object), **PP** (*about*/*of*).
    - With nothing to the right: **matrix_wh** (*what do you think?*), **left_clause**
      (parentheticals, *it's raining I think*), else **none** (*I don't know*).

    `polar` marks a verb heading a yes/no question: an auxiliary before its subject and no
    matrix wh-word (*do you know where it is?*; not *you know what?*, which has no inversion).

    Cross-check: spaCy (`en_core_web_md`) runs the same rules on its own parse of each
    utterance (parsed once, cached as a DocBin). CHILDES glosses have no punctuation, so its
    input ends in `?` for question-type utterances and `.` otherwise, and `i` is capitalized.
    Its known-answer check uses the benchmark's think/know single sentences, which carry
    `embedded_type` labels with the same names and have no CHILDES parse. Benchmark `none`
    includes parentheticals, which the rules split into `left_clause` by design.
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
    _verbs = _verbs[_verbs.lemma.isin(["think", "know"])]
    _pred = _verbs.drop_duplicates("text_index").set_index("text_index").category.reindex(range(len(_s))).fillna("no verb found")
    benchmark_check = pd.crosstab(_s.embedded_type.rename("benchmark label").to_numpy(), _pred.rename("classifier").to_numpy())
    benchmark_check.index.name, benchmark_check.columns.name = "benchmark label", "classifier"
    benchmark_check
    return


@app.cell
def _(CACHE, ch, nlp, utterances):
    target_tokens = ch.load_target_tokens(CACHE)
    verbs = ch.classify_gra_utterances(target_tokens, utterances)
    verbs = verbs.assign(speaker=verbs.speaker_role.eq("Target_Child").map({True: "target child", False: "other speakers"}))
    think_know = verbs[verbs.lemma.isin(["think", "know"])]
    spacy_verbs = ch.classify_utterances(utterances, nlp, CACHE / "childes_mental_verb_docs.spacy")
    return spacy_verbs, target_tokens, think_know, verbs


@app.cell
def _(mo):
    mo.md(r"""
    ### Accuracy against gold trees

    [UD_English-CHILDES](https://github.com/UniversalDependencies/UD_English-CHILDES) has hand-checked
    UD trees for ~48k CHILDES utterances (copied from the
    [CAIT toolkit](https://github.com/fpadovani/CAIT-Toolkit) repo into `cache/ud_english_childes/`,
    with CAIT's own predictions on the test split). The reference label is the rules applied to the
    gold tree, so this measures parse errors, not rule errors. %gra and spaCy are matched to gold
    sentences by corpus and gloss and scored on all three splits. CAIT (a RoBERTa-large biaffine
    parser trained on the train split) is scored on test only, against %gra on the same sentences.
    """)
    return


@app.cell
def _(CACHE, ch, pd, spacy_verbs, utterances, verbs):
    _dir = CACHE / "ud_english_childes"
    _splits = {s: ch.read_conllu(_dir / f"en_childes-ud-{s}.conllu") for s in ("train", "dev", "test")}
    _cait = ch.classify_conllu(ch.read_conllu(_dir / "supar_childes_roberta_test.conllu"), _splits["test"])


    def _childes_db(classified, sentences):
        # classified childes-db verbs, keyed like classify_conllu rows of `sentences`
        ids = ch.gold_utterance_ids(sentences, utterances).rename("id").reset_index()
        return classified.assign(k=classified.groupby(["id", "lemma"]).cumcount()).merge(ids, on="id")


    _all = [s for split in _splits.values() for s in split]
    _rows = {}
    for _scope, _lemmas in [("think+know", ["think", "know"]), ("all verbs", list(ch.TARGET_FORMS))]:
        _keep = lambda d: d[d.lemma.isin(_lemmas)]
        _gold_all, _gold_test = _keep(ch.classify_conllu(_all)), _keep(ch.classify_conllu(_splits["test"]))
        _rows[("all splits", "%gra", _scope)] = ch.gold_accuracy(_gold_all, _childes_db(verbs, _all))
        _rows[("all splits", "spaCy md", _scope)] = ch.gold_accuracy(_gold_all, _childes_db(spacy_verbs, _all))
        _rows[("test", "%gra", _scope)] = ch.gold_accuracy(_gold_test, _childes_db(verbs, _splits["test"]))
        _rows[("test", "CAIT", _scope)] = ch.gold_accuracy(_gold_test, _cait)
    pd.DataFrame(_rows).T.rename_axis(["gold", "parse", "verbs"]).astype({"n": int, "n_clausal": int}).round(3)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### %gra vs. spaCy

    Matching each utterance's $k$-th token of each verb across the two parses. Agreement is reported
    over all matched tokens, over those that either parse calls declarative or interrogative, and
    over those both do; the crosstab (think and know; rows %gra, columns spaCy) covers the middle set.
    """)
    return


@app.cell
def _(pd, spacy_verbs, verbs):
    _k = lambda v: v.assign(k=v.groupby(["id", "lemma"]).cumcount())
    matched = _k(verbs).merge(_k(spacy_verbs)[["id", "lemma", "k", "category", "detail"]], on=["id", "lemma", "k"], suffixes=("_gra", "_spacy"))
    _clausal = ["declarative", "interrogative"]
    _either = matched.category_gra.isin(_clausal) | matched.category_spacy.isin(_clausal)
    _both = matched.category_gra.isin(_clausal) & matched.category_spacy.isin(_clausal)
    _agree = matched.category_gra == matched.category_spacy
    _tk = matched.lemma.isin(["think", "know"])
    print(f"{len(matched):,} matched verb tokens; categories agree on {_agree.mean():.1%}")
    print(f"either parse clausal: {_either.sum():,} tokens, agree on {_agree[_either].mean():.1%}")
    print(f"both parses clausal: {_both.sum():,} tokens, agree on {_agree[_both].mean():.1%}")
    _x = matched[_either & _tk]
    pd.crosstab([_x.lemma, _x.category_gra.rename("%gra")], _x.category_spacy.rename("spaCy"))
    return (matched,)


@app.cell
def _(ch, matched, utterances):
    # think/know tokens where either parse says declarative or interrogative and the two disagree,
    # up to 10 per (%gra category, spaCy category) pair
    _clausal = ["declarative", "interrogative"]
    disagreements = matched[
        matched.lemma.isin(["think", "know"])
        & (matched.category_gra != matched.category_spacy)
        & (matched.category_gra.isin(_clausal) | matched.category_spacy.isin(_clausal))
    ]
    _sample = disagreements.sample(frac=1, random_state=0).groupby(["lemma", "category_gra", "category_spacy"]).head(10)
    ch.with_context(_sample, utterances).sort_values(["lemma", "category_gra", "category_spacy"])[
        ["id", "lemma", "category_gra", "detail_gra", "category_spacy", "detail_spacy", "context"]
    ]
    return (disagreements,)


@app.cell
def _(ch, mo, nlp, nx, pd, plt, spacy_verbs, target_tokens, utterances, verbs):
    def _draw(ax, words, title):
        """Dependency tree of (index, word, head index or None, relation) tuples: x by word order,
        y by depth, mental verbs in orange."""
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        if not words:
            ax.text(0.5, 0.5, "no parse", ha="center", va="center", color="#8a8984", transform=ax.transAxes)
            return
        graph = nx.DiGraph()
        graph.add_nodes_from(i for i, *_ in words)
        graph.add_edges_from((h, i, {"rel": r}) for i, _, h, r in words if h is not None)
        order = {i: n for n, (i, *_) in enumerate(sorted(words))}
        depth = {i: len(nx.ancestors(graph, i)) for i in graph}
        pos = {i: (order[i], -depth[i]) for i in graph}
        lemmas = {f for fs in ch.TARGET_FORMS.values() for f in fs}
        colors = ["#eb6834" if w in lemmas else "#cde2fb" for _, w, *_ in sorted(words)]
        nx.draw_networkx_nodes(graph, pos, nodelist=[i for i, *_ in sorted(words)], node_color=colors, node_size=500, ax=ax)
        nx.draw_networkx_labels(graph, pos, labels={i: w for i, w, *_ in words}, font_size=8, ax=ax)
        nx.draw_networkx_edges(graph, pos, arrows=True, arrowsize=10, node_size=500, edge_color="#8a8984", ax=ax)
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=nx.get_edge_attributes(graph, "rel"), font_size=7, font_color="#52514e", ax=ax)


    def compare_parses(utterance_id):
        """%gra and spaCy trees of one utterance side by side, with each parse's classification of its mental verbs."""
        rows = target_tokens[(target_tokens.utterance_id == utterance_id) & target_tokens.gra_index.notna()]
        gra = [(t.i, t.lower_, t.head.i if t.head is not None else None, t.dep_) for t in ch.gra_tree(rows).values()]
        gloss, kind = utterances.loc[utterances.id == utterance_id, ["gloss", "type"]].iloc[0]
        doc = nlp(ch.parser_input(gloss, kind))
        spacy_words = [(t.i, t.lower_, None if t.head == t else t.head.i, t.dep_) for t in doc if not t.is_punct]
        fig, (left, right) = plt.subplots(1, 2, figsize=(16, 4.5))
        _draw(left, gra, "%gra")
        _draw(right, spacy_words, "spaCy")
        fig.tight_layout()
        classified = pd.concat({"%gra": verbs[verbs.id == utterance_id], "spaCy": spacy_verbs[spacy_verbs.id == utterance_id]}, names=["parse"])
        return mo.vstack([mo.md(f"**{gloss}** ({kind})"), fig, classified[["lemma", "form", "category", "detail", "polar"]]])

    return (compare_parses,)


@app.cell
def _(disagreements, mo):
    utterance_picker = mo.ui.dropdown(
        options={f"{r.id}: {r.text}"[:120]: r.id for r in disagreements.drop_duplicates("id").head(300).itertuples()},
        label="disagreement to inspect",
    )
    utterance_picker
    return (utterance_picker,)


@app.cell
def _(compare_parses, utterance_picker):
    compare_parses(utterance_picker.value) if utterance_picker.value else None
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Complement distribution

    Per lemma $v$ with category counts $n_c$ and total $N = \sum_c n_c$:

    - $\text{p\_decl\_of\_clausal} = n_\text{decl} / (n_\text{decl} + n_\text{interrog})$, and $\text{p\_interrog\_of\_clausal} = 1 - \text{p\_decl\_of\_clausal}$
    - $\text{p\_other} = 1 - (n_\text{decl} + n_\text{interrog}) / N$, and $\text{p\_none} = n_\text{none} / N$

    `total` counts the classified verb tokens. The raw form count (whole-word matches of the
    `ch.TARGET_FORMS` forms, e.g. think/thinks/thinking/thought) also includes nouns (*a thought*,
    *a dream*), shown for comparison.
    """)
    return


@app.cell
def _(ch, counts, pd, verbs):
    category_counts = ch.category_table(verbs)
    category_counts.assign(raw_form_count=pd.Series({l: sum(counts[f] for f in fs) for l, fs in ch.TARGET_FORMS.items()})).sort_values("total", ascending=False)
    return (category_counts,)


@app.cell
def _(mo):
    mo.md(r"""
    Polar questions: per lemma and complement type, the share of tokens whose verb heads a yes/no
    question (`polar`, from subject–auxiliary inversion: *do you know where it is?*, *do you think
    it's raining?*). Intonation-only questions (*you know what?*) are not counted.
    """)
    return


@app.cell
def _(ch, verbs):
    ch.polar_table(verbs).sort_values(("interrogative", "n"), ascending=False)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## All mental verbs

    Per verb: $p = n_\text{interrog} / (n_\text{decl} + n_\text{interrog})$ with its transcript-bootstrap
    95% CI (see below), and how often the verb occurs (classified verb tokens, all complement types).
    Color is the class the verb is usually described as belonging to (`ch.VERB_CLASSES`):
    responsive verbs should take both complement types, anti-rogative ones declaratives only,
    rogative ones interrogatives only.
    """)
    return


@app.cell
def _(ch, verbs):
    p_interrog = ch.bootstrap_p_interrog(verbs)
    return (p_interrog,)


@app.cell(hide_code=True)
def _(CACHE, category_counts, ch, p_interrog, plt, utils):
    _colors = {"responsive": "#2a78d6", "anti-rogative": "#eb6834", "rogative": "#1baf7a", "other": "#b8b6b0"}
    _rows = p_interrog.join(category_counts[["declarative", "interrogative", "total"]]).sort_values("p_interrog")
    _color = [_colors[ch.VERB_CLASSES[l]] for l in _rows.index]
    _y = range(len(_rows))
    _fig, (_left, _right) = plt.subplots(1, 2, figsize=(10, 0.28 * len(_rows) + 1.2), sharey=True, gridspec_kw={"width_ratios": [3, 2]})
    _left.hlines(_y, _rows.ci_low, _rows.ci_high, color=_color, lw=2)
    _left.scatter(_rows.p_interrog, _y, color=_color, s=36, zorder=3)
    _left.set(xlim=(0, 1), xlabel="interrogative share of clausal complements", yticks=list(_y), yticklabels=_rows.index)
    _right.barh(_y, _rows.total, color=_color, height=0.7)
    _right.set(xscale="log", xlabel="verb tokens (log scale)")
    for _ax in (_left, _right):
        _ax.spines[["top", "right"]].set_visible(False)
        _ax.tick_params(axis="y", length=0, labelsize=9)
        _ax.grid(axis="x", color="#e8e7e3", lw=0.8)
        _ax.set_axisbelow(True)
    for _label in _left.get_yticklabels():
        _label.set_fontweight("bold" if _label.get_text() in ("think", "know") else "normal")
    _left.legend(handles=[plt.Line2D([], [], color=c, marker="o", lw=2, label=k) for k, c in _colors.items()], frameon=False, fontsize=8, loc="lower right")
    _fig.tight_layout()
    utils.save_figure(_fig, str(CACHE / "figures" / "childes_mental_verbs.png"))
    _fig
    return


@app.cell
def _(think_know):
    think_know.groupby(["lemma", "category", "detail"]).size().rename("n").reset_index().query("category in ['declarative', 'interrogative']").pivot_table(index=["category", "detail"], columns="lemma", values="n", fill_value=0)
    return


@app.cell
def _(mo):
    mo.md(r"""
    By speaker (models train on all speakers; child-directed input is `other speakers`), by
    CHILDES speaker role (most frequent first), and by target-child age in years (ages are stored in days):
    """)
    return


@app.cell
def _(ch, think_know):
    ch.category_table(think_know, by=["speaker"]).reorder_levels(["speaker", "lemma"]).sort_index()
    return


@app.cell
def _(ch, pd, think_know):
    _roles = think_know.speaker_role.value_counts().index
    _table = ch.category_table(think_know, by=["speaker_role"]).reorder_levels(["speaker_role", "lemma"])
    # every role gets a know and a think row, zero counts included
    _counts = [c for c in _table.columns if not c.startswith("p_")]
    _table.reindex(pd.MultiIndex.from_product([_roles, ["know", "think"]], names=_table.index.names)).fillna({c: 0 for c in _counts}).astype({c: int for c in _counts})
    return


@app.cell
def _(ch, pd, think_know, utterances):
    _bins, _labels = [0, 2, 3, 4, 5, 6, float("inf")], ["<2", "2", "3", "4", "5", "6+"]
    _utterance_age = pd.cut(utterances.target_child_age / 365.25, _bins, labels=_labels, right=False).rename("age_years")
    _utterance_speaker = utterances.speaker_role.eq("Target_Child").map({True: "target child", False: "other speakers"}).rename("speaker")
    # words spoken in transcripts of a target child that age, by all speakers and by each speaker group
    _words = pd.concat([
        utterances.num_tokens.groupby([_utterance_age, pd.Series("all speakers", utterances.index, name="speaker")], observed=True).sum(),
        utterances.num_tokens.groupby([_utterance_age, _utterance_speaker], observed=True).sum(),
    ]).rename("words_spoken")
    _aged = think_know.assign(age_years=pd.cut(think_know.target_child_age / 365.25, _bins, labels=_labels, right=False)).dropna(subset=["age_years"])
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
def _(p_interrog):
    p_interrog.loc[["know", "think"]]
    return


@app.cell
def _(think_know):
    _clausal = think_know[think_know.category.isin(["declarative", "interrogative"])]
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

    **Augmentation (counterfactual 2, bottom row).** Add $m I_k$ think + interrogative uses and leave
    know as observed. Think's ratio becomes $D_t / (I_t + m I_k)$, and the added uses are a share
    $m I_k / (N_t + m I_k)$ of all think uses (the confound: synthetic think input). Think's ratio falls
    to know's observed $D_k / I_k$ at $m^* = (D_t I_k / D_k - I_t) / I_k$. The criterion: the think curve
    crosses know's line at $m^*$. The right panel shows how much of think's input that takes.
    """)
    return


@app.cell
def _(category_counts, ch, np):
    sweep = ch.ablation_sweep(category_counts, np.linspace(0, 1, 11))
    augmentation = ch.augmentation_sweep(category_counts, np.linspace(0, 8, 33))
    augmentation_crossing = ch.augmentation_crossing(category_counts)
    sweep
    return augmentation, augmentation_crossing, sweep


@app.cell
def _(augmentation_crossing, category_counts, mo):
    _added = augmentation_crossing * category_counts.loc["know", "interrogative"]
    mo.md(f"""
    Augmentation crossing: $m^* = {augmentation_crossing:.2f}$, i.e. {_added:,.0f} added think + interrogative
    uses, {_added / (category_counts.loc["think", "total"] + _added):.0%} of all think uses.
    """)
    return


@app.cell
def _(
    CACHE,
    augmentation,
    augmentation_crossing,
    category_counts,
    plt,
    sweep,
    utils,
):
    _blue, _orange, _muted = "#2a78d6", "#eb6834", "#52514e"
    _know, _think = category_counts.loc["know"], category_counts.loc["think"]
    _fig, ((_left, _right), (_aug_left, _aug_right)) = plt.subplots(2, 2, figsize=(10, 7.4))
    _left.plot(sweep.fraction, sweep.know_decl_per_interrog, color=_blue, lw=2, marker="o", ms=4, label="know, interrogatives removed")
    _left.plot(sweep.fraction, sweep.think_decl_per_interrog_cf1, color=_orange, lw=2, marker="o", ms=4, label="think, removed uses relabeled as think")
    _left.axhline(_think.declarative / _think.interrogative, color=_orange, lw=1, ls="--", label="think, observed")
    _left.set(yscale="log", xlabel="fraction f of know + interrogative uses moved", ylabel="declarative per interrogative", title="ablation")
    _left.legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1, 0.9))
    _right.plot(sweep.fraction, sweep.share_of_know_removed, color=_blue, lw=2, marker="o", ms=4)
    _right.axhline(0.5, color=_muted, lw=1, ls=":")
    _right.set(xlabel="fraction f of know + interrogative uses removed", ylabel="share of all know uses removed", ylim=(0, 1))
    _aug_left.plot(augmentation.multiple, augmentation.think_decl_per_interrog, color=_orange, lw=2, marker="o", ms=3, label="think, interrogatives added")
    _aug_left.axhline(_think.declarative / _think.interrogative, color=_orange, lw=1, ls="--", label="think, observed")
    _aug_left.axhline(_know.declarative / _know.interrogative, color=_blue, lw=1, ls="--", label="know, observed")
    _aug_left.axvline(augmentation_crossing, color=_muted, lw=1, ls=":")
    _aug_left.annotate(f"m* = {augmentation_crossing:.1f}", (augmentation_crossing, 30), xytext=(4, 0), textcoords="offset points", color=_muted, fontsize=8)
    _aug_left.set(yscale="log", xlabel="think + interrogative uses added (multiples m of know's)", ylabel="declarative per interrogative", title="augmentation")
    _aug_left.legend(frameon=False, fontsize=8, loc="center right")
    _aug_right.plot(augmentation.multiple, augmentation.share_of_think_added, color=_orange, lw=2, marker="o", ms=3)
    _aug_right.axvline(augmentation_crossing, color=_muted, lw=1, ls=":")
    _aug_right.axhline(0.5, color=_muted, lw=1, ls=":")
    _aug_right.set(xlabel="think + interrogative uses added (multiples m of know's)", ylabel="share of all think uses added", ylim=(0, 1))
    for _ax in (_left, _right, _aug_left, _aug_right):
        _ax.spines[["top", "right"]].set_visible(False)
        _ax.title.set_size(10)
    _fig.tight_layout()
    utils.save_figure(_fig, str(CACHE / "figures" / "childes_ablation_sweep.png"))
    return


@app.cell
def _(mo):
    mo.md("""
    ## Samples for manual review

    Ten random think/know utterances per lemma and category
    """)
    return


@app.cell
def _(ch, think_know, utterances):
    ch.with_context(think_know.sample(frac=1, random_state=0).groupby(["lemma", "category"]).head(10), utterances).sort_values(["lemma", "category"])[["lemma", "category", "detail", "speaker_code", "text", "context"]]
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Browse mental-verb tokens

    Every mental-verb token, with its %gra and spaCy classifications side by side. Rows where the two
    disagree are red, and rows only one parser has (e.g. %gra skipped an unparsed utterance) are
    yellow. Filters left empty match everything. A random sample of the matches is shown, with the
    context turns from the same transcript (speaker codes as in the transcript, the token's own turn
    in brackets). Select a row to see both parse trees and its full context.
    """)
    return


@app.cell
def _(np, spacy_verbs, utterances, verbs):
    _k = lambda v: v.assign(k=v.groupby(["id", "lemma"]).cumcount())[["id", "lemma", "k", "form", "category", "detail", "polar"]]
    browse = _k(verbs).merge(_k(spacy_verbs), on=["id", "lemma", "k"], how="outer", suffixes=("_gra", "_spacy"))
    browse["form"] = browse.form_gra.fillna(browse.form_spacy)
    browse["match"] = np.select(
        [browse.category_spacy.isna(), browse.category_gra.isna(), browse.category_gra == browse.category_spacy],
        ["%gra only", "spaCy only", "agree"], "disagree",
    )
    browse = browse.merge(utterances[["id", "gloss", "type", "speaker_code", "speaker_role", "corpus_name", "collection_name", "target_child_age", "transcript_id", "utterance_order"]], on="id")
    browse["age_years"] = (browse.target_child_age / 365.25).round(1)
    return (browse,)


@app.cell
def _(browse, ch, mo):
    _categories = list(ch.CATEGORIES)
    browse_lemma = mo.ui.multiselect(sorted(ch.TARGET_FORMS), value=["think", "know"], label="verb")
    browse_gra = mo.ui.multiselect(_categories, label="%gra category")
    browse_spacy = mo.ui.multiselect(_categories, label="spaCy category")
    browse_match = mo.ui.dropdown(["any", "disagree", "agree", "%gra only", "spaCy only"], value="any", label="parsers")
    browse_polar = mo.ui.dropdown(["any", "polar", "not polar"], value="any", label="polar question (%gra)")
    browse_role = mo.ui.multiselect(browse.speaker_role.value_counts().index.tolist(), label="speaker role")
    browse_corpus = mo.ui.multiselect(sorted(browse.corpus_name.dropna().unique()), label="corpus")
    browse_collection = mo.ui.multiselect(sorted(browse.collection_name.dropna().unique()), label="collection")
    browse_age = mo.ui.range_slider(0, 12, step=0.5, value=[0, 12], label="target-child age (years; 12 = 12+)", show_value=True)
    browse_search = mo.ui.text(placeholder="regex, e.g. ^do you", label="utterance matches")
    browse_window = mo.ui.slider(0, 10, value=2, label="context turns either side", show_value=True)
    browse_n = mo.ui.slider(10, 500, step=10, value=100, label="rows shown", show_value=True)
    browse_seed = mo.ui.number(0, 1_000_000, value=0, label="sample seed")
    mo.vstack([
        mo.hstack([browse_lemma, browse_gra, browse_spacy, browse_match, browse_polar], wrap=True, justify="start"),
        mo.hstack([browse_role, browse_corpus, browse_collection, browse_age], wrap=True, justify="start"),
        mo.hstack([browse_search, browse_window, browse_n, browse_seed], wrap=True, justify="start"),
    ])
    return (
        browse_age,
        browse_collection,
        browse_corpus,
        browse_gra,
        browse_lemma,
        browse_match,
        browse_n,
        browse_polar,
        browse_role,
        browse_search,
        browse_seed,
        browse_spacy,
        browse_window,
    )


@app.cell
def _(
    browse,
    browse_age,
    browse_collection,
    browse_corpus,
    browse_gra,
    browse_lemma,
    browse_match,
    browse_n,
    browse_polar,
    browse_role,
    browse_search,
    browse_seed,
    browse_spacy,
    browse_window,
    ch,
    mo,
    pd,
    re,
    utterances,
):
    _keep = pd.Series(True, index=browse.index)
    for _column, _ui in [("lemma", browse_lemma), ("category_gra", browse_gra), ("category_spacy", browse_spacy),
                         ("speaker_role", browse_role), ("corpus_name", browse_corpus), ("collection_name", browse_collection)]:
        if _ui.value:
            _keep &= browse[_column].isin(_ui.value)
    if browse_match.value != "any":
        _keep &= browse.match == browse_match.value
    if browse_polar.value != "any":
        _keep &= browse.polar_gra.eq(browse_polar.value == "polar")
    _low, _high = browse_age.value
    if (_low, _high) != (0, 12):
        _keep &= (browse.age_years >= _low) & ((browse.age_years <= _high) | (_high == 12))
    _error = None
    if browse_search.value:
        try:
            _keep &= browse.gloss.str.contains(browse_search.value, flags=re.IGNORECASE, regex=True, na=False)
        except re.error as e:
            _error = f"bad regex: {e}"
    _matches = browse[_keep]
    browse_shown = ch.with_context(
        _matches.sample(min(browse_n.value, len(_matches)), random_state=int(browse_seed.value)).reset_index(drop=True),
        utterances, window=browse_window.value,
    )
    _columns = ["lemma", "form", "category_gra", "detail_gra", "category_spacy", "detail_spacy", "match", "polar_gra",
                "speaker_code", "speaker_role", "corpus_name", "age_years", "context"]


    def _style(row_id, column, value):
        match = browse_shown.match.iloc[int(row_id)]
        if match == "disagree":
            return {"backgroundColor": "rgba(227, 73, 72, 0.16)", "fontWeight": "bold" if column.startswith("category") else "normal"}
        if match != "agree":
            return {"backgroundColor": "rgba(237, 161, 0, 0.16)"}
        return {}


    browse_table = mo.ui.table(
        browse_shown[_columns], selection="single", style_cell=_style, wrapped_columns=["context"],
        page_size=20, show_column_summaries=False,
        label=f"{len(_matches):,} matching tokens ({(_matches.match == 'disagree').mean():.0%} %gra/spaCy disagreements); showing {len(browse_shown)}",
    )
    mo.vstack([mo.md(f"**{_error}**") if _error else mo.md(""), browse_table])
    return browse_shown, browse_table


@app.cell
def _(
    browse_shown,
    browse_table,
    browse_window,
    compare_parses,
    mo,
    re,
    utterances,
):
    if browse_table.value is None or len(browse_table.value) == 0:
        _out = mo.md("*Select a row above to see its parses and context.*")
    else:
        _row = browse_shown.loc[browse_table.value.index[0]]
        _window = max(browse_window.value, 3)
        _turns = utterances[(utterances.transcript_id == _row.transcript_id)
                            & utterances.utterance_order.between(_row.utterance_order - _window, _row.utterance_order + _window)].sort_values("utterance_order")
        _lines = []
        for _t in _turns.itertuples():
            _text = f"{_t.speaker_code}: {_t.gloss}"
            if _t.id == _row.id:
                _text = "**" + re.sub(rf"(?<!\S){re.escape(_row.form)}(?!\S)", lambda m: f"<mark>{m.group(0)}</mark>", _text) + "**"
            _lines.append(f"- {_text}")
        _age = "unknown" if _row.age_years != _row.age_years else f"{_row.age_years} years"
        _out = mo.vstack([
            mo.md(
                f"**{_row.lemma}** · %gra `{_row.category_gra}/{_row.detail_gra}` · spaCy `{_row.category_spacy}/{_row.detail_spacy}` · {_row.match}\n\n"
                f"{_row.corpus_name} ({_row.collection_name}) · {_row.speaker_code} ({_row.speaker_role}) · target child {_age} · "
                f"{_row.type} · utterance {_row.id}\n\n" + "\n".join(_lines)
            ),
            compare_parses(int(_row.id)),
        ])
    _out
    return


if __name__ == "__main__":
    app.run()
