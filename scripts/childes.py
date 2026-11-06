"""Eng-NA/Eng-UK CHILDES (childes-db 2026.1 on Redivis): loading, vocabulary
frequency and same-tag alternatives, and spaCy classification of think/know complements."""

import collections
import itertools
import os
import re

import numpy as np
import pandas as pd

REDIVIS_OWNER, REDIVIS_DATASET = "datapages", "childes_db:b6q6:v1_4"  # childes-db 2026.1
COLLECTIONS = ("Eng-NA", "Eng-UK")
TARGET_FORMS = {
    "think": ("think", "thinks", "thinking", "thought"),
    "know": ("know", "knows", "knew", "known", "knowing"),
}
WH_WORDS = frozenset({"who", "whom", "whose", "what", "which", "where", "when", "why", "how"})
WHETHER_WORDS = frozenset({"whether", "if"})
# CHILDES codes for unintelligible (xxx), unphonologized (yyy) and untranscribed (www) speech
PLACEHOLDERS = frozenset({"xxx", "yyy", "www"})
# benchmark embedded_type names, plus CHILDES-only categories
# %mor features that depend on the utterance rather than the word: case, subject agreement, mood
CONTEXT_FEATURES = frozenset({"Nom", "Acc", "Gen", "Dat", "S", "S1", "S2", "S3", "P", "P1", "P2", "P3", "Ind", "irr"})
CATEGORIES = ("declarative", "interrogative", "NP", "PP", "so", "infinitival", "matrix_wh", "left_clause", "none")


def _collection_filter():
    return "collection_name IN (" + ", ".join(f"'{c}'" for c in COLLECTIONS) + ")"


def query_cached(sql, path):
    """Run `sql` against childes-db on Redivis once and cache the result as parquet at `path`."""
    if not os.path.exists(path):
        import pyarrow.parquet as pq
        import redivis

        table = redivis.organization(REDIVIS_OWNER).dataset(REDIVIS_DATASET).query(sql).to_arrow_table()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pq.write_table(table, path)
    return pd.read_parquet(path)


def load_utterances(cache_dir):
    """Every Eng-NA/Eng-UK utterance, one row each: gloss plus speaker, position in its
    transcript, age, and corpus metadata."""
    sql = (
        "SELECT id, gloss, type, speaker_code, speaker_role, utterance_order, target_child_age, "
        f"collection_name, corpus_name, transcript_id, num_tokens FROM utterance WHERE {_collection_filter()}"
    )
    return query_cached(sql, os.path.join(cache_dir, "childes_eng_utterances.parquet"))


def load_token_tags(cache_dir):
    """(gloss, part_of_speech, suffix, count) over Eng-NA/Eng-UK tokens, from the CHILDES %mor tier."""
    sql = (
        "SELECT LOWER(gloss) AS gloss, part_of_speech, suffix, COUNT(*) AS count FROM token "
        f"WHERE {_collection_filter()} GROUP BY 1, 2, 3"
    )
    return query_cached(sql, os.path.join(cache_dir, "childes_eng_token_tags.parquet"))


def load_verb_stems(cache_dir):
    """(stem, count) over Eng-NA/Eng-UK tokens tagged verb on the %mor tier, so inflected forms
    count toward their stem (knows/knew -> know)."""
    sql = (
        "SELECT LOWER(stem) AS stem, COUNT(*) AS count FROM token "
        f"WHERE part_of_speech = 'verb' AND {_collection_filter()} GROUP BY 1"
    )
    return query_cached(sql, os.path.join(cache_dir, "childes_eng_verb_stems.parquet"))


def load_target_tokens(cache_dir):
    """Every token, with its %gra index, head and relation, of the Eng-NA/Eng-UK utterances
    that have a think/know token tagged verb on the %mor tier."""
    target = f"SELECT utterance_id FROM token WHERE stem IN ('think', 'know') AND part_of_speech = 'verb' AND {_collection_filter()}"
    sql = (
        "SELECT utterance_id, token_order, LOWER(gloss) AS gloss, LOWER(stem) AS stem, part_of_speech, "
        f"gra_index, gra_head, gra_relation FROM token WHERE {_collection_filter()} AND utterance_id IN ({target})"
    )
    return query_cached(sql, os.path.join(cache_dir, "childes_think_know_tokens.parquet"))


def load_lemma_counts(cache_dir):
    """(lemma, count) over every Eng-NA/Eng-UK morpheme on the token_morpheme tier, stems and
    clitics alike (so "it's" counts toward both it and be)."""
    sql = f"SELECT LOWER(lemma) AS lemma, COUNT(*) AS count FROM token_morpheme WHERE {_collection_filter()} GROUP BY 1"
    return query_cached(sql, os.path.join(cache_dir, "childes_eng_lemma_counts.parquet"))


def load_gloss_lemmas(cache_dir):
    """(gloss, lemma, count): how often each token gloss has each lemma on its stem morpheme."""
    sql = (
        "SELECT LOWER(t.gloss) AS gloss, LOWER(m.lemma) AS lemma, COUNT(*) AS count FROM token t "
        "JOIN token_morpheme m ON m.token_id = t.id AND m.type = 'stem' "
        f"WHERE {_collection_filter().replace('collection_name', 't.collection_name')} GROUP BY 1, 2"
    )
    return query_cached(sql, os.path.join(cache_dir, "childes_eng_gloss_lemmas.parquet"))


def tokens(gloss):
    """Lowercased whitespace tokens of a CHILDES gloss, without xxx/yyy/www."""
    return [t for t in gloss.lower().split() if t not in PLACEHOLDERS]


def word_counts(glosses):
    """Counter word -> occurrences over an iterable of glosses."""
    return collections.Counter(itertools.chain.from_iterable(tokens(g) for g in glosses))


def phrase_counts(glosses, phrases):
    """{phrase: whole-word occurrences across glosses} for multiword phrases (lowercased)."""
    text = pd.Series([" ".join(tokens(g)) for g in glosses])
    return {p: int(text.str.count(rf"(?<!\S){re.escape(p)}(?!\S)").sum()) for p in phrases}



def word_tags(token_tags):
    """word -> its most frequent tag: part of speech plus its word-level features, without
    CONTEXT_FEATURES (e.g. 'keys' -> 'noun-Plur', 'stashed' -> 'verb-Fin Past')."""
    tagged = token_tags.dropna(subset=["part_of_speech"]).query("part_of_speech != ''")
    features = [" ".join(f for f in str(x or "").split() if f not in CONTEXT_FEATURES) for x in tagged.suffix]
    tag = tagged.part_of_speech + pd.Series([f"-{f}" if f else "" for f in features], index=tagged.index)
    best = tagged.assign(tag=tag).groupby(["gloss", "tag"])["count"].sum().reset_index()
    best = best.sort_values("count", ascending=False).drop_duplicates("gloss")
    return dict(zip(best.gloss, best.tag))


def _unit(vectors):
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def alternatives(words, tags, counts, nlp, threshold, k=8):
    """{word: k most vector-similar CHILDES words with the same %mor tag and count >= threshold}.
    An unattested word takes the tag of its most similar attested word."""
    pool = [w for w, c in counts.items() if c >= threshold and w in tags and nlp.vocab[w].vector_norm > 0]
    pool_vectors = _unit(np.stack([nlp.vocab[w].vector for w in pool]))
    pool_tags = np.array([tags[w] for w in pool])
    result = {}
    for word in words:
        sims = pool_vectors @ _unit(nlp.vocab[word].vector[None])[0]
        tag = tags[word] if word in tags else pool_tags[np.argmax(sims)]
        order = [i for i in np.argsort(-sims) if pool_tags[i] == tag and pool[i] != word][:k]
        result[word] = (tag, [pool[i] for i in order])
    return result


def parser_input(gloss, utterance_type):
    """CHILDES gloss -> spaCy input: drop xxx/yyy/www, capitalize 'i', and end with '?' for
    question-type utterances and '.' otherwise (CHILDES glosses carry no punctuation)."""
    words = [("I" + t[1:]) if t == "i" or t.startswith("i'") else t for t in tokens(gloss)]
    return " ".join(words) + (" ?" if "question" in str(utterance_type) else " .")


def _in_inner_clause(token, clause):
    """True if `token` sits in an adverbial or subject clause inside `clause`, as 'if' does in
    'think [[if you push it] it will go]' and 'what' in 'think [[what we should do] is play]'."""
    for node in itertools.chain([token], token.ancestors):
        if node == clause:
            return False
        if node.dep_ in ("advcl", "csubj"):
            return True
    raise ValueError(f"{token!r} is not inside {clause!r}")


def classify_complement(verb):
    """(category, detail) for a think/know verb token's complement; category is in CATEGORIES.
    Only dependents whose span lies right of the verb count as complements. Without one, a
    wh-word left of the verb in its subtree marks a matrix wh-question (what do you think?),
    or a left_clause if it sits in a clause attached leftward (where is it I don't know)."""
    right = [c for c in verb.children if c.i > verb.i and c.left_edge.i > verb.i]
    for dep in ("ccomp", "xcomp", "advcl"):
        for clause in (c for c in right if c.dep_ == dep):
            edge = clause.left_edge
            if edge.lower_ in WHETHER_WORDS | WH_WORDS and _in_inner_clause(edge, clause):
                return "declarative", "adjunct_first"
            if edge.lower_ in WHETHER_WORDS:
                return "interrogative", edge.lower_
            if edge.lower_ in WH_WORDS:
                return "interrogative", "wh_infinitive" if dep == "xcomp" else "wh_clause"
            if dep == "ccomp":
                return "declarative", "that" if edge.lower_ == "that" and edge.dep_ == "mark" else "zero"
            if dep == "xcomp":
                return "infinitival", "to"
    for child in right:
        if child.lower_ == "so" and child.dep_ in ("advmod", "dobj", "obj", "ccomp"):
            return "so", "so"
        if child.dep_ in ("dobj", "obj"):
            if child.lower_ in WH_WORDS:
                return "interrogative", "wh_object" if any(True for _ in child.children) else "wh_bare"
            return "NP", "pronoun" if child.pos_ == "PRON" else "noun"
        if child.dep_ == "prep":
            return "PP", child.lower_
        # UD (%gra) attaches the noun as obl, with the preposition as its case dependent
        case = next((c for c in child.children if c.dep_ == "case"), None) if child.dep_ == "obl" else None
        if case is not None:
            return "PP", case.lower_
    left = [c for c in verb.children if c.i < verb.i and c.dep_ in ("ccomp", "parataxis", "advcl")]
    left_wh = next((t for t in verb.subtree if t.i < verb.i and t.lower_ in WH_WORDS), None)
    if left_wh is not None and not any(left_wh in c.subtree for c in left):
        return "matrix_wh", left_wh.lower_
    if left:
        return "left_clause", "wh" if left_wh is not None else "clause"
    return "none", verb.dep_


def verb_features(verb):
    """Subject and negation of a verb token: (subject lowercased or '', negated)."""
    subject = next((c.lower_ for c in verb.children if c.dep_ in ("nsubj", "nsubjpass")), "")
    # %gra keeps n't on its auxiliary ('don't' is one aux token) and labels not/never advmod
    return subject, any(c.dep_ == "neg" or c.lower_ in ("not", "never") or c.lower_.endswith("n't") for c in verb.children)


class GraToken:
    """One %gra token, with the attributes of a spaCy token that classify_complement and
    verb_features read. dep_ is the UD relation lowercased, without its subtype
    (NSUBJ-PASS -> nsubj); a head that is a clitic with no token row of its own leaves head None."""

    def __init__(self, i, lower_, lemma_, pos_, dep_):
        self.i, self.lower_, self.lemma_, self.pos_, self.dep_ = i, lower_, lemma_, pos_, dep_
        self.head, self.children = None, []

    def __repr__(self):
        return f"GraToken({self.i}, {self.lower_!r}, {self.dep_!r})"

    @property
    def subtree(self):
        yield self
        for child in self.children:
            yield from child.subtree

    @property
    def left_edge(self):
        return min(self.subtree, key=lambda t: t.i)

    @property
    def ancestors(self):
        node, seen = self.head, {self.i}
        while node is not None:
            if node.i in seen:
                raise ValueError(f"cycle in %gra parse at {node!r}")
            seen.add(node.i)
            yield node
            node = node.head


def gra_tree(rows):
    """{gra_index: GraToken} for one utterance's token rows (load_target_tokens columns)."""
    nodes = {int(r.gra_index): GraToken(int(r.gra_index), r.gloss, r.stem, str(r.part_of_speech).upper(), r.gra_relation.lower().split("-")[0])
             for r in rows.itertuples()}
    for r in rows.itertuples():
        node, head = nodes[int(r.gra_index)], nodes.get(int(r.gra_head))
        if head is not None and head is not node:
            node.head = head
            head.children.append(node)
    for node in nodes.values():
        node.children.sort(key=lambda t: t.i)
    return nodes


def classify_gra(tokens):
    """One row per think/know verb (%mor stem, tagged verb) in each parsed utterance of `tokens`
    (load_target_tokens): utterance id, lemma, form, category, detail, subject, negated.
    Utterances with any token missing a %gra annotation are skipped."""
    parsed = tokens.groupby("utterance_id").filter(lambda g: g.gra_index.notna().all() and g.gra_relation.notna().all())
    rows = []
    for utterance_id, group in parsed.groupby("utterance_id"):
        nodes = gra_tree(group)
        for r in group[group.stem.isin(TARGET_FORMS) & (group.part_of_speech == "verb")].itertuples():
            verb = nodes[int(r.gra_index)]
            category, detail = classify_complement(verb)
            subject, negated = verb_features(verb)
            rows.append((utterance_id, r.stem, r.gloss, category, detail, subject, negated))
    return pd.DataFrame(rows, columns=["id", "lemma", "form", "category", "detail", "subject", "negated"])


def classify_gra_utterances(tokens, utterances, path):
    """classify_gra joined with utterance metadata (the gloss as `text`); cached at `path`."""
    if os.path.exists(path):
        return pd.read_parquet(path)
    result = classify_gra(tokens).merge(utterances.assign(text=utterances.gloss), on="id", how="left")
    result.to_parquet(path)
    return result


def target_verbs(doc):
    """think/know tokens tagged as verbs in a spaCy doc."""
    return [t for t in doc if t.lemma_ in TARGET_FORMS and t.pos_ == "VERB"]


def classify_texts(texts, nlp, batch_size=2000):
    """One row per think/know verb in each text: text index, lemma, form, category, detail,
    subject, negated. Also returns the count of target forms spaCy did not tag as verbs."""
    rows, non_verbs = [], collections.Counter()
    forms = {f: lemma for lemma, fs in TARGET_FORMS.items() for f in fs}
    for index, doc in enumerate(nlp.pipe(texts, batch_size=batch_size)):
        for t in doc:
            if t.lower_ in forms and not (t.lemma_ in TARGET_FORMS and t.pos_ == "VERB"):
                non_verbs[(t.lower_, t.pos_)] += 1
        for verb in target_verbs(doc):
            category, detail = classify_complement(verb)
            subject, negated = verb_features(verb)
            rows.append((index, verb.lemma_, verb.lower_, category, detail, subject, negated))
    columns = ["text_index", "lemma", "form", "category", "detail", "subject", "negated"]
    return pd.DataFrame(rows, columns=columns), non_verbs


def target_utterances(utterances):
    """Utterances containing any think/know form as a whole word."""
    forms = "|".join(f for fs in TARGET_FORMS.values() for f in fs)
    return utterances[utterances.gloss.str.lower().str.contains(rf"(?<!\S)(?:{forms})(?!\S)", regex=True)]


def classify_utterances(utterances, nlp, path):
    """classify_texts over the think/know utterances, joined with utterance metadata; cached at `path`."""
    if os.path.exists(path):
        return pd.read_parquet(path)
    hits = target_utterances(utterances).reset_index(drop=True)
    texts = [parser_input(g, t) for g, t in zip(hits.gloss, hits.type)]
    verbs, _ = classify_texts(texts, nlp)
    result = verbs.join(hits.assign(text=texts), on="text_index").drop(columns="text_index")
    result.to_parquet(path)
    return result


def with_context(rows, utterances, window=1):
    """`rows` (with an utterance `id`) plus its speaker_code and `context`: the utterance and up to
    `window` turns either side in its transcript, as 'CODE: gloss' joined by ' | ', with the
    row's own turn in [brackets]."""
    position = rows[["id"]].drop_duplicates().merge(utterances[["id", "transcript_id", "utterance_order", "speaker_code"]], on="id", how="left")
    offsets = np.arange(-window, window + 1)
    around = position.loc[position.index.repeat(len(offsets))].assign(offset=np.tile(offsets, len(position)))
    around["utterance_order"] += around.offset
    turns = around.merge(utterances[["transcript_id", "utterance_order", "speaker_code", "gloss"]], on=["transcript_id", "utterance_order"], suffixes=("_row", ""))
    turns["line"] = turns.speaker_code + ": " + turns.gloss
    turns.loc[turns.offset == 0, "line"] = "[" + turns.line + "]"
    context = turns.sort_values(["id", "offset"]).groupby("id").line.agg(" | ".join)
    return rows.assign(speaker_code=rows.id.map(position.set_index("id").speaker_code), context=rows.id.map(context))


def category_table(verbs, by=()):
    """Counts and within-group proportions of each complement category per lemma (and `by` columns)."""
    keys = ["lemma", *by]
    counts = verbs.groupby([*keys, "category"]).size().unstack("category", fill_value=0)
    counts = counts.reindex(columns=[c for c in CATEGORIES if c in counts.columns])
    counts["total"] = counts.sum(axis=1)
    counts["p_decl_of_clausal"] = counts.declarative / (counts.declarative + counts.interrogative)
    counts["p_interrog_of_clausal"] = counts.interrogative / (counts.declarative + counts.interrogative)
    counts["p_other"] = 1 - (counts.declarative + counts.interrogative) / counts.total
    counts["p_none"] = counts.none / counts.total
    return counts


def bootstrap_p_interrog(verbs, n_boot=2000, seed=0):
    """Per lemma: P(interrogative | declarative or interrogative) with a 95% CI from
    resampling transcripts (utterances within a transcript are not independent)."""
    rng = np.random.default_rng(seed)
    rows = []
    for lemma, group in verbs[verbs.category.isin(["declarative", "interrogative"])].groupby("lemma"):
        per = group.groupby("transcript_id").category.agg(
            interrog=lambda c: (c == "interrogative").sum(), n="size"
        )
        draws = rng.integers(0, len(per), size=(n_boot, len(per)))
        boot = per.interrog.to_numpy()[draws].sum(1) / per.n.to_numpy()[draws].sum(1)
        rows.append((lemma, per.interrog.sum() / per.n.sum(), *np.quantile(boot, [0.025, 0.975]), len(per)))
    return pd.DataFrame(rows, columns=["lemma", "p_interrog", "ci_low", "ci_high", "n_transcripts"]).set_index("lemma")


def ablation_sweep(table, fractions):
    """Remove (ablation) or relabel as think (counterfactual 1) a fraction f of know+interrogative
    uses. `table` is category_table(verbs) indexed by lemma. Rows per f: know's and think's
    resulting declarative-per-interrogative ratio (inf for know at f = 1), and the share of all
    know uses removed."""
    know, think = table.loc["know"], table.loc["think"]
    moved = pd.Series(fractions) * know.interrogative
    return pd.DataFrame({
        "fraction": fractions,
        "know_interrog_left": know.interrogative - moved,
        "know_decl_per_interrog": know.declarative / (know.interrogative - moved),
        "share_of_know_removed": moved / know.total,
        "think_decl_per_interrog_cf1": think.declarative / (think.interrogative + moved),
    })


def augmentation_sweep(table, multiples):
    """Add m * (know's interrogative count) think+interrogative uses, know unchanged (counterfactual 2).
    Rows per m: think's resulting declarative-per-interrogative ratio and the share of all think
    uses that are added."""
    know, think = table.loc["know"], table.loc["think"]
    added = pd.Series(multiples) * know.interrogative
    return pd.DataFrame({
        "multiple": multiples,
        "think_interrog_added": added,
        "think_decl_per_interrog": think.declarative / (think.interrogative + added),
        "share_of_think_added": added / (think.total + added),
    })


def augmentation_crossing(table):
    """The m at which augmented think's declarative-per-interrogative ratio falls to know's observed one:
    D_t / (I_t + m I_k) = D_k / I_k, so m = (D_t I_k / D_k - I_t) / I_k."""
    know, think = table.loc["know"], table.loc["think"]
    return (think.declarative * know.interrogative / know.declarative - think.interrogative) / know.interrogative
