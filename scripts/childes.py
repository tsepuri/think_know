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
# mental-state verbs, each with the complements it is usually described as taking: responsive (declarative and interrogative), rogative (interrogative only),
# anti-rogative (declarative only); `mean` is left unclassed
VERB_CLASSES = {
    "know": "responsive", "remember": "responsive", "forget": "responsive", "guess": "responsive",
    "understand": "responsive", "learn": "responsive", "notice": "responsive", "figure": "responsive",
    "decide": "responsive", "realize": "responsive",
    "wonder": "rogative",
    "think": "anti-rogative", "believe": "anti-rogative", "suppose": "anti-rogative", "hope": "anti-rogative",
    "wish": "anti-rogative", "imagine": "anti-rogative", "pretend": "anti-rogative", "bet": "anti-rogative",
    "expect": "anti-rogative", "dream": "anti-rogative", "reckon": "anti-rogative", "doubt": "anti-rogative",
    "mean": "other",
}
# %mor stem -> surface forms, for matching glosses
TARGET_FORMS = {
    "know": ("know", "knows", "knew", "known", "knowing"),
    "remember": ("remember", "remembers", "remembered", "remembering"),
    "forget": ("forget", "forgets", "forgot", "forgotten", "forgetting"),
    "guess": ("guess", "guesses", "guessed", "guessing"),
    "understand": ("understand", "understands", "understood", "understanding"),
    "learn": ("learn", "learns", "learned", "learnt", "learning"),
    "notice": ("notice", "notices", "noticed", "noticing"),
    "figure": ("figure", "figures", "figured", "figuring"),
    "decide": ("decide", "decides", "decided", "deciding"),
    "realize": ("realize", "realizes", "realized", "realizing"),
    "wonder": ("wonder", "wonders", "wondered", "wondering"),
    "think": ("think", "thinks", "thinking", "thought"),
    "believe": ("believe", "believes", "believed", "believing"),
    "suppose": ("suppose", "supposes", "supposed", "supposing"),
    "hope": ("hope", "hopes", "hoped", "hoping"),
    "wish": ("wish", "wishes", "wished", "wishing"),
    "imagine": ("imagine", "imagines", "imagined", "imagining"),
    "pretend": ("pretend", "pretends", "pretended", "pretending"),
    "bet": ("bet", "bets", "betting"),
    "expect": ("expect", "expects", "expected", "expecting"),
    "dream": ("dream", "dreams", "dreamed", "dreamt", "dreaming"),
    "reckon": ("reckon", "reckons", "reckoned", "reckoning"),
    "doubt": ("doubt", "doubts", "doubted", "doubting"),
    "mean": ("mean", "means", "meant", "meaning"),
}
WH_WORDS = frozenset({"who", "whom", "whose", "what", "which", "where", "when", "why", "how"})
WHETHER_WORDS = frozenset({"whether", "if"})
# CHILDES codes for unintelligible (xxx), unphonologized (yyy) and untranscribed (www) speech
PLACEHOLDERS = frozenset({"xxx", "yyy", "www"})
# benchmark embedded_type names, plus CHILDES-only categories
# %mor features that depend on the utterance rather than the word: case, subject agreement, mood
# TODO: add explanation here of what each of these mean
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
    that have a TARGET_FORMS stem tagged verb on the %mor tier."""
    stems = ", ".join(f"'{s}'" for s in TARGET_FORMS)
    target = f"SELECT utterance_id FROM token WHERE LOWER(stem) IN ({stems}) AND part_of_speech = 'verb' AND {_collection_filter()}"
    sql = (
        "SELECT utterance_id, token_order, LOWER(gloss) AS gloss, LOWER(stem) AS stem, part_of_speech, "
        f"gra_index, gra_head, gra_relation FROM token WHERE {_collection_filter()} AND utterance_id IN ({target})"
    )
    return query_cached(sql, os.path.join(cache_dir, "childes_mental_verb_tokens.parquet"))


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
    """{phrase: whole-word occurrences across glosses} for multiword phrases (lowercased), from one
    pass over each gloss's n-grams of the phrases' lengths."""
    wanted = {tuple(p.split()) for p in phrases}
    sizes = {len(p) for p in wanted}
    found = collections.Counter()
    for gloss in glosses:
        words = tokens(gloss)
        for n in sizes:
            for i in range(len(words) - n + 1):
                if tuple(words[i:i + n]) in wanted:
                    found[" ".join(words[i:i + n])] += 1
    return {p: found[p] for p in phrases}



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


# dependents passed over when finding the word that opens a clause: fillers, conjunctions,
# vocatives, and the clause head's own non-wh adverbs (know [exactly what...], think [maybe you...])
_OPENER_SKIP = frozenset({"discourse", "intj", "reparandum", "cc", "punct", "vocative"})


def _word(token):
    """`token` lowercased, without a clitic: %gra keeps what's and that's whole."""
    return re.sub(r"'(s|re|ll|ve|d|m)$", "", token.lower_)


def _opener(clause):
    """The word that opens `clause`: its first word, past any _OPENER_SKIP dependent."""
    words = sorted(clause.subtree, key=lambda t: t.i)
    skip = lambda t: t.dep_ in _OPENER_SKIP or (t.dep_ == "advmod" and t.head == clause and t.lower_ not in WH_WORDS)
    return next((t for t in words if not skip(t)), words[0])


def _matrix_wh(verb):
    """The first wh-word left of `verb` in its subtree, unless it sits in a clause attached
    leftward (where is it [I don't know]); None if there is none."""
    left = [c for c in verb.children if c.i < verb.i and c.dep_ in ("ccomp", "parataxis", "advcl")]
    wh = next((t for t in verb.subtree if t.i < verb.i and _word(t) in WH_WORDS), None)
    return None if wh is None or any(wh in c.subtree for c in left) else wh


def is_matrix_polar(verb):
    """True if `verb` heads a polar (yes/no) question: an auxiliary precedes its subject
    (do you know, can't you think) and no matrix wh-word precedes it (what do you know)."""
    subjects = [c.i for c in verb.children if c.dep_.startswith("nsubj")]
    inverted = bool(subjects) and any(c.dep_.startswith("aux") and c.i < min(subjects) for c in verb.children)
    return inverted and _matrix_wh(verb) is None


def classify_complement(verb):
    """(category, detail) for a mental verb token's complement; category is in CATEGORIES.
    Only dependents whose span lies right of the verb count as complements. A clause is
    interrogative only if a wh-word or whether/if opens it: a wh-word deeper inside (think
    [you know what this is]) or inside an object's relative clause (know [a lady who...])
    doesn't count, and an if-clause that is an adverbial, or is followed by the clause it
    conditions (think [if you said it] [would he know]), is a conditional."""
    right = [c for c in verb.children if c.i > verb.i and c.left_edge.i > verb.i]
    for dep in ("ccomp", "xcomp", "advcl"):
        for clause in (c for c in right if c.dep_ == dep):
            opener = _opener(clause)
            word = _word(opener)
            if word in WHETHER_WORDS | WH_WORDS and _in_inner_clause(opener, clause):
                return "declarative", "adjunct_first"
            if word == "if" and dep == "advcl":
                continue
            if word == "if" and any(c.i > clause.i and c.dep_ in ("ccomp", "parataxis", "conj") for c in right):
                return "declarative", "adjunct_first"
            if word in WHETHER_WORDS:
                return "interrogative", word
            # a wh-word opens the clause, or is %gra's object of the verb just before it (know [what] [is])
            if word in WH_WORDS or any(c.dep_ in ("dobj", "obj", "dep") and _word(c) in WH_WORDS and c.i < clause.i for c in right):
                return "interrogative", "wh_infinitive" if dep == "xcomp" else "wh_clause"
            if dep == "ccomp":
                return "declarative", "that" if word == "that" and opener.dep_ == "mark" else "zero"
            if dep == "xcomp":
                return "infinitival", "to"
    for child in right:
        if child.lower_ == "so" and child.dep_ in ("advmod", "dobj", "obj", "ccomp"):
            return "so", "so"
        if child.dep_ in ("dobj", "obj"):
            if _word(child) in WH_WORDS:
                return "interrogative", "wh_object" if any(True for _ in child.children) else "wh_bare"
            # a wh-determiner opening the object (know [which one], [how many cars]), not an
            # exclamative (think [what a pretty girl])
            edge = child.left_edge
            if _word(edge) in WH_WORDS and child in list(edge.ancestors)[:2] and not any(c.lower_ in ("a", "an") for c in child.children):
                return "interrogative", "wh_noun_phrase"
            return "NP", "pronoun" if child.pos_ == "PRON" else "noun"
        # %gra attaches a sluiced wh-adverb as advmod (I don't know [why])
        if child.dep_ == "advmod" and _word(child) in WH_WORDS and not child.children:
            return "interrogative", "wh_bare"
        if child.dep_ == "prep":
            return "PP", child.lower_
        # UD (%gra) attaches the noun as obl, with the preposition as its case dependent
        case = next((c for c in child.children if c.dep_ == "case"), None) if child.dep_ == "obl" else None
        if case is not None:
            return "PP", case.lower_
    wh = _matrix_wh(verb)
    if wh is not None:
        return "matrix_wh", _word(wh)
    left = [c for c in verb.children if c.i < verb.i and c.dep_ in ("ccomp", "parataxis", "advcl")]
    if left:
        left_wh = any(t.i < verb.i and _word(t) in WH_WORDS for t in verb.subtree)
        return "left_clause", "wh" if left_wh else "clause"
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


def _gra_nodes(indices, glosses, stems, tags, relations, heads):
    """{gra_index: GraToken} from one utterance's parallel %gra columns. CHILDES gives a clitic
    (the 's of there's) a %gra index of its own but no token row, so a dependent whose head is a
    clitic would be cut off with it ('think [there's something]' would lose its complement).
    Each such clitic becomes a GraToken (relation 'clitic') attached as ccomp to the nearest
    verb before its host, when every word in between belongs to the clitic's subtree (so a
    relative clause stays put: 'know the guy [that's here]'); advcl instead if a subordinator
    other than that/if/whether marks it ('think [because it's raining]')."""
    nodes = {i: GraToken(i, g, s, str(t).upper(), r.lower().split("-")[0]) for i, g, s, t, r in zip(indices, glosses, stems, tags, relations)}
    for i, h in zip(indices, heads):
        if h != 0 and h not in nodes:
            nodes[h] = GraToken(h, "'" + nodes[h - 1].lower_.rsplit("'", 1)[-1] if h - 1 in nodes else "", "", "AUX", "clitic")
    for i, h in zip(indices, heads):
        node, head = nodes[i], nodes.get(h)
        if head is not None and head is not node:
            node.head = head
            head.children.append(node)
    for node in nodes.values():
        node.children.sort(key=lambda t: t.i)
    for clitic in [n for n in nodes.values() if n.dep_ == "clitic"]:
        host = nodes.get(clitic.i - 1)
        verb = next((nodes[j] for j in range(clitic.i - 2, 0, -1) if j in nodes and nodes[j].pos_ == "VERB"), None)
        if host is None or verb is None:
            continue
        span = {t.i for t in clitic.subtree}
        if verb.i not in span and all(j in span or j not in nodes for j in range(verb.i + 1, host.i)):
            marks = {c.lower_ for c in clitic.children if c.dep_ == "mark"}
            clitic.dep_ = "advcl" if marks - {"that", "if", "whether"} else "ccomp"
            clitic.head = verb
            verb.children = sorted(verb.children + [clitic], key=lambda t: t.i)
    return nodes


def gra_tree(rows):
    """{gra_index: GraToken} for one utterance's token rows (load_target_tokens columns)."""
    return _gra_nodes(rows.gra_index.astype(int).tolist(), rows.gloss.tolist(), rows.stem.tolist(),
                      rows.part_of_speech.tolist(), rows.gra_relation.tolist(), rows.gra_head.astype(int).tolist())


COLUMNS = ["id", "lemma", "form", "category", "detail", "subject", "negated", "polar"]


def _verb_row(verb):
    """category, detail, subject, negated, polar for one verb token."""
    return (*classify_complement(verb), *verb_features(verb), is_matrix_polar(verb))


def classify_gra(tokens):
    """One row per mental verb (a TARGET_FORMS stem on the %mor tier, i.e. the dictionary form
    CHILDES lemmatizes the token to, tagged verb) in each parsed utterance of `tokens`
    (load_target_tokens): COLUMNS. xxx/yyy/www tokens carry no %gra and are dropped; utterances
    with any other token missing a %gra annotation are skipped."""
    tokens = tokens[~(tokens.gloss.isin(PLACEHOLDERS) & tokens.gra_index.isna())]
    unparsed = tokens.loc[tokens.gra_index.isna() | tokens.gra_relation.isna(), "utterance_id"].unique()
    parsed = tokens[~tokens.utterance_id.isin(unparsed)].sort_values(["utterance_id", "gra_index"])
    ids = parsed.utterance_id.to_numpy()
    columns = [parsed.gra_index.astype(int).tolist(), parsed.gloss.tolist(), parsed.stem.tolist(),
               parsed.part_of_speech.tolist(), parsed.gra_relation.tolist(), parsed.gra_head.astype(int).tolist()]
    is_target = (parsed.stem.isin(TARGET_FORMS) & (parsed.part_of_speech == "verb")).to_numpy()
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    rows = []
    for start, end in zip(starts, np.r_[starts[1:], len(ids)]):
        targets = np.flatnonzero(is_target[start:end])
        if not len(targets):
            continue
        nodes = _gra_nodes(*(c[start:end] for c in columns))
        for k in targets + start:
            rows.append((ids[k], columns[2][k], columns[1][k], *_verb_row(nodes[columns[0][k]])))
    return pd.DataFrame(rows, columns=COLUMNS)


def classify_gra_utterances(tokens, utterances):
    """classify_gra joined with utterance metadata (the gloss as `text`)."""
    return classify_gra(tokens).merge(utterances.assign(text=utterances.gloss), on="id", how="left")


def target_verbs(doc):
    """TARGET_FORMS tokens tagged as verbs in a spaCy doc."""
    return [t for t in doc if t.lemma_ in TARGET_FORMS and t.pos_ == "VERB"]


def classify_docs(docs):
    """One row per mental verb in each spaCy doc: COLUMNS, with the doc's position in place of
    `id`. Also returns the count of target forms spaCy did not tag as verbs."""
    rows, non_verbs = [], collections.Counter()
    forms = {f for fs in TARGET_FORMS.values() for f in fs}
    for index, doc in enumerate(docs):
        for t in doc:
            if t.lower_ in forms and not (t.lemma_ in TARGET_FORMS and t.pos_ == "VERB"):
                non_verbs[(t.lower_, t.pos_)] += 1
        for verb in target_verbs(doc):
            rows.append((index, verb.lemma_, verb.lower_, *_verb_row(verb)))
    return pd.DataFrame(rows, columns=["text_index", *COLUMNS[1:]]), non_verbs


def classify_texts(texts, nlp, batch_size=2000):
    """classify_docs over spaCy parses of `texts`."""
    return classify_docs(nlp.pipe(texts, batch_size=batch_size))


def target_utterances(utterances):
    """Utterances containing any TARGET_FORMS form as a whole word."""
    forms = "|".join(f for fs in TARGET_FORMS.values() for f in fs)
    return utterances[utterances.gloss.str.lower().str.contains(rf"(?<!\S)(?:{forms})(?!\S)", regex=True)]


def parse_cached(texts, nlp, path, batch_size=2000, n_process=1):
    """spaCy docs for `texts`, parsed once and cached as a DocBin at `path`, so changes to
    classify_complement apply without reparsing. n_process > 1 needs an `if __name__ == "__main__"` guard."""
    from spacy.tokens import DocBin

    if not os.path.exists(path):
        docs = DocBin(attrs=["ORTH", "LEMMA", "POS", "TAG", "DEP", "HEAD"])
        for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
            docs.add(doc)
        docs.to_disk(path)
    docs = list(DocBin().from_disk(path).get_docs(nlp.vocab))
    if len(docs) != len(texts) or any(d.text != t for d, t in zip(docs, texts)):
        raise ValueError(f"{path} was parsed from other texts (TARGET_FORMS changed?); delete it to reparse")
    return docs


def classify_utterances(utterances, nlp, path, n_process=1):
    """classify_docs over the target utterances (spaCy parses cached at `path`), joined with
    utterance metadata."""
    hits = target_utterances(utterances).reset_index(drop=True)
    texts = [parser_input(g, t) for g, t in zip(hits.gloss, hits.type)]
    verbs, _ = classify_docs(parse_cached(texts, nlp, path, n_process=n_process))
    return verbs.join(hits.assign(text=texts), on="text_index").drop(columns="text_index")


def read_conllu(path):
    """[(comments, words)] for each sentence of a CoNLL-U file: comments as {key: value}
    (`# key = value` lines), words as (index, form, lemma, upos, relation, head), lowercased,
    with the relation's subtype dropped (acl:relcl -> acl). Multiword-token and empty-node lines are skipped."""
    sentences, comments, words = [], {}, []
    with open(path) as f:
        for line in itertools.chain(f, [""]):
            line = line.rstrip("\n")
            if line.startswith("#"):
                key, _, value = line[2:].partition(" = ")
                comments[key] = value
            elif not line:
                if words:
                    sentences.append((comments, words))
                comments, words = {}, []
            elif not re.match(r"\d+[-.]", line):
                f_ = line.split("\t")
                words.append((int(f_[0]), f_[1].lower(), f_[2].lower(), f_[3], f_[7].split(":")[0], int(f_[6]) if f_[6] != "_" else 0))
    return sentences


def classify_conllu(sentences, reference=None):
    """One row per mental verb in each CoNLL-U sentence: sentence position, lemma, k (its
    occurrence among the sentence's verbs of that lemma), then COLUMNS[3:]. Verbs and lemmas come
    from `reference` (the gold sentences), so a parser's predicted trees of the same words are
    classified at the same verbs."""
    rows = []
    for n, ((_, words), (_, ref)) in enumerate(zip(sentences, reference or sentences)):
        nodes = _gra_nodes([w[0] for w in words], [w[1] for w in words], [r[2] for r in ref], [r[3] for r in ref], [w[4] for w in words], [w[5] for w in words])
        rows += [(n, r[2], *_verb_row(nodes[r[0]])) for r in ref if r[2] in TARGET_FORMS and r[3] == "VERB"]
    result = pd.DataFrame(rows, columns=["sentence", "lemma", *COLUMNS[3:]])
    return result.assign(k=result.groupby(["sentence", "lemma"]).cumcount())


def gold_utterance_ids(sentences, utterances):
    """sentence position -> childes-db utterance id for UD_English-CHILDES sentences whose corpus
    and CHILDES gloss (`childes_toks`) match an utterance's (the first, when several do)."""
    meta = pd.DataFrame([(n, c.get("corpus_name"), c.get("childes_toks", "").lower()) for n, (c, _) in enumerate(sentences)],
                        columns=["sentence", "corpus_name", "gloss"])
    keys = utterances[["id", "corpus_name"]].assign(gloss=utterances.gloss.str.lower()).drop_duplicates(["corpus_name", "gloss"])
    return meta.merge(keys, on=["corpus_name", "gloss"]).set_index("sentence").id


def gold_accuracy(gold, predicted, by="lemma"):
    """Accuracy of `predicted` categories (rows with `sentence`, `lemma`, `k`, `category`) against
    classify_conllu of the gold trees, overall and on verbs whose gold category is declarative or
    interrogative."""
    m = gold.merge(predicted[["sentence", "lemma", "k", "category"]], on=["sentence", "lemma", "k"], suffixes=("", "_pred"))
    clausal = m.category.isin(["declarative", "interrogative"])
    return pd.Series({
        "n": len(m), "accuracy": (m.category == m.category_pred).mean(),
        "n_clausal": clausal.sum(), "accuracy_clausal": (m.category == m.category_pred)[clausal].mean(),
    })


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


def polar_table(verbs):
    """Per lemma, for declarative and interrogative complements: count and the share whose
    matrix clause is a polar question (is_matrix_polar: do you know where it is?)."""
    clausal = verbs[verbs.category.isin(["declarative", "interrogative"])]
    table = clausal.groupby(["lemma", "category"]).polar.agg(n="size", polar="sum", p_polar="mean").unstack("category")
    return table.swaplevel(axis=1).sort_index(axis=1)


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
