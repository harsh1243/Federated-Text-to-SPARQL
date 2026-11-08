"""Final retriever for PROPERTIES and CLASSES (entities come later).

Pipeline position:

    question -> LLM-A -> [ this retriever ] -> LLM-B -> federated SPARQL

LLM-A emits (per llm_A.ipynb) a SPARQL skeleton plus a JSON block:

    ```sparql
    SELECT ?paper WHERE {
      ?paper a ?paper_type .
      ?einstein ?authored ?paper .
    }
    ```
    ```json
    {"properties": {"?authored": "links a person (subject) to a paper they authored (object)"},
     "classes":    {"?paper_type": "academic paper or scholarly publication"},
     "entities":   {"?einstein": "Albert Einstein, theoretical physicist"}}
    ```

This retriever turns each property/class slot into a ranked list of REAL IRIs
from each KG, using the class-aware index (kg_embeddings_class.zip).

The class-aware part
--------------------
The skeleton links a property variable to the class variable that constrains
its subject:

    ?paper a ?paper_type .        <- constrains ?paper
    ?einstein ?authored ?paper .  <- so ?authored is constrained by ?paper_type

So we can, for each property slot, find the class slot that governs it and
boost the properties that actually fire on that class.  The boost is SOFT
(an additive bonus on the cosine score), never a filter: a wrong class then
costs nothing, whereas a hard filter with a wrong class collapsed retrieval
from 89.6% to 63.8%.

Retrieval is PER-KG: every KG contributes its own top-k, because a federated
query spans several KGs and the same conceptual slot exists in each of them
under that KG's own namespace.
"""
import json
import math
import re
import zipfile
import collections
import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_ZIP = "kg_embeddings_class.zip"
MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"
TOP_K = 10

# How much to boost a property that fires on the retrieved class.  Measured
# optimum: flat between 0.10 and 0.15, degrades past 0.20 (a wrong class
# starts pushing the right answer down).  With a wrong class the boost is a
# no-op, so this is safe to leave on.
CLASS_BOOST = 0.12

# How much the LEXICAL channel contributes to a property's score.
#
# The dense channel alone is not enough.  A slot description like "links a
# drug (subject) to its therapeutic indication (object)" is semantically
# closest to drugbank's `target` (a drug's molecular target) -- the encoder
# reads both as "what the drug acts on" -- so `target` outranks `indication`
# 0.741 to 0.679 and the query asks for the wrong thing.
#
# But the LABEL says exactly which one is meant: BM25 over the property
# labels scores `indication` at 4.59 and `target` at 0, because the
# description literally contains the word "indication".  A property NAME is
# an exact technical term, and exact terms are what lexical matching is good
# at; the embedding is good at paraphrase, which is what the lexical channel
# is not.  Combining them is the standard hybrid, and the combination is
# what a reader of the query would do: notice the word "indication".
#
# The weight is small enough that the dense channel still decides when the
# lexical signal is weak (every label shares the generic tokens), and large
# enough to break a near-tie when one label is an exact match.
LEX_WEIGHT = 0.25

# Predicates that are structural (join/anchor vocabulary) rather than
# content.  They fire on almost every entity, so they are never suppressed
# by the class boost, and they are always offered to LLM-B.
STRUCTURAL_PREDICATES = {
    'http://www.w3.org/2000/01/rdf-schema#label',
    'http://www.w3.org/2002/07/owl#sameAs',
    'http://www.w3.org/1999/02/22-rdf-syntax-ns#type',
    'http://www.w3.org/2004/02/skos/core#subject',
    'http://purl.org/dc/terms/title',
}

_TOKEN = re.compile(r'[a-z0-9]+')
_STOP = set("""a an the of to and or in on for with is are was were be
by as at from that this it its their his her they them we you i""".split())


def _stem(t):
    """A crude suffix stripper, so morphological variants match.

    The property labels are technical nouns and the slot descriptions use
    the inflected form of the same word: "the medical condition for which it
    is INDICATED" must reach the predicate `indication`.  Exact-token
    matching cannot do that -- "indicated" and "indication" share no token.

    So both sides are reduced to a stem.  The rules are deliberately few:
    plurals, past tense, gerund, and the -ion/-tion endings, which is what
    separates "indicated"/"indication" and "condition"/"conditional".
    """
    if len(t) <= 3:
        return t
    for suf, repl in (("ies", "y"), ("ied", "y"), ("ing", ""), ("ed", ""),
                       ("ions", "ion"), ("ion", ""), ("ions", ""),
                       ("ation", "at"), ("ations", "at")):
        if t.endswith(suf) and len(t) - len(suf) >= 3:
            return t[:-len(suf)] + repl
    if t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def _tokens(s):
    """Lowercase word tokens, stopwords removed, stemmed."""
    return [_stem(t) for t in _TOKEN.findall(s.lower()) if t not in _STOP]


# A slot description names the SUBJECT type and the OBJECT type in
# parentheses: "links a film (subject) to the person who directed it
# (object)".  The words before "(subject)" describe the entity the property
# starts FROM; the words after it describe the relation.
#
# That distinction matters for the lexical channel.  The subject-type words
# are not about the relation, and matching them against property labels
# picks predicates about the subject instead of the relation: for "links a
# film (subject) to the person who directed it (object)", the token "film"
# matches lmdb's `film_subject` (the film's subject matter) far better than
# it matches `director`, and the wrong predicate wins.
#
# So the lexical channel reads only the OBJECT phrase -- the part that
# actually names what the property points to.
_SUB_MARK = re.compile(r'\(subject\)', re.I)
_OBJ_MARK = re.compile(r'\(object\)', re.I)


def _object_phrase(desc):
    """The part of a slot description that names the relation's target."""
    s = _SUB_MARK.split(desc, 1)
    if len(s) == 2:
        desc = s[1]
    s = _OBJ_MARK.split(desc, 1)
    if len(s) == 2:
        desc = s[0]
    return desc


# Two tokens match when they share a prefix this long.  This is what bridges
# "directed" to the label "director": the two share no stem, but they do
# share a prefix, and a shared prefix of five or more characters between two
# content words is not a coincidence.
_PREFIX = 5


def _matches(qtok, dtok):
    """Do a query token and a document token denote the same word?"""
    if qtok == dtok:
        return True
    n = min(len(qtok), len(dtok))
    if n < _PREFIX:
        return False
    return qtok[:n] == dtok[:n]


def _bm25(query, labels):
    """BM25 of a query against a list of document labels.

    Small and self-contained: the property labels are a few thousand per KG,
    so there is no reason to build a separate index format for them.
    """
    docs = [set(_tokens(l)) for l in labels]
    n = len(docs)
    if not n:
        return np.zeros(0)
    df = collections.Counter()
    for d in docs:
        df.update(d)
    avgdl = sum(len(d) for d in docs) / n
    qt = _tokens(query)
    out = np.zeros(n)
    for i, d in enumerate(docs):
        s = 0.0
        for w in qt:
            # a query token matches a document token exactly, or by prefix
            hit = None
            for dt in d:
                if _matches(w, dt):
                    hit = dt
                    break
            if hit is None:
                continue
            # the IDF is that of the DOCUMENT token that matched: a rare
            # document word matched exactly is a stronger signal than a
            # common one matched by prefix.
            idf = math.log(1 + (n - df[hit] + 0.5) / (df[hit] + 0.5))
            s += idf * 2.5 / (1.5 * (1 - 0.75 + 0.75 * len(d) / avgdl) + 1.5)
        out[i] = s
    return out

_kgs = None
_model = None



# ------------------------------------------------------------------ loading
def _load():
    """Lazy-load model + index once."""
    global _kgs, _model
    if _kgs is not None:
        return _kgs, _model

    _model = SentenceTransformer(MODEL_NAME)
    _kgs = {}

    with zipfile.ZipFile(EMBED_ZIP) as z:
        for name in z.namelist():
            if not name.endswith(".json"):
                continue
            data = json.loads(z.read(name))
            kg = data["kg"]

            prop_matrix = np.array([p["embedding"] for p in data["properties"]],
                                   dtype=np.float32)
            class_matrix = np.array([c["embedding"] for c in data["classes"]],
                                     dtype=np.float32)
            prop_matrix /= (np.linalg.norm(prop_matrix, axis=1, keepdims=True) + 1e-12)
            class_matrix /= (np.linalg.norm(class_matrix, axis=1, keepdims=True) + 1e-12)

            _kgs[kg] = {
                "kg": kg,
                "props_vec": prop_matrix,
                "classes_vec": class_matrix,
                "props_meta": data["properties"],
                "classes_meta": data["classes"],
                "class_predicates": data.get("class_predicates", {}),
            }
    return _kgs, _model


# ------------------------------------------------------------------ parsing
def _parse_llm_a(llm_text):
    """Strip fences, return the JSON block as a dict."""
    s = llm_text.strip()
    if "```" in s:
        # the JSON block is the LAST fenced block
        parts = s.split("```")
        for part in reversed(parts):
            t = part.strip()
            if t.startswith("json"):
                t = t[4:]
            t = t.strip()
            if t.startswith("{"):
                return json.loads(t)
        raise ValueError("no JSON block found in LLM-A output")
    return json.loads(s)


def _parse_skeleton(llm_text):
    """Return the SPARQL skeleton text, or '' if there is none."""
    if "```" not in llm_text:
        return ""
    for part in llm_text.split("```"):
        t = part.strip()
        if t.lower().startswith("sparql"):
            return t[6:].strip()
    return ""


# SPARQL keywords that may open a line which is NOT a triple pattern.
# The LLM-A prompt explicitly allows "OPTIONAL, FILTER, GROUP BY, ORDER BY,
# LIMIT, BIND, VALUES, aggregates", so the skeleton legitimately contains
# lines that are not triples.  A line is only a triple if its predicate
# position holds a variable, the keyword `a`, or an IRI -- which is what
# `split_triple_line` checks.
_SPARQL_KEYWORDS = (
    'filter', 'bind', 'values', 'optional', 'union', 'minus', 'graph',
    'service', 'group', 'order', 'having', 'limit', 'offset', 'select',
    'construct', 'ask', 'describe', 'prefix', 'base', 'insert', 'delete',
)


def split_triple_line(line):
    """Split a skeleton line into (subject, predicate, object), or None.

    Returns None for anything that is not a triple pattern: a FILTER, a
    BIND, a VALUES clause, a blank line, a comment, or a line whose
    predicate position holds something other than a variable, the `a`
    keyword, or an IRI.

    Why this exists: the previous parser accepted any line that split into
    exactly three whitespace-separated tokens.  A FILTER over a string with
    a space in it --

        FILTER(CONTAINS(LCASE(STR(?person_name)), "bernard montgomery"))

    -- splits into three tokens, so it was silently accepted as a triple
    whose predicate was the literal `"bernard`.  That invented a property
    slot that does not exist, and the constraint solver then went looking
    for a predicate called `"bernard`.

    The cached LLM-A run never used FILTER, so this never fired before;
    a live run with a newer model does.
    """
    line = line.strip()
    if not line or line.startswith('#') or line.startswith('}'):
        return None
    if line.endswith('.'):
        line = line[:-1].strip()
    if not line:
        return None
    low = line.lower()
    for kw in _SPARQL_KEYWORDS:
        if low.startswith(kw + '(') or low.startswith(kw + ' '):
            return None
    parts = line.split()
    if len(parts) != 3:
        return None
    s, p, o = parts
    # the predicate position decides it: a variable, `a`, or an IRI
    if not (p.startswith('?') or p == 'a' or (p.startswith('<')
                                            and p.endswith('>'))):
        return None
    # the subject must be a variable or an IRI
    if not (s.startswith('?') or (s.startswith('<') and s.endswith('>'))):
        return None
    return s, p, o


def _class_var_of_property(skeleton, prop_var):
    """Which class variable governs the subject of prop_var?

    In the skeleton, look for  ?x a ?class .  and then find the property
    triple whose subject is ?x.  Returns the class variable name, or None.

    Example:
        ?paper a ?paper_type .
        ?einstein ?authored ?paper .
      -> ?authored is governed by ?paper_type   (?einstein is the subject)

    Note the direction matters: the class that describes the SUBJECT of the
    property triple is the one that constrains which predicates can fire.
    If the subject has no type but the object does, we fall back to the
    object's class, which is still a useful constraint (e.g. a film-typed
    object implies film properties on the way out).
    """
    if not skeleton:
        return None

    # 1. collect  ?x a ?class .   triples
    typed = {}
    triples = []
    for line in skeleton.splitlines():
        parts = split_triple_line(line)
        if parts is None:
            continue
        triples.append(parts)
        if parts[1] == 'a':
            typed[parts[0]] = parts[2]

    # 2. find the triple in which prop_var is the predicate
    for s, p, o in triples:
        if p != prop_var:
            continue
        if s in typed:
            return typed[s]
        if o in typed:
            return typed[o]
        return None
    return None


# ------------------------------------------------------------------ scoring
def _topk(query_vec, kgs, slot, top_k):
    """Candidates for the given slot ('props' or 'classes'), per KG."""
    hits = []
    for kg, d in kgs.items():
        matrix = d["%s_vec" % slot]
        meta = d["%s_meta" % slot]
        scores = matrix @ query_vec
        top_idx = np.argsort(-scores)[:top_k]
        for i in top_idx:
            hits.append({
                "kg": kg,
                "uri": meta[i]["uri"],
                "label": meta[i]["label"],
                "score": round(float(scores[i]), 4),
            })
    hits.sort(key=lambda x: -x["score"])
    return hits


def _topk_by_kg(query_vec, kgs, slot, top_k, boost_sets=None, boost=CLASS_BOOST,
                query_text=None):
    """Same as _topk but grouped by KG: {kg: [hit, ...]}.

    boost_sets: {kg: set(predicate_uris)} whose score gets +boost.
    query_text: the slot description, used by the lexical channel (props).
    """
    grouped = {}
    for kg, d in kgs.items():
        matrix = d["%s_vec" % slot]
        meta = d["%s_meta" % slot]
        scores = matrix @ query_vec
        if slot == "props":
            if boost_sets and boost_sets.get(kg):
                allowed = boost_sets[kg]
                for i, m in enumerate(meta):
                    if m["uri"] in allowed or m["uri"] in STRUCTURAL_PREDICATES:
                        scores[i] += boost
            if query_text:
                scores = scores + LEX_WEIGHT * _lexical(d, query_text)
        top_idx = np.argsort(-scores)[:top_k]
        hits = [{
            "kg": kg,
            "uri": meta[i]["uri"],
            "label": meta[i]["label"],
            "score": round(float(scores[i]), 4),
        } for i in top_idx]
        if slot == "props":
            hits = _ensure_structural(hits, meta, scores, kg)
        grouped[kg] = hits
    return grouped


_LEX_CACHE = {}


def _lexical(d, query_text):
    """The lexical channel for one KG's property list, cached.

    BM25 over the property LABELS, normalised so it is comparable with the
    cosine scores (which live in roughly [0.4, 0.9]).  Normalising by the
    maximum keeps the two channels on a common scale, so LEX_WEIGHT means
    the same thing for every slot.
    """
    kg = d["kg"]
    cached = _LEX_CACHE.get(kg)
    if cached is None:
        labels = [m["label"] for m in d["props_meta"]]
        cached = (labels, np.zeros(len(labels)))
        _LEX_CACHE[kg] = cached
    labels, _ = cached
    sc = _bm25(_object_phrase(query_text), labels)
    m = float(sc.max()) if len(sc) else 0.0
    return sc / m if m > 0 else sc


def _ensure_structural(hits, meta, scores, kg):
    """Guarantee the structural (join/anchor) vocabulary reaches LLM-B.

    A cross-KG federated query is built on rdfs:label (to match the entity by
    name) and owl:sameAs (to join across KGs); rdf:type and skos:subject carry
    the class and category constraints.  These are semantically generic, so
    plain cosine ranks them below 10 and LLM-B would never see them -- and a
    federated query that cannot join is useless.

    So: any structural predicate that exists in this KG and did not make the
    cut is appended, so it is always available to LLM-B.
    """
    have = {h["uri"] for h in hits}
    extra = [(float(scores[i]), i) for i, m in enumerate(meta)
             if m["uri"] in STRUCTURAL_PREDICATES and m["uri"] not in have]
    if not extra:
        return hits
    extra.sort(reverse=True)
    for _, i in extra:
        hits.append({
            "kg": kg,
            "uri": meta[i]["uri"],
            "label": meta[i]["label"],
            "score": round(float(scores[i]), 4),
        })
    return hits


# ------------------------------------------------------------------ main API
def retrieve(llm_a_output, top_k=TOP_K, use_class_boost=True):
    """
    Input  : raw LLM-A text (with or without ```json fences)
    Output : dict ready to feed into LLM-B stage
        {
          "properties": {
              "?authored": [{"kg", "uri", "label", "score"}, ...per-KG top-k...],
              ...
          },
          "classes": {
              "?paper_type": [{"kg", "uri", "label", "score"}, ...top_k...],
              ...
          },
          "properties_by_kg": { var: {kg: [hits]} },
          "classes_by_kg":    { var: {kg: [hits]} },
          "class_boost": { var: {kg: [class_uris_used]} },
          "entities": { ... untouched from LLM-A ... }
        }
    """
    kgs, model = _load()
    parsed = _parse_llm_a(llm_a_output)
    skeleton = _parse_skeleton(llm_a_output)

    props = parsed.get("properties", {})
    classes = parsed.get("classes", {})

    out = {"properties": {}, "classes": {}, "entities": parsed.get("entities", {})}
    out["properties_by_kg"] = {}
    out["classes_by_kg"] = {}
    out["class_boost"] = {}

    # ---- classes first: they decide which properties get boosted
    class_hits_by_var = {}
    if classes:
        descs = list(classes.values())
        vecs = model.encode(descs, normalize_embeddings=True, convert_to_numpy=True)
        for var, desc in classes.items():
            q = vecs[descs.index(desc)]
            out["classes"][var] = _topk(q, kgs, "classes", top_k)
            by_kg = _topk_by_kg(q, kgs, "classes", top_k)
            out["classes_by_kg"][var] = by_kg
            class_hits_by_var[var] = by_kg

    # ---- properties, boosted by the class that governs them
    if props:
        descs = list(props.values())
        vecs = model.encode(descs, normalize_embeddings=True, convert_to_numpy=True)
        for var, desc in props.items():
            q = vecs[descs.index(desc)]

            boost_sets, used = _boost_sets(skeleton, var, class_hits_by_var, kgs)
            if boost_sets and use_class_boost:
                out["class_boost"][var] = used

            out["properties"][var] = _topk(q, kgs, "props", top_k)
            out["properties_by_kg"][var] = _topk_by_kg(
                q, kgs, "props", top_k,
                boost_sets=boost_sets if use_class_boost else None,
                query_text=desc)

    return out


def _boost_sets(skeleton, prop_var, class_hits_by_var, kgs, top_classes=3):
    """The predicate sets to boost for this property variable.

    Find the class variable that governs the property's subject, take that
    class's top few retrieved candidates, and union their predicate sets.
    """
    cls_var = _class_var_of_property(skeleton, prop_var)
    if not cls_var or cls_var not in class_hits_by_var:
        return None, {}

    boost_sets = {}
    used = {}
    for kg, hits in class_hits_by_var[cls_var].items():
        preds = set()
        picked = []
        for h in hits[:top_classes]:
            cp = kgs[kg]["class_predicates"].get(h["uri"])
            if cp:
                preds |= set(cp)
                picked.append(h["uri"])
        if preds:
            boost_sets[kg] = preds
            used[kg] = picked
    if not boost_sets:
        return None, {}
    return boost_sets, used


def retrieve_from_parsed(parsed, skeleton="", top_k=TOP_K, use_class_boost=True):
    """Same as retrieve(), but for an already-parsed LLM-A JSON dict."""
    if skeleton:
        text = "```sparql\n%s\n```\n\n```json\n%s\n```" % (
            skeleton, json.dumps(parsed))
    else:
        text = "```json\n%s\n```" % json.dumps(parsed)
    return retrieve(text, top_k=top_k, use_class_boost=use_class_boost)


if __name__ == "__main__":

    llm_a_text = """
```sparql
SELECT ?paper WHERE {
  ?paper a ?paper_type .
  ?einstein ?authored ?paper .
  ?einstein ?worked_at ?princeton .
}
```

```json
{
  "properties": {
    "?authored": "links a person (subject) to a paper they authored (object)",
    "?worked_at": "links a person (subject) to an organization they worked at (object)"
  },
  "classes": {
    "?paper_type": "academic paper or scholarly publication"
  },
  "entities": {
    "?einstein": "Albert Einstein, theoretical physicist",
    "?princeton": "Princeton University"
  }
}
```
"""
    result = retrieve(llm_a_text, top_k=5)
    print(json.dumps(result, indent=2))
