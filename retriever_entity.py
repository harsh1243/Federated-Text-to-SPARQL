"""Entity retriever: LLM-A entity slot -> candidate entity IRIs per KG.

Pipeline position:

    question -> LLM-A -> [ this retriever ] -> LLM-B -> federated SPARQL

LLM-A emits a SPARQL skeleton plus a JSON block.  The entity slot is a CRISP
NAME, not a description:

    ```json
    {"entities": {"?the_rock": "The Rock", "?cyclosporine": "Cyclosporine"}}
    ```

This matches BM25: a name is matched on surface form, so lexical scoring finds
exact and near-exact name hits, and no embedding pass over millions of
entities is needed (which would take ~10 hours on this CPU-only box).

Two ideas do the real work.

1. ANCHOR, THEN EXPAND.  A federated query anchors on ONE entity -- in one
   KG, by name -- and reaches the other KGs through owl:sameAs:

       SERVICE <lmdb>    { ?actor owl:sameAs ?x . ?actor rdfs:label "..." . }
       SERVICE <dbpedia> { ?x dbo:birthDate ?d . }

   So we retrieve the anchor in every KG (each KG has its own namespace and
   its own naming conventions), then expand each hit through the sameAs edge
   table to offer the entity's counterparts in the other KGs.  Measured on the
   50 gold queries: every single cross-KG partner (56/56) is reachable this
   way.  Searching each KG independently for the name instead would be much
   weaker -- dbpedia spells films "The Rock (film)" while lmdb spells the same
   film "The Rock".

2. EXACT-FIRST, THEN LEXICAL.  The overwhelmingly common case is that the
   name in the question IS the label in the KG.  So we first try an exact
   (case-folded, whitespace-normalised) label lookup, and only fall back to
   BM25 for names that miss.  This makes the common case exact and keeps
   BM25's recall for the rest.

The index is per-KG and the results are per-KG: a federated query spans
several KGs and the same conceptual entity exists in each of them under that
KG's own namespace.

Storage is memory-mapped (see build_entity_index.py) so a query only faults
in the pages it touches -- the box has 8 GB of RAM.
"""
import hashlib
import json
import os
import zipfile

import bm25s
import numpy as np

INDEX_DIR = '_entity_bm25'
INDEX_ZIP = 'kg_entities_bm25.zip'
TOP_K = 10

KGS = ('dbpedia', 'lmdb', 'drugbank')

POOL_NAMES = {0: 'class', 1: 'untyped', 2: 'label_only'}

SAMEAS_FILE = 'sameas_joins.json'

_state = None


def _norm(s):
    """Normalisation for the exact-match lookup.

    Case-folded, whitespace-collapsed.  The trailing disambiguator
    parenthetical is KEPT -- "The Rock (film)" and "The Rock" are different
    dbpedia resources and must not collapse into one another.
    """
    return ' '.join(s.split()).lower()


def _hash(s):
    return int.from_bytes(
        hashlib.blake2b(s.encode('utf-8'), digest_size=8).digest(), 'big')


def _uri_hash(s):
    return int.from_bytes(
        hashlib.blake2b(s.encode('utf-8'), digest_size=8).digest(), 'big')


class _KG:
    """One KG's BM25 index plus its memory-mapped metadata."""

    def __init__(self, d):
        self.bm25 = bm25s.BM25.load(d, load_corpus=False, show_progress=False)
        self.uri_blob = np.memmap(os.path.join(d, 'uri_blob.bin'), dtype=np.uint8,
                                   mode='r')
        self.lab_blob = np.memmap(os.path.join(d, 'lab_blob.bin'), dtype=np.uint8,
                                   mode='r')
        self.uri_off = np.load(os.path.join(d, 'uri_off.npy'))
        self.lab_off = np.load(os.path.join(d, 'lab_off.npy'))
        self.cls = np.load(os.path.join(d, 'cls.npy'))
        self.pool = np.load(os.path.join(d, 'pool.npy'))
        self.cls_names = json.load(open(os.path.join(d, 'cls_names.json'),
                                        encoding='utf-8'))
        self.exact = np.load(os.path.join(d, 'exact.npy'))
        self.exact_doc = np.load(os.path.join(d, 'exact_doc.npy'))
        self.uri_hash = np.load(os.path.join(d, 'uri_hash.npy'))
        self.uri_doc = np.load(os.path.join(d, 'uri_doc.npy'))

    def uri(self, i):
        return bytes(self.uri_blob[self.uri_off[i]:self.uri_off[i + 1]]).decode('utf-8')

    def label(self, i):
        return bytes(self.lab_blob[self.lab_off[i]:self.lab_off[i + 1]]).decode('utf-8')

    def class_name(self, i):
        c = int(self.cls[i])
        return self.cls_names[c] if c >= 0 else None

    def exact_ids(self, name):
        """Doc ids whose label equals `name` (after normalisation)."""
        h = np.uint64(_hash(_norm(name)))
        lo = int(np.searchsorted(self.exact, h, 'left'))
        hi = int(np.searchsorted(self.exact, h, 'right'))
        if lo == hi:
            return []
        out = []
        for j in range(lo, hi):
            i = int(self.exact_doc[j])
            if _norm(self.label(i)) == _norm(name):
                out.append(i)
        return out

    def class_of_uri(self, uri):
        """The partition class of an entity given its URI.

        Needed when an entity is reached through owl:sameAs: the counterpart
        lives in another KG with its own class ("actor" in lmdb is "Actor"
        in dbpedia), and the class decides which predicates can fire on it.
        """
        h = np.uint64(_uri_hash(uri))
        lo = int(np.searchsorted(self.uri_hash, h, 'left'))
        hi = int(np.searchsorted(self.uri_hash, h, 'right'))
        for j in range(lo, hi):
            i = int(self.uri_doc[j])
            if self.uri(i) == uri:
                return self.class_name(i)
        return None


def _load():
    """Lazy-load the BM25 indexes and the sameAs table once."""
    global _state
    if _state is not None:
        return _state

    root = INDEX_DIR
    if not os.path.isdir(root):
        raise RuntimeError(
            '%s not found -- run build_entity_index.py first' % INDEX_DIR)

    kgs = {}
    for kg in KGS:
        kgs[kg] = _KG(os.path.join(root, kg))

    sameas = {}
    sa = os.path.join(root, SAMEAS_FILE)
    if os.path.exists(sa):
        for j in json.load(open(sa, encoding='utf-8'))['joins']:
            sameas.setdefault(j['subject'], set()).add(j['object'])
            sameas.setdefault(j['object'], set()).add(j['subject'])

    _state = {'kgs': kgs, 'sameas': sameas}
    return _state


def _hit(kg, i, score, source):
    g = kg
    return {
        'kg': None,          # filled by the caller
        'uri': g.uri(i),
        'label': g.label(i),
        'class': g.class_name(i),
        'pool': POOL_NAMES.get(int(g.pool[i]), '?'),
        'score': round(float(score), 4),
        'source': source,
    }


def retrieve_entity(name, kg, top_k=TOP_K, expand_sameas=True):
    """Candidates for one entity name in one KG.

    Returns a list of hits, best first.  Each hit may carry `also_as`: the
    same entity's URI in the OTHER KGs, reached through owl:sameAs.
    """
    st = _load()
    g = st['kgs'][kg]

    # ---- 1. exact label match first
    exact = g.exact_ids(name)

    # ---- 2. BM25 to fill out the list (and to cover names that miss exactly)
    bm = []
    if len(exact) < top_k:
        qt = bm25s.tokenize([name], stopwords='en', show_progress=False)
        idx, sc = g.bm25.retrieve(qt, k=top_k, show_progress=False)
        bm = [(int(i), float(s)) for i, s in zip(idx[0], sc[0])]

    seen = set()
    hits = []
    for i in exact:
        if i in seen:
            continue
        seen.add(i)
        h = _hit(g, i, 100.0, 'exact')
        h['kg'] = kg
        hits.append(h)
    for i, s in bm:
        if i in seen:
            continue
        seen.add(i)
        h = _hit(g, i, s, 'bm25')
        h['kg'] = kg
        hits.append(h)

    # ---- 3. cross-KG expansion through owl:sameAs
    if expand_sameas:
        partners = set()
        for h in hits[:top_k]:
            partners |= st['sameas'].get(h['uri'], set())
        if partners:
            for h in hits:
                h['also_as'] = sorted(p for p in partners if p != h['uri'])

    return hits


def retrieve_entities(llm_a_output, top_k=TOP_K, expand_sameas=True):
    """Retrieve candidates for every entity slot in LLM-A's output.

    Input  : raw LLM-A text (with or without ```json fences)
    Output : { "?the_rock": { "dbpedia": [hit, ...], "lmdb": [hit, ...] }, ... }
    """
    parsed = _parse_llm_a(llm_a_output)
    ents = parsed.get('entities', {}) or {}

    out = {}
    for var, name in ents.items():
        name = (name or '').strip()
        if not name:
            continue
        per_kg = {}
        for kg in KGS:
            per_kg[kg] = retrieve_entity(name, kg, top_k=top_k,
                                          expand_sameas=expand_sameas)
        out[var] = per_kg
    return out


def _parse_llm_a(llm_text):
    s = llm_text.strip()
    if '```' in s:
        parts = s.split('```')
        for part in reversed(parts):
            t = part.strip()
            if t.startswith('json'):
                t = t[4:]
            t = t.strip()
            if t.startswith('{'):
                return json.loads(t)
        raise ValueError('no JSON block found in LLM-A output')
    return json.loads(s)


def load_zip_to_dir(zip_path=INDEX_ZIP, dest=INDEX_DIR):
    """Unpack the shipped index archive (for deployment)."""
    if os.path.isdir(dest):
        return dest
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    return dest


if __name__ == '__main__':

    llm_a_text = """
```sparql
SELECT ?director ?release_date WHERE {
  ?the_rock a ?film_type .
  ?the_rock ?directed_by ?director .
}
```

```json
{
  "properties": {
    "?directed_by": "links a film (subject) to the person who directed it (object)"
  },
  "classes": {
    "?film_type": "feature film or motion picture"
  },
  "entities": {
    "?the_rock": "The Rock"
  }
}
```
"""
    res = retrieve_entities(llm_a_text, top_k=5)
    for var, per_kg in res.items():
        print('%s' % var)
        for kg, hits in per_kg.items():
            print('  %-9s' % kg)
            for h in hits[:5]:
                extra = ''
                if h.get('also_as'):
                    extra = '  also_as=%s' % ','.join(
                        p.rsplit('/', 1)[-1] for p in h['also_as'][:3])
                print('      %-6s %-30s [%-16s] %s%s'
                      % (h['source'], h['label'], h['class'] or '-',
                         h['uri'].rsplit('/', 1)[-1], extra))
