"""Constraint-aware joint grounding for federated text-to-SPARQL.

LLM-A emits a KG-AGNOSTIC skeleton: a set of triple patterns over variables,
plus NL descriptions for the property, class and entity slots.  Retrieval then
produces, per slot, a ranked list of candidate IRIs *per KG*.  The two lists
are independent, so the naive candidate space handed to LLM-B contains a large
number of combinations that no endpoint can ever execute.

This module replaces that flat list with a ranked set of GROUNDED PLANS, where
a plan is a complete, executable assignment.  Two constraints do the work.

C1  KG-LOCALITY
    Every triple in a federated query is served by exactly ONE endpoint.  So
    the predicate, the class and the entity bound in a triple must all come
    from that same KG.  A triple whose predicate is dbpedia's dbo:birthDate
    but whose subject is an lmdb resource is not executable by any endpoint,
    no matter how well the two retrieve independently.

C2  BRIDGEABILITY
    Entities that are joined across endpoint boundaries must be linked by an
    identity edge.  In our dumps that edge is owl:sameAs.  When two entities
    sit in the SAME KG and are joined through a shared value variable, the
    equivalent link is a shared skos:subject category -- the same mechanism,
    simulated inside one graph.

    C2 has two faces:
      C2a  one logical entity realised in several KGs -> those realisations
           must be sameAs-linked (this is how a 2-KG query works).
      C2b  two different entities joined through a shared variable -> they
           must be bridged, and the shared variable is then BOUND to the
           bridging value.  This is how the 3-KG T-series queries work: the
           shared variable becomes the literal category.

Both constraints are checked against the real dump contents, so a rejected
combination is rejected because it provably cannot execute -- not because a
model guessed.

The output of solve() is what LLM-B should consume: a small ranked list of
plans, each carrying the KG of every triple, the URI bound to every slot, the
sameAs edges and category hubs used, and the constant that the shared
variable must take.
"""
import collections
import itertools
import json
import os
import re
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import retriever_entity as RE
import retriever_final as RF

BRIDGE_FILE = os.path.join(BASE, '_bridge_graph.json')
RANGES_FILE = os.path.join(BASE, '_predicate_ranges.json')
CLASSPRED_FILE = os.path.join(BASE, '_class_predicates.json')

# A category shared by more entities than this is a bucket ("1981_deaths"),
# not a bridge.  Buckets are still allowed, but they rank last.
MAX_HUB_MEMBERS = 60

# How many sameAs hops to follow when looking for a bridge.
MAX_BRIDGE_DEPTH = 3

# The JOIN and ANCHOR vocabulary.  These predicates express how a federated
# query is stitched together -- the identity edge and the name literal --
# never the fact the question asks for.  They are retrieved separately (the
# class retriever force-includes them so the compiler can build the query),
# so a CONTENT property slot must never bind one of them.
#
# rdf:type is here too: it carries a class CONSTRAINT, which is what the
# class slot is for.  A content property slot never means "the type of this
# thing".
#
# skos:subject is NOT here: the T-series genuinely retrieves a category
# through skos:subject.
JOIN_ONLY_PREDICATES = {
    'http://www.w3.org/2002/07/owl#sameAs',
    'http://www.w3.org/2000/01/rdf-schema#label',
    'http://purl.org/dc/terms/title',
    'http://www.w3.org/1999/02/22-rdf-syntax-ns#type',
}

# skos:subject is CONTENT only when the slot's object is a shared category
# variable -- that is the T-series join.  Everywhere else it is just "the
# wiki category this resource is filed under", which is never the fact a
# question asks for, and it would otherwise outrank the right predicate on
# a bare cosine score (it fires on nearly every dbpedia entity).
SKOS_SUBJECT = 'http://www.w3.org/2004/02/skos/core#subject'


# ===================================================================== bridge
_KG_STATE = None


def _kg_state():
    """The entity index state, loaded once."""
    global _KG_STATE
    if _KG_STATE is None:
        _KG_STATE = RE._load()
    return _KG_STATE


class Bridge:
    """The cross-KG join surface: owl:sameAs + the dbpedia category hub.

    Category membership is stored, not pairs -- the first version materialised
    every pair of entities sharing a category and blew past 500K edges,
    silently dropping real chains.  Membership is 209K rows and exact.
    """

    def __init__(self, path=BRIDGE_FILE):
        if not os.path.exists(path):
            raise RuntimeError('%s not found -- run build_bridge_graph.py'
                               % path)
        d = json.load(open(path, encoding='utf-8'))
        self.sameas = d['sameas']                    # uri -> [[partner, kg],..]
        self.cats = d['cats']                        # uri -> [category index]
        self.cat_names = d['cat_names']
        self.cat_cand = d['cat_cand']                # bridge-candidates per cat
        self.cat_global = d['cat_global']            # members in the whole dump
        self.meta = d.get('meta', {})

        # inverted category index -> member entities
        self.cat_members = collections.defaultdict(list)
        for u, idxs in self.cats.items():
            for i in idxs:
                self.cat_members[i].append(u)

    # -------------------------------------------------------------- sameAs
    def partners(self, uri, kg=None):
        """sameAs partners of `uri`, optionally restricted to one KG."""
        out = [(p, k) for p, k in self.sameas.get(uri, [])]
        if kg is not None:
            out = [(p, k) for p, k in out if k == kg]
        return out

    def partner_uris(self, uri, kg):
        return [p for p, k in self.partners(uri, kg)]

    # ----------------------------------------------------------------- hub
    def shared_categories(self, u1, u2):
        """Categories both entities belong to, most specific first.

        "Most specific" = fewest bridge candidates, then fewest members in
        the whole dump.  A bucket category like Category:1981_deaths is
        shared by hundreds of entities and means nothing; Category:Heroin is
        shared by three and is a real join.
        """
        a = self.cats.get(u1)
        b = self.cats.get(u2)
        if not a or not b:
            return []
        sb = set(b)
        out = []
        for i in set(a) & sb:
            out.append((self.cat_cand[i], self.cat_global[i], i))
        out.sort()
        return [(self.cat_names[i], nc, ng) for nc, ng, i in out]

    def best_shared_category(self, u1, u2, max_members=MAX_HUB_MEMBERS):
        """The single most specific shared category, or None."""
        for cat, nc, ng in self.shared_categories(u1, u2):
            if nc <= max_members:
                return cat, nc, ng
        return None

    def _hub_neighbours(self, uri):
        """Entities sharing a category with `uri` (dbpedia-internal hop)."""
        out = []
        for i in self.cats.get(uri, ()):
            if self.cat_cand[i] > MAX_HUB_MEMBERS:
                continue
            for u in self.cat_members[i]:
                if u != uri:
                    out.append(u)
        return out

    def _neighbours(self, uri):
        out = [(p, 'sameAs') for p, k in self.partners(uri)]
        if 'dbpedia.org' in uri:
            out += [(u, 'category') for u in self._hub_neighbours(uri)]
        return out

    def find_bridge(self, u1, u2, max_depth=MAX_BRIDGE_DEPTH):
        """A path from u1 to u2, or None.

        Returns {'kind', 'path': [(uri, edge_kind), ...]}.
        """
        if u1 == u2:
            return {'kind': 'same', 'path': [(u1, None)]}
        for p, k in self.partners(u1):
            if p == u2:
                return {'kind': 'sameAs', 'path': [(u1, None), (u2, 'sameAs')]}

        seen = {u1}
        frontier = {u1: [(u1, None)]}
        for _ in range(max_depth):
            nxt = {}
            for u, path in frontier.items():
                for v, kind in self._neighbours(u):
                    if v in seen:
                        continue
                    npath = path + [(v, kind)]
                    if v == u2:
                        ks = [e for _, e in npath if e]
                        return {'kind': ks[0] if ks else 'sameAs',
                                'path': npath}
                    seen.add(v)
                    nxt[v] = npath
            if not nxt:
                break
            frontier = nxt
        return None


# ============================================================ skeleton parsing
def parse_llm_a(text):
    """Skeleton triples + slot roles from raw LLM-A output."""
    parsed = RF._parse_llm_a(text)
    skeleton = RF._parse_skeleton(text)
    props = parsed.get('properties', {}) or {}
    classes = parsed.get('classes', {}) or {}
    entities = parsed.get('entities', {}) or {}

    triples = []
    body = skeleton[skeleton.find('{') + 1:skeleton.rfind('}')] \
        if '{' in skeleton else skeleton
    for line in body.splitlines():
        parts = RF.split_triple_line(line)
        if parts is None:
            continue
        s, p, o = parts
        if p == 'a':
            triples.append({'s': s, 'p': 'a', 'o': o, 'kind': 'class',
                            'slot': o})
        else:
            triples.append({'s': s, 'p': p, 'o': o, 'kind': 'prop',
                            'slot': p})

    # The projected variables: what the QUESTION asks for.  The compiler
    # needs them to write the SELECT clause, and they are also the ground
    # truth for which slots are answers as opposed to join vocabulary.
    select_vars = []
    m = re.search(r'SELECT\s+(.*?)\s+WHERE', skeleton, re.S | re.I)
    if m:
        head = re.sub(r'\bDISTINCT\b', '', m.group(1), flags=re.I)
        for v in re.findall(r'\?[A-Za-z0-9_]+', head):
            if v not in select_vars:
                select_vars.append(v)

    return {'triples': triples, 'props': props, 'classes': classes,
            'entities': entities, 'select_vars': select_vars}


def _local_name(uri):
    """'http://.../ontology/Settlement' -> 'Settlement'."""
    u = uri.split('#')[-1]
    u = u.rstrip('/').split('/')[-1]
    return u


def _class_matches(cls_uri, entity_class):
    """Does a class IRI denote the same class as the partition slug?"""
    if not cls_uri or not entity_class:
        return False
    return _local_name(cls_uri).lower() == entity_class.lower()


class PredicateRanges:
    """What kind of object each predicate points at, per KG.

    Cosine retrieval matches on language, so it ranked dbo:genre above
    skos:subject for "links a film to its subject category" -- "genre" is a
    natural-language match for "category".  But a shared variable bound to a
    Category: URI can only be produced by a predicate that actually points
    at categories, and the dump says exactly one does: skos:subject
    (10,925,705 entity->category triples; skos:broader points at categories
    but only from other categories).

    Without this check the solver produced a plan binding ?shared_category to
    Category:Heroin while reaching it through dbo:genre -- a plan no endpoint
    can execute.
    """

    def __init__(self, path=RANGES_FILE):
        if not os.path.exists(path):
            raise RuntimeError('%s not found -- run build_predicate_ranges.py'
                               % path)
        self.ranges = json.load(open(path, encoding='utf-8'))

    def points_at_category(self, kg, pred_uri):
        """True if this predicate points at a Category: resource from an entity."""
        row = self.ranges.get(kg, {}).get(pred_uri)
        return bool(row) and row[0] > 0

    def category_capable(self, kg):
        """Every predicate in `kg` that can point at a category from an entity."""
        return {p for p, v in self.ranges.get(kg, {}).items() if v[0] > 0}


class ClassPredicates:
    """Which predicates actually fire on the instances of each class.

    Endpoint locality (C1) is necessary but not sufficient: it rejects a
    dbpedia predicate paired with an lmdb entity, yet happily accepts
    dbpedia:birthDate paired with a dbpedia FILM -- which also cannot
    execute.

    Type applicability closes that gap.  A predicate can only fire on an
    entity if that entity's class has the predicate, and the partition
    knows exactly which predicates each class has:

        dbpedia    227 classes, 24 predicates/class
        lmdb        51 classes,  6 predicates/class
        drugbank     7 classes, 19 predicates/class

    An lmdb film has 48 usable predicates and dbo:birthDate is not among
    them, so the wrong endpoint cannot survive the gate.
    """

    def __init__(self, path=CLASSPRED_FILE):
        if not os.path.exists(path):
            raise RuntimeError('%s not found -- run '
                               'build_class_predicate_table.py' % path)
        self.table = json.load(open(path, encoding='utf-8'))

    def allows(self, kg, class_slug, pred_uri):
        """May this predicate fire on an entity of this class in this KG?"""
        if not class_slug:
            return True          # unknown class: cannot rule it out
        preds = self.table.get(kg, {}).get(class_slug)
        if preds is None:
            return True          # class not in the table: cannot rule it out
        return pred_uri in preds


class Liveness:
    """Does this endpoint actually HAVE this triple?

    TYPE APPLICABILITY is a class-level statement: "a `film` may have a
    `country`".  But the dump is finer than that.  The particular film
    Scream has no `country` triple in lmdb, so a query that asks for it
    there matches zero triples -- and a federated query whose SERVICE
    block matches zero triples eliminates the whole solution.

    This is the endpoint-level check: the (entity, predicate) pair must
    actually occur in that endpoint's dump.

    It decides the cases the class-level check cannot.  For F06 the
    question asks for the country and the language of Scream; both are
    legal on the class `film`, so type applicability says nothing.  But
    Scream has no `country` in lmdb, only in dbpedia -- so the answer is
    only executable if `country` is served by dbpedia.

    The index is the dump itself, so this is a measurement, not a
    heuristic.  Pairs that are absent are rejected; a subject the index
    does not know about is not rejected (absence of evidence).
    """

    def __init__(self, index=None):
        self._index = index
        self._loaded = False

    def _idx(self):
        if not self._loaded:
            self._loaded = True
            if self._index is None:
                try:
                    import triple_index as TI
                    self._index = TI.TripleIndex()
                except Exception:
                    self._index = None
        return self._index

    def fires(self, kg, subject_uri, pred_uri):
        """Can this triple fire?  True when unknown (cannot rule it out)."""
        if not subject_uri or not pred_uri:
            return True
        idx = self._idx()
        if idx is None:
            return True
        r = idx.has(kg, subject_uri, pred_uri)
        if r is None:
            return True          # no index for this KG: cannot rule it out
        return r


# ===================================================================== solver
class ConstraintSolver:
    def __init__(self, bridge=None, ranges=None, classpred=None, top_k=10,
                 entity_k=8, max_realizations=8, liveness=None):
        self.bridge = bridge or Bridge()
        self.ranges = ranges or PredicateRanges()
        self.classpred = classpred or ClassPredicates()
        self.liveness = liveness if liveness is not None else Liveness()
        self.top_k = top_k
        self.entity_k = entity_k
        self.max_realizations = max_realizations
        self._names = {}          # entity slot -> the name from LLM-A
        # how many sameAs edges point INTO each KG -- the hub is the join
        # target, not the place a name is fetched from
        self._hub_kg = collections.Counter()
        for u, ps in self.bridge.sameas.items():
            for p, kg in ps:
                self._hub_kg[kg] += 1

    # ------------------------------------------------------------- retrieval
    def _candidates(self, parsed):
        """Per-KG property / class / entity candidates for every slot."""
        text = '```json\n%s\n```' % json.dumps({
            'properties': parsed['props'], 'classes': parsed['classes'],
            'entities': parsed['entities']})
        ret = RF.retrieve(text, top_k=self.top_k)
        props_by_kg = ret['properties_by_kg']
        classes_by_kg = ret['classes_by_kg']

        ents = {}
        for var, name in parsed['entities'].items():
            name = (name or '').strip()
            if not name:
                continue
            per_kg = {}
            for kg in RE.KGS:
                hits = RE.retrieve_entity(name, kg, top_k=self.entity_k,
                                          expand_sameas=False)
                per_kg[kg] = hits
            ents[var] = per_kg
        return props_by_kg, classes_by_kg, ents

    # ------------------------------------------------------------- realising
    def _anchor_kg(self, var, ents, props_by_kg=None, allowed=None):
        """The KG an entity is fetched BY NAME in.

        A federated query says "?x rdfs:label 'The Rock'" in ONE endpoint;
        the other endpoints reach ?x through owl:sameAs.  So the anchor is
        the endpoint whose LABEL LITERAL the query would contain -- the KG
        whose label matches the name character for character, because that
        is the string the question used.

        Character-for-character matters: dbpedia drops the disambiguator
        suffix, so the question's "Leslie Howard (Actor)" is lmdb's label
        while dbpedia's is "Leslie Howard (actor)".  A case-insensitive
        match would anchor in the wrong endpoint.

        `allowed` restricts the choice to the endpoints this partition
        actually uses.  That matters: the anchor is WHERE THE NAME IS
        FETCHED, so it must be an endpoint the query touches.  Choosing it
        globally and forcing it into every partition made a single-endpoint
        query (P02's dbpedia-only Abraham Lincoln) demand a cross-endpoint
        sameAs edge that does not exist, so every plan died.

        When several KGs match exactly (P02's "Abraham Lincoln" is both a
        dbpedia politician and an lmdb film), the tie is broken by CONTENT:
        the anchor has to be an endpoint that can actually serve the
        question's predicates.  dbpedia's OfficeHolder has dbpedia:party and
        dbpedia:orderInOffice; lmdb's film has neither.
        """
        per_kg = ents.get(var) or {}
        name = self._names.get(var)
        kgs = [kg for kg in RE.KGS if allowed is None or kg in allowed]

        exact = []
        for kg in kgs:
            for h in per_kg.get(kg, ()):
                if h.get('label') == name:
                    exact.append((kg, h))
        if not exact:
            # no exact label: fall back to the strongest hit
            best = None
            for kg in kgs:
                for h in per_kg.get(kg, ()):
                    if best is None or h['score'] > best['score']:
                        best = h
            return None if best is None else best['kg']

        # rank the exact matches by how well their KG can serve the content,
        # then by hub-ness: the hub is where every sameAs edge points INTO
        # (dbpedia 32,245, lmdb 0, drugbank 0), so it is the join TARGET,
        # reached through sameAs, not the place a name is fetched from.
        scored = []
        for kg, h in exact:
            support = self._content_support(kg, h.get('class'), props_by_kg)
            scored.append((-support, self._hub_kg.get(kg, 0), kg))
        scored.sort()
        return scored[0][2]

    def _content_support(self, kg, cls, props_by_kg):
        """How many of the question's property slots can this KG serve for an
        entity of this class?  Used to break anchor ties."""
        if not props_by_kg or not cls:
            return 0
        n = 0
        for var, by_kg in props_by_kg.items():
            hits = by_kg.get(kg) or []
            if not hits:
                continue
            if any(self.classpred.allows(kg, cls, h['uri']) for h in hits):
                n += 1
        return n

    def _servable_elsewhere(self, slot, subj, pick, kg, props_by_kg):
        """Could another endpoint serve this slot for this entity?

        The slot did not fire in `kg`, but the entity may live in another
        endpoint too.  If one of those endpoints DOES have the fact, then
        `kg` is simply the wrong endpoint for it, and the partition should be
        rejected so the solver can place the slot in the endpoint that has
        it.  This is what distinguishes "wrong endpoint" from "the entity
        does not have this fact at all".
        """
        per_kg = pick.get(subj) or {}
        for kg2, r2 in per_kg.items():
            if kg2 == kg:
                continue
            uri = r2.get('uri')
            if not uri:
                continue
            for h in props_by_kg.get(slot, {}).get(kg2, []):
                if self.liveness.fires(kg2, uri, h['uri']):
                    return True
        return False

    def _realisations(self, name, needed_kgs, per_kg, anchor_kg=None):
        """Ways to realise one logical entity in every needed KG.

        The entity is ANCHORED in one KG -- the one whose label matches the
        name exactly, and the one the query fetches it by -- and reaches the
        other KGs through sameAs.  Every realisation is therefore
        sameAs-linked by construction: that IS constraint C2a, enforced
        during generation rather than checked afterwards.

        The anchor matters: a federated query says "?x rdfs:label 'The Rock'"
        in ONE endpoint, and every other endpoint reaches ?x through
        owl:sameAs.  Without the anchor the solver happily realises an
        entity in dbpedia alone, which would collapse a 2-KG question into a
        1-KG one.
        """
        out = []
        roots = list(needed_kgs)
        if anchor_kg and anchor_kg in needed_kgs:
            # The anchor is tried FIRST, but not exclusively: the anchor is
            # where the query fetches the entity BY NAME, and the label
            # literal binds that URI -- but the entity's other KGs are
            # reached through sameAs, so a realisation rooted elsewhere is
            # still valid as long as the anchor's URI is one of its members.
            roots = [anchor_kg] + [k for k in roots if k != anchor_kg]
        for root in roots:
            hits = per_kg.get(root) or []
            if not hits:
                continue
            best = max((h['score'] for h in hits), default=0.0) or 1.0
            for h in hits:
                real = {root: {'uri': h['uri'], 'label': h['label'],
                               'class': h.get('class'),
                               'score': h['score'] / best, 'source': h['source']}}
                ok = True
                for g in needed_kgs:
                    if g == root:
                        continue
                    ps = self.bridge.partner_uris(h['uri'], g)
                    if not ps:
                        ok = False
                        break
                    # the counterpart lives in ANOTHER KG, so its class is
                    # that KG's class -- "actor" in lmdb is "Actor" in
                    # dbpedia.  Without this the type check would be skipped
                    # for every sameAs realisation, which is exactly where
                    # it matters most.
                    partner_cls = None
                    try:
                        partner_cls = _kg_state()['kgs'][g].class_of_uri(ps[0])
                    except Exception:
                        partner_cls = None
                    real[g] = {'uri': ps[0], 'label': None,
                               'class': partner_cls,
                               'score': (h['score'] / best) * 0.9,
                               'source': 'sameAs'}
                if ok:
                    out.append(real)
        out.sort(key=lambda r: -sum(v['score'] for v in r.values()))
        return out[:self.max_realizations]

    # -------------------------------------------------------------- planning
    def solve(self, llm_a_output, parsed=None):
        parsed = parsed or parse_llm_a(llm_a_output)
        triples = parsed['triples']
        props_by_kg, classes_by_kg, ents = self._candidates(parsed)
        self._names = {v: (n or '').strip()
                       for v, n in parsed['entities'].items()}

        stats = {'partitions': 0, 'partitions_invalid_c1': 0,
                 'realisations_rejected_c2a': 0, 'plans': 0,
                 'plans_rejected_c2b': 0, 'plans_rejected_class': 0,
                 'plans_rejected_type': 0}

        if not triples:
            return {'plans': [], 'stats': stats, 'parsed': parsed}

        # --- group triples that share a slot: they must share a KG (C1)
        parent = list(range(len(triples)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        slot_owner = {}
        for i, t in enumerate(triples):
            key = t['slot']
            if key in slot_owner:
                union(slot_owner[key], i)
            else:
                slot_owner[key] = i
        blocks = collections.defaultdict(list)
        for i in range(len(triples)):
            blocks[find(i)].append(i)

        # --- the shared free variable: how two entities get joined (C2b)
        var_triples = collections.defaultdict(list)
        for i, t in enumerate(triples):
            for v in (t['s'], t['o']):
                var_triples[v].append(i)
        free_vars = set(parsed['entities'])
        for v in parsed['props']:
            free_vars.add(v)
        for v in parsed['classes']:
            free_vars.add(v)
        shared_vars = []
        for v, idxs in var_triples.items():
            if v in free_vars or len(idxs) < 2:
                continue
            ents_in = []
            for i in idxs:
                for w in (triples[i]['s'], triples[i]['o']):
                    if w in parsed['entities']:
                        ents_in.append(w)
            if len(set(ents_in)) >= 2:
                shared_vars.append((v, sorted(set(ents_in)), sorted(idxs)))

        # property slots whose OBJECT is a shared variable -- these are the
        # slots that must be filled by a category-capable predicate
        shared_slots = set()
        for v, ent_vars2, idxs in shared_vars:
            for i in idxs:
                t = triples[i]
                if t['kind'] == 'prop' and t['o'] == v:
                    shared_slots.add(t['slot'])

        # --- candidate KGs per block
        block_kgs = []
        for root, idxs in sorted(blocks.items()):
            kinds = {triples[i]['kind'] for i in idxs}
            slots = {triples[i]['slot'] for i in idxs}
            options = []
            for kg in RE.KGS:
                ok = True
                for s in slots:
                    if triples[idxs[0]]['kind'] == 'class':
                        table = classes_by_kg.get(s, {})
                    else:
                        table = props_by_kg.get(s, {})
                    if not table.get(kg):
                        ok = False
                        break
                if ok:
                    options.append(kg)
            # a class triple may also be dropped outright: the type triple is
            # optional (it is a constraint, not content)
            opts = list(options)
            if kinds == {'class'}:
                opts.append(None)
            block_kgs.append({'root': root, 'idxs': idxs, 'slots': slots,
                              'kind': triples[idxs[0]]['kind'], 'options': opts})

        # --- enumerate KG partitions
        plans = []
        for combo in itertools.product(*[b['options'] for b in block_kgs]):
            stats['partitions'] += 1
            assign = {}
            for b, kg in zip(block_kgs, combo):
                if kg is None:
                    for i in b['idxs']:
                        assign[i] = None
                else:
                    for i in b['idxs']:
                        assign[i] = kg
            live = [i for i, kg in assign.items() if kg]

            # entity realisations, per C2a
            need = collections.defaultdict(set)
            for i in live:
                for v in (triples[i]['s'], triples[i]['o']):
                    if v in parsed['entities']:
                        need[v].add(assign[i])
            # The ANCHOR KG is always in the query: the entity is fetched by
            # name there, even if no content triple lives there.  But the
            # anchor must be an endpoint THIS partition uses -- a query that
            # only touches dbpedia cannot fetch its entity by name in lmdb.
            for v in list(need):
                ak = self._anchor_kg(v, ents, props_by_kg,
                                     allowed=need[v] | set(assign.values()))
                if ak:
                    need[v].add(ak)
            real = {}
            bad = False
            for v, kgs in need.items():
                rs = self._realisations(parsed['entities'][v], sorted(kgs),
                                        ents.get(v, {}),
                                        anchor_kg=self._anchor_kg(
                                            v, ents, props_by_kg,
                                            allowed=kgs))
                if not rs:
                    stats['realisations_rejected_c2a'] += 1
                    bad = True
                    break
                real[v] = rs
            if bad:
                continue

            # enumerate entity combinations and check C2b
            ent_vars = sorted(real)
            for combo_r in itertools.product(*[real[v] for v in ent_vars]):
                pick = dict(zip(ent_vars, combo_r))
                ok = True
                bridges = []
                shared_bind = {}
                for v, ent_vars2, idxs in shared_vars:
                    if len(ent_vars2) < 2:
                        continue
                    kg = assign[idxs[0]]
                    if kg is None:
                        continue
                    # every triple that uses the shared variable must be in the
                    # same KG, and that KG must own the bridging value
                    if any(assign[i] != kg for i in idxs):
                        ok = False
                        break
                    u1 = pick.get(ent_vars2[0], {}).get(kg)
                    u2 = pick.get(ent_vars2[1], {}).get(kg)
                    if not u1 or not u2:
                        ok = False
                        break
                    u1 = u1['uri']
                    u2 = u2['uri']
                    if u1 == u2:
                        shared_bind[v] = None
                        bridges.append({'kind': 'same', 'entities': [u1, u2],
                                        'via': None})
                        continue
                    cat = self.bridge.best_shared_category(u1, u2)
                    if cat is None:
                        ok = False
                        break
                    cat_uri, nc, ng = cat
                    # the shared value is a dbpedia resource, so the triple
                    # that binds it has to be served by dbpedia
                    if kg != 'dbpedia':
                        ok = False
                        break
                    # RANGE CHECK: the predicate reaching the shared value
                    # must actually point at a category.  Cosine ranked
                    # dbo:genre above skos:subject here, which would have
                    # produced an inexecutable plan.
                    slot = triples[idxs[0]]['slot']
                    pred_hits = props_by_kg.get(slot, {}).get(kg, [])
                    if not any(self.ranges.points_at_category(kg, h['uri'])
                               for h in pred_hits):
                        ok = False
                        break
                    shared_bind[v] = cat_uri
                    bridges.append({'kind': 'category', 'entities': [u1, u2],
                                    'via': cat_uri, 'n_candidates': nc,
                                    'n_members': ng})
                if not ok:
                    stats['plans_rejected_c2b'] += 1
                    continue

                # score the plan
                score = 0.0
                props = {}
                classes = {}
                # slots whose fact does not exist for this entity: written as
                # OPTIONAL so the answerable part of the question survives
                optional_slots = set()

                # TYPE APPLICABILITY: the predicate must be one that can
                # actually fire on the subject's class in that endpoint.
                # This is checked before scoring, so an illegal predicate
                # never enters the plan.
                for b in block_kgs:
                    kg = combo[block_kgs.index(b)]
                    if kg is None or b['kind'] != 'prop':
                        continue
                    for s in b['slots']:
                        hits = props_by_kg.get(s, {}).get(kg, [])
                        # a content slot never binds the join vocabulary
                        hits = [h for h in hits
                                if h['uri'] not in JOIN_ONLY_PREDICATES]
                        if s not in shared_slots:
                            # skos:subject is content only for a shared
                            # category slot
                            hits = [h for h in hits
                                    if h['uri'] != SKOS_SUBJECT]
                        if s in shared_slots:
                            hits = [h for h in hits
                                    if self.ranges.points_at_category(kg, h['uri'])]
                        # the subject of the first triple using this slot
                        subj = None
                        for i in b['idxs']:
                            if triples[i]['slot'] == s:
                                subj = triples[i]['s']
                                break
                        subj_uri = None
                        if subj in pick and kg in pick[subj]:
                            cls = pick[subj][kg].get('class')
                            subj_uri = pick[subj][kg].get('uri')
                            if cls:
                                hits = [h for h in hits
                                        if self.classpred.allows(kg, cls, h['uri'])]
                        # LIVENESS: the triple must actually occur in this
                        # endpoint for THIS entity.  Type applicability says
                        # the predicate is legal on the class; the dump says
                        # whether it is there.  Only the second one makes
                        # the SERVICE block match anything, and a block that
                        # matches nothing eliminates the whole solution.
                        #
                        # The predicate is NOT re-ranked here: the top
                        # candidate is the one the slot MEANS, and liveness
                        # decides whether this ENDPOINT can serve it.
                        # Re-ranking by liveness would silently swap
                        # `deathDate` for `birthDate` just because the
                        # entity happens to have a birth date.
                        if subj_uri and hits:
                            if not self.liveness.fires(kg, subj_uri,
                                                       hits[0]['uri']):
                                # This endpoint cannot serve this slot.  Can
                                # ANOTHER one?  If so, this partition is
                                # wrong and the solver will find the endpoint
                                # that has the fact.  If no endpoint can,
                                # the entity simply does not have it (A09's
                                # Peter O'Toole has no deathDate), and the
                                # slot is optional so the answerable part of
                                # the question still gets answered.
                                if self._servable_elsewhere(
                                        s, subj, pick, kg, props_by_kg):
                                    ok = False
                                    break
                                optional_slots.add(s)
                        if hits:
                            props[s] = dict(hits[0], kg=kg)
                        else:
                            # no legal predicate for this slot in this KG:
                            # the plan cannot execute, so drop it
                            ok = False
                            break
                    if not ok:
                        stats['plans_rejected_type'] += 1
                        continue

                for b in block_kgs:
                    kg = combo[block_kgs.index(b)]
                    if kg is None:
                        continue
                    for s in b['slots']:
                        if b['kind'] == 'class':
                            hits = classes_by_kg.get(s, {}).get(kg, [])
                            if hits:
                                classes[s] = dict(hits[0], kg=kg)
                                score += hits[0]['score']
                        else:
                            h = props.get(s)
                            if h:
                                score += h['score']
                for v, r in pick.items():
                    score += sum(x['score'] for x in r.values())
                # the class must agree with the entity's own class
                for b in block_kgs:
                    if b['kind'] != 'class':
                        continue
                    kg = combo[block_kgs.index(b)]
                    if kg is None:
                        continue
                    s = list(b['slots'])[0]
                    cls_hits = classes_by_kg.get(s, {}).get(kg, [])
                    if not cls_hits:
                        continue
                    # the class slot must name the SAME class as the entity
                    # it constrains -- this is what makes the class slot a
                    # constraint rather than a label
                    best = None
                    for i in b['idxs']:
                        subj = triples[i]['s']
                        if subj not in pick or kg not in pick[subj]:
                            continue
                        ent_cls = pick[subj][kg].get('class')
                        if not ent_cls:
                            continue
                        for h in cls_hits:
                            if _class_matches(h['uri'], ent_cls):
                                if best is None or h['score'] > best['score']:
                                    best = h
                    if best is None:
                        # no class candidate names the entity's class -> this
                        # partition is not executable, drop it
                        ok = False
                        break
                    classes[s] = dict(best, kg=kg)
                    score += best['score']
                if not ok:
                    stats['plans_rejected_class'] += 1
                    continue

                kgs = sorted({g for g in assign.values() if g})
                plans.append({
                    'kg_of_triple': {str(i): assign[i] for i in assign},
                    'kgs': kgs,
                    'properties': props,
                    'classes': classes,
                    'entities': {v: pick[v] for v in pick},
                    'bridges': bridges,
                    'shared_var_bindings': shared_bind,
                    'optional_slots': sorted(optional_slots),
                    'score': round(score, 4),
                })

        plans.sort(key=lambda p: -p['score'])
        stats['plans'] = len(plans)
        return {'plans': plans[:20], 'stats': stats, 'parsed': parsed}


# ======================================================================== main
if __name__ == '__main__':
    llm_a_text = """
```sparql
SELECT ?indication ?drug_category WHERE {
  ?trainspotting a ?film_type .
  ?heroin a ?drug_type .
  ?trainspotting ?subject_category ?shared_category .
  ?heroin ?subject_category ?shared_category .
  ?heroin ?has_indication ?indication .
  ?heroin ?has_drug_category ?drug_category .
}
```

```json
{
  "properties": {
    "?subject_category": "links a film or drug (subject) to a subject category associated with it (object)",
    "?has_indication": "links a drug (subject) to the medical condition for which it is indicated (object)",
    "?has_drug_category": "links a drug (subject) to its drug category or classification (object)"
  },
  "classes": {
    "?film_type": "film or motion picture",
    "?drug_type": "pharmaceutical or recreational drug"
  },
  "entities": {
    "?trainspotting": "Trainspotting",
    "?heroin": "Heroin"
  }
}
```
"""
    solver = ConstraintSolver()
    res = solver.solve(llm_a_text)
    print('stats:', res['stats'])
    for n, p in enumerate(res['plans'][:3]):
        print()
        print('--- plan %d  score=%.3f  kgs=%s' % (n + 1, p['score'], p['kgs']))
        for v, r in p['entities'].items():
            for kg, h in r.items():
                print('    entity %-16s %-9s %s' % (v, kg, h['uri']))
        for v, h in p['properties'].items():
            print('    prop   %-16s %-9s %s' % (v, h['kg'], h['uri']))
        for v, h in p['classes'].items():
            print('    class  %-16s %-9s %s' % (v, h['kg'], h['uri']))
        for v, u in p['shared_var_bindings'].items():
            print('    shared %-16s -> %s' % (v, u))
        for b in p['bridges']:
            print('    bridge %s via %s' % (b['kind'], b.get('via')))
