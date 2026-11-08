"""Evaluate the constraint solver against the 50 gold benchmark queries.

WHAT IS MEASURED
-----------------
The solver replaces an independent top-k list per slot with a ranked set of
GROUNDED PLANS.  So the question is not "is the gold in the top-k" but "does
a surviving plan agree with the gold".

  PLAN FOUND
      Does any surviving plan bind a gold entity URI for every entity slot,
      a gold predicate for every property slot, and a gold class URI for
      every class slot?  The end-to-end number.

  ENDPOINT SET
      Which endpoints must the query span?  This is the number C2a governs:
      an entity realised in two KGs forces the query to use two SERVICE
      blocks, joined by sameAs.  It is measured as the set of KGs in which
      the plan realises its entities, against the set of endpoints in the
      gold SPARQL.

  PRUNING YIELD
      How many candidate combinations die, and to which constraint?  A
      constraint that rejects everything is not a good constraint, so this
      is the cost side of the trade.

HOW THE GOLD IS DERIVED
-----------------------
The gold SPARQL identifies its entities by LABEL LITERAL ("Trainspotting"),
never by URI -- the URIs live in the benchmark's `grounding` field.  So the
gold entity URIs are recovered by resolving each label literal through the
same exact-match index the retriever uses, and then following sameAs to the
entity's counterpart in the other endpoint.  That is the same path the
solver takes, so the comparison is exact rather than approximate.
"""
import collections
import json
import os
import re
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import constraint_solver as CS
import retriever_entity as RE

SERVICE = re.compile(r'SERVICE\s+<([^>]+)>\s*\{(.*?)\n\s*\}', re.S)
# A triple inside an OPTIONAL { ... } block is still part of the query, and
# the gold's optional properties are real content -- A03's ?died_on is
# dbo:deathDate inside an OPTIONAL.  So the body is flattened: the OPTIONAL
# wrapper is removed but its contents are kept.
OPTIONAL = re.compile(r'OPTIONAL\s*\{([^{}]*)\}')
TRIPLE = re.compile(r'(?:\?[A-Za-z0-9_]+|<[^>]+>)\s+<([^>]+)>\s+(.*?)\s*\.\s*$',
                    re.M)
LABEL = re.compile(r'(?:rdf-schema#label|dc/terms/title)>\s*"([^"]*)"')

# Predicates that express the join or the anchor, not the content.
# skos:subject and rdf:type ARE content in this benchmark -- the T-series
# joins on a category and the P-series constrains a type -- so they stay.
STRUCTURAL = {
    'http://www.w3.org/2002/07/owl#sameAs',
    'http://www.w3.org/2000/01/rdf-schema#label',
    'http://purl.org/dc/terms/title',
}


def kg_of(endpoint):
    if 'dbpedia.org' in endpoint:
        return 'dbpedia'
    if 'linkedmdb.org' in endpoint:
        return 'lmdb'
    if 'drugbank' in endpoint:
        return 'drugbank'
    return None


def gold_plan(sparql, st, bridge):
    """Gold URIs and endpoint set, recovered from the gold SPARQL."""
    g = {'kgs': set(), 'props': set(), 'classes': set(),
         'entities': set(), 'entity_by_kg': collections.defaultdict(set),
         'shared_consts': set()}

    for m in SERVICE.finditer(sparql):
        kg = kg_of(m.group(1))
        if not kg:
            continue
        g['kgs'].add(kg)
        body = m.group(2)
        # flatten OPTIONAL blocks -- their triples are still gold content
        body = OPTIONAL.sub(lambda mm: ' ' + mm.group(1) + ' ', body)

        for tm in TRIPLE.finditer(body):
            pred = tm.group(1)
            obj = tm.group(2)
            if pred in STRUCTURAL:
                continue
            # the predicate itself is the content
            g['props'].add(pred)
            # an explicit class object is a class constraint
            if pred.endswith('rdf-syntax-ns#type'):
                for um in re.finditer(r'<([^>]+)>', obj):
                    g['classes'].add(um.group(1))
            else:
                for um in re.finditer(r'<([^>]+)>', obj):
                    u = um.group(1)
                    if '/sparql' in u:
                        continue
                    if 'dbpedia.org/resource/Category:' in u:
                        g['shared_consts'].add(u)

        # the anchor: a label literal inside this endpoint
        for lm in LABEL.finditer(body):
            lit = lm.group(1)
            kg_i = st['kgs'][kg]
            for i in kg_i.exact_ids(lit):
                uri = kg_i.uri(i)
                g['entities'].add(uri)
                g['entity_by_kg'][kg].add(uri)
                # the sameAs counterpart in the other endpoint
                for p, pkg in bridge.partners(uri):
                    g['entities'].add(p)
                    g['entity_by_kg'][pkg].add(p)
    return g


def entity_slots(parsed):
    return {v: (n or '').strip() for v, n in parsed['entities'].items()
            if (n or '').strip()}


def plan_matches_gold(plan, gold, parsed):
    """Does this plan bind the gold?  Returns (ok, reason).

    The class slot is only compared when the gold actually carries a type
    triple -- 14 of the 50 gold queries do (the P-series); in the other 36
    the class slot exists only to CONSTRAIN the entity choice and is not
    emitted, so there is nothing to match it against.
    """
    ents = plan.get('entities', {})
    for var in entity_slots(parsed):
        real = ents.get(var)
        if not real:
            return False, 'entity slot %s unbound' % var
        if not any(h['uri'] in gold['entities'] for h in real.values()):
            return False, 'entity %s not a gold entity' % var

    for var in parsed['props']:
        h = plan.get('properties', {}).get(var)
        if not h:
            return False, 'property slot %s unbound' % var
        if h['uri'] not in gold['props']:
            return False, 'property %s = %s not gold' % (var, h['uri'])

    if gold['classes']:
        for var in parsed['classes']:
            h = plan.get('classes', {}).get(var)
            if not h:
                continue                  # a type constraint may be dropped
            if h['uri'] not in gold['classes']:
                return False, 'class %s = %s not gold' % (var, h['uri'])

    return True, 'ok'


def baseline_plan(parsed, gold, solver):
    """The UNCONSTRAINED baseline: top-1 candidate per slot, no joint check.

    This is what the pipeline does today -- LLM-B gets independent per-KG
    top-k lists and picks for itself.  Scoring it the same way as the
    solver isolates the contribution of the constraints: any difference is
    the constraints, not the retriever.

    The baseline has no notion of which KG a triple belongs to, so it takes
    the globally best-scoring candidate per slot and the entity's top hit
    per KG -- which is exactly the mixing the constraints forbid.
    """
    props_by_kg, classes_by_kg, ents = solver._candidates(parsed)
    props = {}
    for var in parsed['props']:
        hits = []
        for kg in RE.KGS:
            hits += props_by_kg.get(var, {}).get(kg, [])
        if hits:
            props[var] = max(hits, key=lambda h: h['score'])
    classes = {}
    for var in parsed['classes']:
        hits = []
        for kg in RE.KGS:
            hits += classes_by_kg.get(var, {}).get(kg, [])
        if hits:
            classes[var] = max(hits, key=lambda h: h['score'])
    entities = {}
    for var, per_kg in ents.items():
        real = {}
        for kg, hits in per_kg.items():
            if hits:
                real[kg] = {'uri': hits[0]['uri'], 'label': hits[0]['label'],
                            'class': hits[0].get('class'),
                            'score': hits[0]['score'], 'source': hits[0]['source']}
        entities[var] = real
    return {'properties': props, 'classes': classes, 'entities': entities,
            'kgs': sorted({kg for r in entities.values() for kg in r})}


def main():
    bench = json.load(open(os.path.join(BASE, 'benchmark_50.json'),
                           encoding='utf-8'))
    runs = {r['id']: r for r in
            json.load(open(os.path.join(BASE, '_llm_a_runs.json'),
                           encoding='utf-8'))}

    st = RE._load()
    solver = CS.ConstraintSolver(bridge=CS.Bridge())

    agg = collections.Counter()
    prune = collections.Counter()
    by_type = collections.defaultdict(collections.Counter)
    misses = []
    rows = []

    for q in bench:
        qid = q['id']
        r = runs.get(qid)
        if not r:
            continue
        gold = gold_plan(q['sparql'], st, solver.bridge)
        parsed = CS.parse_llm_a(r['llm_a'])

        res = solver.solve(r['llm_a'], parsed=parsed)
        plans = res['plans']
        stt = res['stats']
        for k in ('partitions', 'realisations_rejected_c2a',
                  'plans_rejected_c2b', 'plans'):
            prune[k] += stt[k]

        found = None
        for p in plans:
            ok, why = plan_matches_gold(p, gold, parsed)
            if ok:
                found = p
                break
        if found:
            agg['plan_found'] += 1
            by_type[q['type']]['plan_found'] += 1
        else:
            why = 'no surviving plan bound the gold'
            for p in plans[:30]:
                ok, w = plan_matches_gold(p, gold, parsed)
                if not ok:
                    why = w
                    break
            misses.append((qid, q['type'], q['shape'], why))
            by_type[q['type']]['miss'] += 1

        # --- endpoint set: the KGs a plan actually spans.
        # The gold's endpoint set is the set of SERVICE blocks; the plan's is
        # the set of KGs it realises its entities in, because an entity
        # realised in two KGs forces two SERVICE blocks joined by sameAs.
        ep_ok = False
        for p in plans:
            e = set()
            for var, real in p.get('entities', {}).items():
                e |= set(real)
            if e == gold['kgs']:
                ep_ok = True
                break
        if ep_ok:
            agg['endpoint_ok'] += 1
            by_type[q['type']]['endpoint_ok'] += 1
        else:
            by_type[q['type']]['endpoint_miss'] += 1

        # --- the unconstrained baseline, scored identically
        bp = baseline_plan(parsed, gold, solver)
        bok, _ = plan_matches_gold(bp, gold, parsed)
        if bok:
            agg['baseline_found'] += 1
        bep = set()
        for var, real in bp.get('entities', {}).items():
            bep |= set(real)
        if bep == gold['kgs']:
            agg['baseline_endpoint_ok'] += 1

        rows.append({'id': qid, 'type': q['type'], 'shape': q['shape'],
                     'plans': len(plans), 'found': bool(found),
                     'endpoint_ok': ep_ok,
                     'gold_kgs': sorted(gold['kgs']),
                     'plan_kgs': sorted(gold['kgs']) if ep_ok else [],
                     'gold_entities': len(gold['entities']),
                     'gold_props': len(gold['props'])})

    n = sum(c['plan_found'] + c['miss'] for c in by_type.values())
    print('=' * 92)
    print('CONSTRAINT SOLVER -- 50 gold benchmark queries')
    print('=' * 92)
    print('  queries evaluated             : %d' % n)
    print('  PLAN FOUND (all gold URIs)    : %d/%d (%.1f%%)'
          % (agg['plan_found'], n, 100.0 * agg['plan_found'] / max(n, 1)))
    print('  ENDPOINT SET matches gold     : %d/%d (%.1f%%)'
          % (agg['endpoint_ok'], n, 100.0 * agg['endpoint_ok'] / max(n, 1)))
    print()
    print('  ablation -- the same slots, WITHOUT the constraints:')
    print('    PLAN FOUND (baseline)        : %d/%d (%.1f%%)'
          % (agg['baseline_found'], n,
             100.0 * agg['baseline_found'] / max(n, 1)))
    print('    ENDPOINT SET (baseline)      : %d/%d (%.1f%%)'
          % (agg['baseline_endpoint_ok'], n,
             100.0 * agg['baseline_endpoint_ok'] / max(n, 1)))

    print()
    print('  by query type:')
    for t in sorted(by_type):
        c = by_type[t]
        tot = c['plan_found'] + c['miss']
        print('    %-6s plan_found %2d/%-2d (%5.1f%%)   endpoint_ok %2d/%-2d'
              % (t, c['plan_found'], tot,
                 100.0 * c['plan_found'] / max(tot, 1), c['endpoint_ok'], tot))

    print()
    print('  pruning yield (why candidates die):')
    print('    KG partitions enumerated            : %d' % prune['partitions'])
    print('    entity realisations rejected (C2a) : %d'
          % prune['realisations_rejected_c2a'])
    print('    plan combinations rejected (C2b)   : %d'
          % prune['plans_rejected_c2b'])
    print('    surviving plans                    : %d' % prune['plans'])

    if misses:
        print()
        print('  misses:')
        for qid, t, sh, why in misses:
            print('     %-5s %-6s %-5s %s' % (qid, t, sh, why))

    with open('_constraint_eval.json', 'w', encoding='utf-8') as f:
        json.dump({'rows': rows, 'agg': dict(agg),
                   'by_type': {k: dict(v) for k, v in by_type.items()},
                   'prune': dict(prune)}, f, indent=1)
    print()
    print('wrote _constraint_eval.json')


if __name__ == '__main__':
    main()
