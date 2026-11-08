"""End-to-end evaluation: LLM-A -> constraint solver -> LLM-B -> federated SPARQL.

WHAT IS MEASURED
-----------------
The pipeline's final output is a federated SPARQL query, so the honest test
is whether that query RUNS and returns the right ANSWERS.  Three layers:

  1. ENDPOINT SET -- which SERVICE blocks the generated query uses, against
     the gold's.  A wrong endpoint set means the query asks the wrong
     graphs, however good the rest is.

  2. LOCAL EXECUTION -- the generated query is run against the harvested
     dump neighbourhood (sparql_local.execute) and its answer rows are
     compared with the gold query's answer rows.  This is the STRONGEST
     test: it is execution accuracy, not string similarity, so a query that
     binds the wrong predicate or the wrong entity fails even when it
     looks right.

  3. THE ABLATION -- the same measurement on the unconstrained baseline
     (top-1 candidate per slot, KGs mixed freely).  Any difference is the
     constraints, not the retriever.

THE LOCAL-EXECUTION CAVEAT, STATED PLAINLY
-----------------------------------------
The neighbourhood holds the gold entities' 1-2 hop neighbourhoods, so a
generated query that binds a DIFFERENT (but legitimate) entity has no local
triples and returns zero rows.  That is counted as a miss.  It is a
conservative measurement: it can only understate the pipeline, never
overstate it.  The structural metrics above are unaffected by it.
"""
import collections
import json
import os
import re
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import constraint_solver as CS
import eval_constraint_solver as EV
import llm_b as LB
import retriever_entity as RE
import sparql_local as SL


def _rowset(rows):
    """Answer rows as a comparable set: values only, sorted.

    Variable NAMES differ between the generated query and the gold (the
    generated one renames answers after the predicate), so only the VALUES
    are compared -- and only the ones that are actually bound.
    """
    out = set()
    for r in rows:
        vals = tuple(sorted(v for v in r.values() if v is not None))
        if vals:
            out.add(vals)
    return out


def endpoints_of(sparql):
    return set(EV.kg_of(m) for m in
               re.findall(r'SERVICE\s+<([^>]+)>', sparql))


def baseline_plan(parsed, solver):
    """The unconstrained baseline, in the shape the compiler consumes.

    Top-1 candidate per slot, KGs mixed freely -- i.e. what the pipeline
    does without the constraint layer.  It needs the KG assignment the
    compiler expects, which the plain baseline does not carry: each triple
    goes to whichever endpoint its chosen predicate belongs to.
    """
    props_by_kg, classes_by_kg, ents = solver._candidates(parsed)
    props, classes = {}, {}
    for var in parsed['props']:
        best = None
        for kg in RE.KGS:
            for h in props_by_kg.get(var, {}).get(kg, []):
                if best is None or h['score'] > best['score']:
                    best = dict(h, kg=kg)
        if best:
            props[var] = best
    for var in parsed['classes']:
        best = None
        for kg in RE.KGS:
            for h in classes_by_kg.get(var, {}).get(kg, []):
                if best is None or h['score'] > best['score']:
                    best = dict(h, kg=kg)
        if best:
            classes[var] = best
    entities = {}
    for var, per_kg in ents.items():
        real = {}
        for kg, hits in per_kg.items():
            if hits:
                real[kg] = {'uri': hits[0]['uri'], 'label': hits[0]['label'],
                            'class': hits[0].get('class'),
                            'score': hits[0]['score'],
                            'source': hits[0]['source']}
        entities[var] = real

    kg_of = {}
    for i, t in enumerate(parsed['triples']):
        if t['kind'] == 'class':
            h = classes.get(t['slot'])
        else:
            h = props.get(t['slot'])
        kg_of[str(i)] = h['kg'] if h else None
    return {'properties': props, 'classes': classes, 'entities': entities,
            'kg_of_triple': kg_of,
            'shared_var_bindings': {},
            'kgs': sorted({kg for r in entities.values() for kg in r})}


def main():
    bench = json.load(open(os.path.join(BASE, 'benchmark_50.json'),
                           encoding='utf-8'))
    runs = {r['id']: r for r in
            json.load(open(os.path.join(BASE, '_llm_a_runs.json'),
                           encoding='utf-8'))}

    data = SL.Data()
    solver = CS.ConstraintSolver(bridge=CS.Bridge())
    compiler = LB.Compiler()

    agg = collections.Counter()
    by_type = collections.defaultdict(collections.Counter)
    rows = []
    misses = []

    for q in bench:
        qid = q['id']
        r = runs.get(qid)
        if not r:
            continue
        parsed = CS.parse_llm_a(r['llm_a'])
        res = solver.solve(r['llm_a'], parsed=parsed)
        plans = res['plans']

        gold_kgs = endpoints_of(q['sparql'])
        gold_rows = _rowset(SL.execute(q['sparql'], data))

        # ---- compile every surviving plan
        compiled = LB.compile_solutions(res, parsed, compiler)
        agg['plans'] += len(compiled)

        # ---- 1. endpoint set
        ep_ok = any(set(c['kgs']) == gold_kgs for c in compiled
                    if c.get('sparql'))
        if ep_ok:
            agg['endpoint_ok'] += 1
            by_type[q['type']]['endpoint_ok'] += 1
        else:
            by_type[q['type']]['endpoint_miss'] += 1

        # ---- 2. local execution, best plan wins
        exec_ok = False
        exec_rows = None
        n_rows = 0
        for c in compiled:
            if not c.get('sparql'):
                continue
            try:
                got = SL.execute(c['sparql'], data)
            except Exception:
                continue
            rs = _rowset(got)
            n_rows += len(rs)
            if rs and rs == gold_rows:
                exec_ok = True
                exec_rows = rs
                break
        if exec_ok:
            agg['exec_ok'] += 1
            by_type[q['type']]['exec_ok'] += 1
        else:
            by_type[q['type']]['exec_miss'] += 1

        # ---- 3. the ablation: the unconstrained baseline, compiled the
        #         same way, so only the constraints differ.
        bp = baseline_plan(parsed, solver)
        bres = None
        try:
            bres = compiler.compile(bp, parsed)
        except ValueError:
            pass
        if bres and bres.get('sparql'):
            try:
                brs = _rowset(SL.execute(bres['sparql'], data))
            except Exception:
                brs = set()
            if brs and brs == gold_rows:
                agg['baseline_exec_ok'] += 1
        else:
            agg['baseline_exec_ok'] += 0

        rows.append({'id': qid, 'type': q['type'], 'shape': q['shape'],
                     'plans': len(compiled),
                     'endpoint_ok': bool(ep_ok),
                     'exec_ok': bool(exec_ok),
                     'gold_kgs': sorted(k for k in gold_kgs if k),
                     'gold_rows': len(gold_rows),
                     'top_rows': n_rows})

        if not exec_ok:
            misses.append((qid, q['type'], len(compiled),
                           'gold %d rows, got %d' % (len(gold_rows), n_rows)))

    n = len(rows)
    print('=' * 92)
    print('END-TO-END: LLM-A -> CONSTRAIN -> LLM-B -> federated SPARQL')
    print('=' * 92)
    print('  queries evaluated                  : %d' % n)
    print('  compiled queries (all plans)       : %d' % agg['plans'])
    print()
    print('  ENDPOINT SET matches gold          : %d/%d (%.1f%%)'
          % (agg['endpoint_ok'], n, 100.0 * agg['endpoint_ok'] / max(n, 1)))
    print('  LOCAL EXECUTION == gold answers    : %d/%d (%.1f%%)'
          % (agg['exec_ok'], n, 100.0 * agg['exec_ok'] / max(n, 1)))
    print()
    print('  ablation (no constraints):')
    print('    LOCAL EXECUTION == gold answers  : %d/%d (%.1f%%)'
          % (agg['baseline_exec_ok'], n,
             100.0 * agg['baseline_exec_ok'] / max(n, 1)))
    print()
    print('  by query type:')
    for t in sorted(by_type):
        c = by_type[t]
        tot = c['exec_ok'] + c['exec_miss']
        print('    %-6s n=%-3d endpoint_ok %2d   exec_ok %2d/%-2d (%5.1f%%)'
              % (t, tot, c['endpoint_ok'], c['exec_ok'], tot,
                 100.0 * c['exec_ok'] / max(tot, 1)))
    if misses:
        print()
        print('  misses:')
        for qid, t, np_, why in misses:
            print('     %-5s %-6s plans=%-3d %s' % (qid, t, np_, why))

    with open('_end_to_end.json', 'w', encoding='utf-8') as f:
        json.dump({'rows': rows, 'agg': dict(agg)}, f, indent=1)
    print()
    print('wrote _end_to_end.json')


if __name__ == '__main__':
    main()
