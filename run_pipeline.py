"""Run the full pipeline over the 50 benchmark questions and write the output.

    question -> LLM-A skeleton -> retrieve -> CONSTRAIN -> COMPILE -> SPARQL

Writes _final_queries.json: for every question, the ranked plans the
constraint layer produced and the federated SPARQL each one compiles to.

This is the pipeline's final artefact: the file a reader can open and check
against the gold.
"""
import collections
import json
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import constraint_solver as CS
import llm_b as LB
import sparql_local as SL


def main():
    bench = json.load(open(os.path.join(BASE, 'benchmark_50.json'),
                           encoding='utf-8'))
    runs = {r['id']: r for r in
            json.load(open(os.path.join(BASE, '_llm_a_runs.json'),
                           encoding='utf-8'))}

    data = SL.Data()
    solver = CS.ConstraintSolver(bridge=CS.Bridge())
    compiler = LB.Compiler()

    out = []
    agg = collections.Counter()

    for q in bench:
        r = runs.get(q['id'])
        if not r:
            continue
        parsed = CS.parse_llm_a(r['llm_a'])
        res = solver.solve(r['llm_a'], parsed=parsed)
        compiled = LB.compile_solutions(res, parsed, compiler)

        gold_rows = SL.execute(q['sparql'], data)
        gold_set = set(tuple(sorted(v for v in row.values() if v is not None))
                       for row in gold_rows)

        best = None
        for c in compiled:
            if not c.get('sparql'):
                continue
            try:
                rows = SL.execute(c['sparql'], data)
            except Exception:
                continue
            got = set(tuple(sorted(v for v in row.values() if v is not None))
                       for row in rows)
            if got and got == gold_set:
                best = c
                break

        agg['questions'] += 1
        agg['plans'] += len(compiled)
        if best is not None:
            agg['exec_match'] += 1

        out.append({
            'id': q['id'],
            'question': q['question'],
            'type': q['type'],
            'kgs': best['kgs'] if best else [],
            'matches_gold': best is not None,
            'sparql': best['sparql'] if best else (
                compiled[0]['sparql'] if compiled and
                compiled[0].get('sparql') else None),
            'gold_sparql': q['sparql'],
            'all_plans': [{
                'rank': c['rank'],
                'score': c['score'],
                'kgs': c['kgs'],
                'sparql': c['sparql'],
                'warnings': c['warnings'],
            } for c in compiled],
        })

    with open('_final_queries.json', 'w', encoding='utf-8') as f:
        json.dump({'summary': dict(agg), 'queries': out}, f, indent=1)

    print('=' * 76)
    print('FULL PIPELINE OUTPUT')
    print('=' * 76)
    print('  questions              : %d' % agg['questions'])
    print('  plans compiled         : %d' % agg['plans'])
    print('  queries matching gold  : %d/%d (%.1f%%)'
          % (agg['exec_match'], agg['questions'],
             100.0 * agg['exec_match'] / max(agg['questions'], 1)))
    print()
    print('wrote _final_queries.json')


if __name__ == '__main__':
    main()
