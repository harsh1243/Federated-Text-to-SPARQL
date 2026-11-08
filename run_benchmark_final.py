"""FINAL BENCHMARK RUN -- end to end, with the full metric set.

    question -> LLM-A -> retrieve -> CONSTRAIN -> compile -> federated SPARQL
             -> execute -> answer rows -> compare with gold

This is the harness that produces the numbers for the results report.  It is
the same pipeline the project has always run (constraint_solver.solve ->
llm_b.compile_solutions -> sparql_local.execute), but it is written to
produce a complete, per-query, machine-checkable record so the report can be
regenerated from one artefact.

WHAT IS MEASURED, per question
------------------------------
  plan_found      a surviving plan binds every gold URI            (plan level)
  endpoint_set    some plan's SERVICE set equals the gold's        (structural)
  exec_em         some plan executes to EXACTLY the gold's rows     (execution)
  exec_r0         RANK 0 (top-scored plan) executes to the gold's rows
  exec_f1         set-level F1 of the best plan's rows vs gold's
  r0_f1           set-level F1 of rank 0's rows vs gold's
  executable      at least one plan is fully executable in the dump
  baseline_*      the same, for the UNCONSTRAINED ablation
  baseline_f1     the ablation's F1

Row comparison
--------------
Rows are compared by their VALUES (the generated query renames answer
variables after the predicate it binds, so variable names cannot be
compared).  Two row-set forms are reported:

  value-set   the sorted tuple of non-null values in a row -- the comparison
              the project's existing evals use, and the one the headline
              numbers are quoted against;
  strict      the value tuple in the gold's own SELECT order -- a stricter
              variant that additionally penalises a query that binds the
              right values under a different projection.

F1 is the standard set F1 over rows:  P = |P n G| / |P|,  R = |P n G| / |G|,
F1 = 2PR/(P+R), with the usual 1.0 convention when both sets are empty and
0.0 when exactly one is empty.

Usage:
    python run_benchmark_final.py --runs _llm_a_live_claude_opus_5_5.json \
                                  --out _bench_final_opus.json
"""
import argparse
import collections
import json
import os
import re
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import constraint_solver as CS
import eval_constraint_solver as EV
import eval_end_to_end as E2E
import llm_b as LB
import retriever_entity as RE
import sparql_local as SL


# --------------------------------------------------------------------- rows
def value_rows(rows):
    """Rows as sorted tuples of non-null values (the project's convention)."""
    out = set()
    for r in rows:
        vals = tuple(sorted(v for v in r.values() if v is not None))
        if vals:
            out.add(vals)
    return out


def strict_rows(rows, order):
    """Rows as value tuples in a fixed variable order."""
    out = set()
    for r in rows:
        vals = tuple(r.get(v) for v in order)
        if all(v is not None for v in vals):
            out.add(vals)
    return out


def f1(pred, gold):
    """Set-level P/R/F1 over answer rows."""
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    if not pred or not gold:
        return 0.0, 0.0, 0.0
    inter = len(pred & gold)
    if inter == 0:
        return 0.0, 0.0, 0.0
    p = inter / len(pred)
    r = inter / len(gold)
    return p, r, 2 * p * r / (p + r)


def endpoints_of(sparql):
    return set(EV.kg_of(m) for m in
               re.findall(r'SERVICE\s+<([^>]+)>', sparql))


def select_order(sparql):
    m = re.search(r'SELECT\s+(.*?)\s+WHERE', sparql, re.S | re.I)
    if not m:
        return []
    head = re.sub(r'\bDISTINCT\b', '', m.group(1), flags=re.I)
    return re.findall(r'\?[A-Za-z0-9_]+', head)


def baseline_plan(parsed, solver):
    """The unconstrained ablation: top-1 per slot, graphs mixed freely.

    Delegates to the project's own implementation so this harness and the
    published 0/50 baseline are literally the same code path.
    """
    return E2E.baseline_plan(parsed, solver)


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default='_llm_a_runs.json')
    ap.add_argument('--bench', default='benchmark_50.json')
    ap.add_argument('--out', default='_bench_final.json')
    ap.add_argument('--label', default='')
    args = ap.parse_args()

    bench = json.load(open(os.path.join(BASE, args.bench), encoding='utf-8'))
    if isinstance(bench, dict):
        bench = bench['queries']

    blob = json.load(open(os.path.join(BASE, args.runs), encoding='utf-8'))
    if isinstance(blob, dict) and 'runs' in blob:
        runs = {r['id']: r for r in blob['runs']}
        run_meta = {k: v for k, v in blob.items() if k != 'runs'}
    else:
        runs = {r['id']: r for r in blob}
        run_meta = {}

    print('=' * 92)
    print('FINAL BENCHMARK   runs=%s   model=%s   questions=%d'
          % (args.runs, run_meta.get('model', 'n/a'), len(bench)))
    print('=' * 92)

    data = SL.Data()
    solver = CS.ConstraintSolver(bridge=CS.Bridge())
    compiler = LB.Compiler()
    st = RE._load()

    rows_out = []
    agg = collections.Counter()
    by_type = collections.defaultdict(collections.Counter)
    micro = collections.Counter()

    for q in bench:
        qid = q['id']
        r = runs.get(qid)
        rec = {'id': qid, 'type': q['type'], 'shape': q['shape'],
               'question': q['question'],
               'gold_kgs': sorted(k for k in q.get('kgs', []) if k)}

        if not r or not r.get('llm_a'):
            rec['error'] = 'no LLM-A output'
            rows_out.append(rec)
            agg['llm_missing'] += 1
            continue

        parsed = CS.parse_llm_a(r['llm_a'])
        rec['n_slots'] = {
            'properties': len(parsed['props']),
            'classes': len(parsed['classes']),
            'entities': len(parsed['entities']),
            'triples': len(parsed['triples']),
        }

        res = solver.solve(r['llm_a'], parsed=parsed)
        compiled = LB.compile_solutions(res, parsed, compiler)
        rec['n_plans_raw'] = res['stats']['plans']
        rec['n_plans'] = len(compiled)
        rec['stats'] = res['stats']
        agg['plans_raw'] += res['stats']['plans']
        agg['plans'] += len(compiled)

        # ---- gold
        gold_rows = SL.execute(q['sparql'], data)
        g_vals = value_rows(gold_rows)
        g_strict = strict_rows(gold_rows, select_order(q['sparql']))
        rec['gold_rows'] = len(g_vals)

        # ---- plan level: does a surviving plan bind every gold URI?
        try:
            gold_plan = EV.gold_plan(q['sparql'], st, solver.bridge)
            found = False
            for p in res['plans']:
                ok, _why = EV.plan_matches_gold(p, gold_plan, parsed)
                if ok:
                    found = True
                    break
        except Exception:
            found = None
        rec['plan_found'] = found
        agg['plan_found'] += 1 if found else 0

        # ---- structural: endpoint set
        ep_ok = any(set(c['kgs']) == set(rec['gold_kgs']) for c in compiled
                    if c.get('sparql'))
        rec['endpoint_ok'] = bool(ep_ok)
        agg['endpoint_ok'] += 1 if ep_ok else 0

        # ---- execution, every plan
        best = None          # (f1, em, rows, plan)
        rank0 = None
        any_exec = False
        exec_plans = 0
        for i, c in enumerate(compiled):
            if not c.get('sparql'):
                continue
            try:
                got = SL.execute(c['sparql'], data)
            except Exception:
                continue
            gv = value_rows(got)
            gs = strict_rows(got, select_order(q['sparql']))
            p, rr, f = f1(gv, g_vals)
            em = (gv == g_vals)
            sem = (gs == g_strict)
            if gv:
                exec_plans += 1
            if em:
                any_exec = True
            if i == 0:
                rank0 = {'rows': len(gv), 'em': em, 'strict_em': sem,
                         'p': p, 'r': rr, 'f1': f,
                         'kgs': c['kgs'], 'sparql': c['sparql']}
            if best is None or f > best['f1'] or (f == best['f1'] and em
                                                 and not best['em']):
                best = {'rank': i, 'rows': len(gv), 'em': em,
                        'strict_em': sem, 'p': p, 'r': rr, 'f1': f,
                        'kgs': c['kgs']}
        rec['exec_em'] = any_exec
        rec['exec_plans_with_rows'] = exec_plans
        agg['exec_em'] += 1 if any_exec else 0
        if best:
            agg['best_f1_sum'] += best['f1']
            agg['best_p_sum'] += best['p']
            agg['best_r_sum'] += best['r']
            micro['best_inter'] += 0  # filled below from per-query counts
            rec['best'] = best
        if rank0:
            agg['r0_em'] += 1 if rank0['em'] else 0
            agg['r0_f1_sum'] += rank0['f1']
            agg['r0_rows_nonempty'] += 1 if rank0['rows'] else 0
            rec['rank0'] = rank0

        # ---- executable plan (every triple present in the dump)
        rec['has_executable_plan'] = exec_plans > 0
        agg['has_executable_plan'] += 1 if exec_plans else 0

        # ---- the ablation
        bp = baseline_plan(parsed, solver)
        try:
            bres = compiler.compile(bp, parsed)
        except Exception:
            bres = None
        b_rows = []
        if bres and bres.get('sparql'):
            try:
                b_rows = SL.execute(bres['sparql'], data)
            except Exception:
                b_rows = []
        bv = value_rows(b_rows)
        bp_, br_, bf_ = f1(bv, g_vals)
        rec['baseline'] = {'rows': len(bv), 'em': bv == g_vals and bool(bv),
                           'p': bp_, 'r': br_, 'f1': bf_,
                           'kgs': bres.get('kgs') if bres else []}
        agg['baseline_em'] += 1 if rec['baseline']['em'] else 0
        agg['baseline_f1_sum'] += bf_
        agg['baseline_p_sum'] += bp_
        agg['baseline_r_sum'] += br_
        rec['baseline']['endpoint_ok'] = (
            set(bres.get('kgs', [])) == set(rec['gold_kgs']) if bres else False)
        agg['baseline_endpoint_ok'] += 1 if rec['baseline']['endpoint_ok'] else 0

        # ---- per-family bookkeeping
        bt = by_type[q['type']]
        bt['n'] += 1
        bt['plan_found'] += 1 if found else 0
        bt['endpoint_ok'] += 1 if ep_ok else 0
        bt['exec_em'] += 1 if any_exec else 0
        bt['r0_em'] += 1 if (rank0 and rank0['em']) else 0
        bt['exec_plan'] += 1 if exec_plans else 0
        bt['baseline_em'] += 1 if rec['baseline']['em'] else 0
        bt['best_f1'] += best['f1'] if best else 0.0
        bt['r0_f1'] += rank0['f1'] if rank0 else 0.0
        bt['plans'] += len(compiled)

        rows_out.append(rec)

    n = len(rows_out)
    scored = [r for r in rows_out if 'best' in r or 'rank0' in r]

    # ---- macro / micro F1 over the best plan
    def macro(key):
        vals = [r[key] for r in rows_out if key in r]
        return (sum(vals) / len(vals)) if vals else 0.0

    def macro_family(t, key):
        vals = []
        for r in rows_out:
            if r['type'] != t or key not in r:
                continue
            vals.append(r[key])
        return (sum(vals) / len(vals)) if vals else 0.0

    summary = {
        'n_questions': n,
        'n_scored': len(scored),
        'model': run_meta.get('model'),
        'runs_file': args.runs,
        'label': args.label,
        'run_meta': run_meta,
        'plans_raw': agg['plans_raw'],
        'plans_compiled': agg['plans'],
        'plan_found': agg['plan_found'],
        'endpoint_ok': agg['endpoint_ok'],
        'exec_em': agg['exec_em'],
        'exec_em_pct': 100.0 * agg['exec_em'] / max(n, 1),
        'has_executable_plan': agg['has_executable_plan'],
        'r0_em': agg['r0_em'],
        'r0_em_pct': 100.0 * agg['r0_em'] / max(n, 1),
        'r0_rows_nonempty': agg['r0_rows_nonempty'],
        'best_f1_macro': macro_best_f1(rows_out),
        'best_p_macro': macro_best_p(rows_out),
        'best_r_macro': macro_best_r(rows_out),
        'r0_f1_macro': macro_r0_f1(rows_out),
        'baseline_em': agg['baseline_em'],
        'baseline_em_pct': 100.0 * agg['baseline_em'] / max(n, 1),
        'baseline_f1_macro': macro_baseline_f1(rows_out),
        'baseline_endpoint_ok': agg['baseline_endpoint_ok'],
        'baseline_p_macro': macro_baseline_p(rows_out),
        'baseline_r_macro': macro_baseline_r(rows_out),
        'by_type': {t: dict(c) for t, c in by_type.items()},
    }

    payload = {'summary': summary, 'rows': rows_out}
    with open(os.path.join(BASE, args.out), 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)

    # ---- console
    print()
    print('  questions evaluated             : %d' % n)
    print('  LLM-A failures (no output)       : %d' % agg['llm_missing'])
    print('  plans surviving (raw / capped)   : %d / %d'
          % (agg['plans_raw'], agg['plans']))
    print()
    print('  PLAN FOUND (all gold URIs bound) : %d/%d (%.1f%%)'
          % (agg['plan_found'], n, 100.0 * agg['plan_found'] / max(n, 1)))
    print('  ENDPOINT SET matches gold        : %d/%d (%.1f%%)'
          % (agg['endpoint_ok'], n, 100.0 * agg['endpoint_ok'] / max(n, 1)))
    print('  EXEC EXACT MATCH (some plan)     : %d/%d (%.1f%%)'
          % (agg['exec_em'], n, 100.0 * agg['exec_em'] / max(n, 1)))
    print('  EXEC EXACT MATCH (rank 0 only)   : %d/%d (%.1f%%)'
          % (agg['r0_em'], n, 100.0 * agg['r0_em'] / max(n, 1)))
    print('  >=1 EXECUTABLE plan              : %d/%d (%.1f%%)'
          % (agg['has_executable_plan'], n,
             100.0 * agg['has_executable_plan'] / max(n, 1)))
    print()
    print('  F1  best plan (macro)            : %.4f' % summary['best_f1_macro'])
    print('  P   best plan (macro)            : %.4f' % summary['best_p_macro'])
    print('  R   best plan (macro)            : %.4f' % summary['best_r_macro'])
    print('  F1  rank 0 (macro)               : %.4f' % summary['r0_f1_macro'])
    print()
    print('  ABLATION (unconstrained)')
    print('    EXEC EXACT MATCH               : %d/%d (%.1f%%)'
          % (agg['baseline_em'], n, 100.0 * agg['baseline_em'] / max(n, 1)))
    print('    ENDPOINT SET                   : %d/%d (%.1f%%)'
          % (agg['baseline_endpoint_ok'], n,
             100.0 * agg['baseline_endpoint_ok'] / max(n, 1)))
    print('    F1 (macro)                     : %.4f'
          % summary['baseline_f1_macro'])
    print()
    print('  by family:')
    for t in sorted(by_type):
        c = by_type[t]
        tot = c['n']
        print('    %-5s n=%2d  plan %2d  ep %2d  exec %2d  r0 %2d  execplan %2d  '
              'F1 %.3f  base %d'
              % (t, tot, c['plan_found'], c['endpoint_ok'], c['exec_em'],
                 c['r0_em'], c['exec_plan'],
                 c['best_f1'] / max(tot, 1), c['baseline_em']))
    print()
    print('wrote %s' % args.out)


# ------------------------------------------------------------- macro helpers
def macro_best_f1(rows):
    vals = [r['best']['f1'] for r in rows if 'best' in r]
    return sum(vals) / len(vals) if vals else 0.0


def macro_best_p(rows):
    vals = [r['best']['p'] for r in rows if 'best' in r]
    return sum(vals) / len(vals) if vals else 0.0


def macro_best_r(rows):
    vals = [r['best']['r'] for r in rows if 'best' in r]
    return sum(vals) / len(vals) if vals else 0.0


def macro_r0_f1(rows):
    vals = [r['rank0']['f1'] for r in rows if 'rank0' in r]
    return sum(vals) / len(vals) if vals else 0.0


def macro_baseline_f1(rows):
    vals = [r['baseline']['f1'] for r in rows if 'baseline' in r]
    return sum(vals) / len(vals) if vals else 0.0


def macro_baseline_p(rows):
    vals = [r['baseline']['p'] for r in rows if 'baseline' in r]
    return sum(vals) / len(vals) if vals else 0.0


def macro_baseline_r(rows):
    vals = [r['baseline']['r'] for r in rows if 'baseline' in r]
    return sum(vals) / len(vals) if vals else 0.0


if __name__ == '__main__':
    main()
