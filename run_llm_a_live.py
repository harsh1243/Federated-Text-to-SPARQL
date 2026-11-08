"""Run LLM-A LIVE over the 50 benchmark questions.

This is the fresh run: it calls the API, it does not read `_llm_a_runs.json`.
The system prompt is the one extracted verbatim from `llm_A.ipynb`
(`_llm_a_prompt.txt`), so the only thing that changed since the cached run is
the model itself.

Every response is validated before it is accepted:
  * `stop_reason` must be `end_turn` (not `max_tokens` -- the model can spend
    the whole budget on a thinking block, which is what broke 3 calls in the
    first run);
  * the text must contain a ```sparql block and a ```json block;
  * the JSON block must parse;
  * the skeleton must contain at least one triple.

A response that fails any of these is retried (up to 4 attempts) with a
larger token budget.  The result is written to `_llm_a_live_<modeltag>.json`
with the same shape as `_llm_a_runs.json` plus per-call provenance.

Usage:
    python run_llm_a_live.py --model claude-opus-5-5
    python run_llm_a_live.py --model claude-sonnet-5-5
"""
import argparse
import concurrent.futures as cf
import io
import json
import os
import re
import sys
import threading
import time

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

PROMPT_FILE = os.path.join(BASE, '_llm_a_prompt.txt')

SPARQL_BLOCK = re.compile(r'```sparql\s*(.*?)```', re.S | re.I)
JSON_BLOCK = re.compile(r'```json\s*(.*?)```', re.S | re.I)


def read_prompt():
    with io.open(PROMPT_FILE, encoding='utf-8') as f:
        return f.read()


def validate(text):
    """Return (ok, reason).

    The acceptance test is the project's own parser: if `parse_llm_a` turns
    the response into a skeleton with at least one triple, the response is
    usable by the pipeline.  Anything less specific than that (counting
    tokens by hand) invents failure modes the pipeline does not have.
    """
    if not text or not text.strip():
        return False, 'empty response'
    m = SPARQL_BLOCK.search(text)
    j = JSON_BLOCK.search(text)
    if not m:
        return False, 'no ```sparql block'
    if not j:
        return False, 'no ```json block'
    try:
        obj = json.loads(j.group(1))
    except Exception as e:
        return False, 'json does not parse: %s' % str(e)[:80]
    if not isinstance(obj, dict):
        return False, 'json block is not an object'
    body = m.group(1)
    if '{' not in body or '}' not in body:
        return False, 'sparql block has no WHERE body'
    try:
        import constraint_solver as CS
        parsed = CS.parse_llm_a(text)
    except Exception as e:
        return False, 'pipeline parser failed: %s' % str(e)[:80]
    if not parsed['triples']:
        return False, 'skeleton has no triple patterns'
    return True, ''


def call_api(client, model, system, question, max_tokens, temperature=0.0):
    """One API call.  Returns (text, stop_reason, usage).

    `temperature` defaults to 0.  Note that this does NOT make the endpoint
    deterministic: measured on this box, 6 identical calls at temperature=0
    produced 6 distinct responses (and 6 distinct responses at the default
    temperature too).  The proxy in front of the API does not implement
    faithful greedy decoding.

    Determinism for a benchmark comes from FREEZING the output, not from the
    temperature: the JSON file this script writes is the artefact, and its
    SHA-256 is recorded so anyone can verify they are running the same one.
    """
    kw = {}
    if temperature is not None:
        kw['temperature'] = temperature
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{'role': 'user', 'content': question}],
        **kw
    )
    text = ''.join(b.text for b in msg.content if b.type == 'text')
    usage = None
    try:
        usage = {'input_tokens': msg.usage.input_tokens,
                 'output_tokens': msg.usage.output_tokens}
    except Exception:
        pass
    return text, msg.stop_reason, usage


def run_one(client, model, system, q, max_tokens, attempts, tag,
            temperature=0.0):
    """Call until the response validates.  Returns a provenance dict."""
    last = None
    for a in range(1, attempts + 1):
        try:
            text, stop, usage = call_api(client, model, system,
                                         q['question'], max_tokens, temperature)
        except Exception as e:
            last = {'attempt': a, 'error': '%s: %s' % (type(e).__name__,
                                                       str(e)[:160])}
            time.sleep(min(2 ** a, 20))
            continue

        if stop == 'max_tokens':
            last = {'attempt': a, 'error': 'truncated (stop_reason=max_tokens)',
                    'usage': usage}
            max_tokens = min(max_tokens * 2, 32000)
            time.sleep(1)
            continue

        ok, why = validate(text)
        if ok:
            return {'id': q['id'], 'question': q['question'],
                    'llm_a': text, 'model': model, 'tag': tag,
                    'attempts': a, 'max_tokens_final': max_tokens,
                    'stop_reason': stop, 'usage': usage, 'error': None}
        last = {'attempt': a, 'error': why, 'usage': usage,
                'stop_reason': stop}
        time.sleep(1)
    return {'id': q['id'], 'question': q['question'],
            'llm_a': None, 'model': model, 'tag': tag,
            'attempts': attempts, 'stop_reason': None, 'usage': None,
            'error': (last or {}).get('error', 'unknown')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=os.environ.get('ANTHROPIC_MODEL',
                                                       'claude-opus-5-5'))
    ap.add_argument('--max-tokens', type=int, default=4000)
    ap.add_argument('--attempts', type=int, default=4)
    ap.add_argument('--workers', type=int, default=5)
    ap.add_argument('--temperature', type=float, default=0.0,
                    help='sampling temperature; 0 is best-effort only -- this '
                         'endpoint is NOT deterministic even at 0')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    import anthropic
    client = anthropic.Anthropic(
        api_key=os.environ.get('ANTHROPIC_API_KEY'),
        base_url=os.environ.get('ANTHROPIC_BASE_URL'),
    )
    system = read_prompt()
    bench = json.load(open(os.path.join(BASE, 'benchmark_50.json'),
                           encoding='utf-8'))
    if isinstance(bench, dict):
        bench = bench['queries']

    tag = re.sub(r'[^A-Za-z0-9]+', '_', args.model).strip('_').lower()
    out_path = args.out or os.path.join(BASE, '_llm_a_live_%s.json' % tag)

    print('=' * 78)
    print('LIVE LLM-A RUN   model=%s   questions=%d   workers=%d'
          % (args.model, len(bench), args.workers))
    print('=' * 78)

    lock = threading.Lock()
    done = [0]
    t0 = time.time()

    def work(q):
        r = run_one(client, args.model, system, q, args.max_tokens,
                    args.attempts, tag, temperature=args.temperature)
        with lock:
            done[0] += 1
            n = done[0]
            ok = r['llm_a'] is not None
            print('  [%2d/%2d] %-5s %s%s   %.0fs'
                  % (n, len(bench), r['id'],
                     'ok' if ok else 'FAIL: ' + str(r['error']),
                     ' (attempts %d)' % r['attempts'] if r['attempts'] > 1 else '',
                     time.time() - t0))
        return r

    results = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(work, bench):
            results.append(r)

    results.sort(key=lambda r: r['id'])
    ok_n = sum(1 for r in results if r['llm_a'] is not None)
    payload = {
        'model': args.model,
        'tag': tag,
        'temperature': args.temperature,
        'n': len(results),
        'n_ok': ok_n,
        'n_failed': len(results) - ok_n,
        'wall_seconds': round(time.time() - t0, 1),
        'runs': results,
    }
    with io.open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)

    # The SHA-256 of the LLM-A OUTPUTS (not the file) is what a reviewer
    # verifies: the artefact is the frozen model output, and this hash pins
    # it.  File-level hashing would also change on a reformat.
    import hashlib
    blob = '\n\x1e\n'.join(
        (r['llm_a'] or '') for r in results).encode('utf-8')
    digest = hashlib.sha256(blob).hexdigest()
    payload['output_sha256'] = digest
    with io.open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)

    print()
    print('  succeeded %d/%d   failed %d   wall %.0fs'
          % (ok_n, len(results), len(results) - ok_n, time.time() - t0))
    for r in results:
        if r['llm_a'] is None:
            print('    FAILED %s: %s' % (r['id'], r['error']))
    print()
    print('  LLM-A output sha256 : %s' % digest)
    print('  (the frozen artefact is the reproducibility guarantee -- this')
    print('   endpoint is not deterministic even at temperature=0)')
    print()
    print('wrote %s' % out_path)


if __name__ == '__main__':
    main()
