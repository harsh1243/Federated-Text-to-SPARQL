"""
Minimal SPARQL evaluator over the harvested local neighbourhood.

Supports the subset used by the benchmark:
  - SERVICE <ep> { triple patterns }   (multiple SERVICE blocks, cross-block vars)
  - triple patterns: ?s <pred> ?o / ?s <pred> "lit" / ?s a <Class> / <lit> <pred> ?o
  - FILTER ( ?var <op> <literal> )    numeric / date comparison
  - OPTIONAL { ... }
  - UNION between two group graph patterns
  - DISTINCT
  - ORDER BY ?var [ASC|DESC]  + LIMIT n
  - aggregate: COUNT(?v) AS ?n  (single group)

This lets us execute every generated query against local dumps and get the
real answer rows -- so the benchmark's gold answers are verified, not asserted.
"""
import json, re, os, collections

BASE = os.path.dirname(os.path.abspath(__file__))

EP2KG = {
    'https://dbpedia.org/sparql': 'dbpedia',
    'http://data.linkedmdb.org/sparql': 'lmdb',
    'http://www4.wiwiss.fu-berlin.de/drugbank/sparql': 'drugbank',
}

XSD = 'http://www.w3.org/2001/XMLSchema#'
NUMERIC = {'integer', 'int', 'long', 'double', 'float', 'decimal', 'gYear'}


class Data:
    def __init__(self):
        self.n = json.load(open(os.path.join(BASE, '_nbhd.json'), encoding='utf-8'))
        self.labels = json.load(open(os.path.join(BASE, '_nbhd_labels.json'), encoding='utf-8'))
        self.index = {}
        for kg in ('dbpedia', 'lmdb', 'drugbank'):
            d = collections.defaultdict(list)
            for s, rows in self.n[kg].items():
                for p, o in rows:
                    d[(s, p.strip('<>'))].append(o)
            self.index[kg] = d

    # ---------- value helpers ----------
    def literal(self, o):
        """Parse a raw N-Triples object into (kind, value, datatype)."""
        o = o.strip()
        if o.startswith('"'):
            m = re.match(r'"((?:[^"\\]|\\.)*)"(?:\^\^<([^>]+)>|@([A-Za-z\-]+))?', o)
            if not m:
                return ('lit', o, None)
            return ('lit', m.group(1), m.group(2) or m.group(3))
        if o.startswith('<') and o.endswith('>'):
            return ('iri', o[1:-1], None)
        return ('lit', o, None)

    def resolve(self, term, row, kg):
        """Resolve a query term (var/literal/iri) to a set of concrete values."""
        if term.startswith('?'):
            return row.get(term, set())
        if term.startswith('<') and term.endswith('>'):
            return {term[1:-1]}
        if term.startswith('"'):
            kind, val, dt = self.literal(term)
            return {val}
        if term.startswith('a ') or term == 'a':
            return set()
        return set()

    def label_of(self, uri):
        return self.labels.get(uri, uri.rsplit('/', 1)[-1].replace('_', ' '))


# ---------------- parsing ----------------

def tokenize_patterns(text):
    """Split a group graph pattern body into triple-pattern strings (handling
    nested braces for OPTIONAL/UNION)."""
    out, depth, cur = [], 0, []
    for ch in text:
        if ch == '{':
            depth += 1
            if depth == 1:
                cur = []
                continue
        elif ch == '}':
            depth -= 1
            if depth == 0:
                out.append(''.join(cur))
                cur = []
                continue
        if depth >= 1:
            cur.append(ch)
    return out


def split_top_level(text, sep):
    """Split on sep at brace/paren/IRI/quote depth 0."""
    parts, depth_p, depth_b, cur = [], 0, 0, []
    in_iri = False
    in_str = False
    esc = False
    i = 0
    while i < len(text):
        ch = text[i]
        if in_str:
            cur.append(ch)
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if in_iri:
            cur.append(ch)
            if ch == '>':
                in_iri = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            cur.append(ch)
            i += 1
            continue
        if ch == '<':
            in_iri = True
            cur.append(ch)
            i += 1
            continue
        if ch == '(':
            depth_p += 1
        elif ch == ')':
            depth_p -= 1
        elif ch == '{':
            depth_b += 1
        elif ch == '}':
            depth_b -= 1
        if depth_p == 0 and depth_b == 0 and text.startswith(sep, i):
            parts.append(''.join(cur))
            cur = []
            i += len(sep)
            continue
        cur.append(ch)
        i += 1
    parts.append(''.join(cur))
    return parts


class Query:
    def __init__(self, sparql, data):
        self.data = data
        self.sparql = sparql
        s = sparql.strip()
        m = re.search(r'SELECT\s+(.*?)\s+WHERE', s, re.S | re.I)
        self.select_raw = m.group(1).strip() if m else '*'
        self.distinct = 'DISTINCT' in self.select_raw.upper()
        self.select_raw = re.sub(r'^DISTINCT\s+', '', self.select_raw, flags=re.I).strip()
        # aggregate
        self.agg = None
        am = re.search(r'\(\s*(COUNT|SUM|MIN|MAX)\s*\(\s*(\?\w+)\s*\)\s+AS\s+(\?\w+)\s*\)',
                       self.select_raw, re.I)
        if am:
            self.agg = (am.group(1).upper(), am.group(2), am.group(3))
            self.select_raw = re.sub(r'\(\s*(COUNT|SUM|MIN|MAX)\s*\(\s*\?\w+\s*\)\s+AS\s+\?\w+\s*\)',
                                     '', self.select_raw, flags=re.I).strip()
        self.select_vars = [v for v in re.findall(r'\?\w+', self.select_raw)]
        self.order = []
        om = re.search(r'ORDER\s+BY\s+(.*?)(?:\s+LIMIT|\s+OFFSET|$)', s, re.S | re.I)
        if om:
            for t in split_top_level(om.group(1), ','):
                t = t.strip()
                dm = re.match(r'(DESC|ASC)\s*\((\?\w+)\)', t, re.I)
                if dm:
                    self.order.append((dm.group(2), dm.group(1).upper() == 'DESC'))
                elif t.startswith('?'):
                    self.order.append((t, False))
        lm = re.search(r'LIMIT\s+(\d+)', s, re.I)
        self.limit = int(lm.group(1)) if lm else None
        # where body
        wm = re.search(r'WHERE\s*\{(.*)\}\s*(?:ORDER|LIMIT|OFFSET|$)', s, re.S | re.I)
        self.body = wm.group(1) if wm else ''
        # split body into service blocks and bare blocks
        self.blocks = self._parse_body(self.body)
        # A FILTER outside every SERVICE block applies to the whole join, so
        # it must be evaluated after all blocks are joined -- not per block.
        self.outer_filters = self._outer_filters()

    def _outer_filters(self):
        """FILTER bodies that sit outside any SERVICE block."""
        residual = self.body
        for kind, _kg, inner in self.blocks:
            if kind == 'service' and inner in residual:
                residual = residual.replace(inner, ' ', 1)
        return filter_spans(residual)

    def _parse_body(self, body):
        blocks = []
        i = 0
        while i < len(body):
            m = re.compile(r'\s*SERVICE\s*<([^>]+)>\s*\{').search(body, i)
            if m:
                j = m.end()
                depth = 1
                while j < len(body) and depth:
                    if body[j] == '{':
                        depth += 1
                    elif body[j] == '}':
                        depth -= 1
                    j += 1
                blocks.append(('service', EP2KG.get(m.group(1), '?'), body[m.end():j - 1]))
                i = j
                continue
            # bare group or triple
            m2 = re.compile(r'\s*\{').search(body, i)
            if m2:
                j = m2.end()
                depth = 1
                while j < len(body) and depth:
                    if body[j] == '{':
                        depth += 1
                    elif body[j] == '}':
                        depth -= 1
                    j += 1
                blocks.append(('bare', None, body[m2.end():j - 1]))
                i = j
                continue
            m3 = re.compile(r'\s*[^.\s][^.]*\.').search(body, i)
            if m3:
                blocks.append(('bare', None, m3.group(0)))
                i = m3.end()
                continue
            break
        return blocks


def parse_triples(text):
    """Return list of (s, p, o) with s/p/o as raw strings, or SPECIAL tuples."""
    text = text.strip()
    out = []
    # handle UNION at this level
    parts = re.split(r'\s+UNION\s+', text, flags=re.I)
    if len(parts) > 1:
        return [('UNION', parse_triples(parts[0]), parse_triples(parts[1]))]
    # handle OPTIONAL
    om = re.search(r'OPTIONAL\s*\{', text, re.I)
    if om:
        j = om.end()
        depth = 1
        while j < len(text) and depth:
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
            j += 1
        inner = text[om.end():j - 1]
        before = text[:om.start()]
        return parse_triples(before) + [('OPTIONAL', parse_triples(inner))]

    # Tokenizer: IRI, var, literal, or the 'a' keyword.
    # Order matters: IRI and literal must be matched before bare words.
    TOK = re.compile(
        r'(?P<iri><[^>]*>)'
        r'|(?P<lit>"(?:[^"\\]|\\.)*"(?:\^\^<[^>]+>|@[A-Za-z\-]+)?)'
        r'|(?P<var>\?\w+)'
        r'|(?P<kw>a)(?![\w])'
    )
    out = []
    for st in split_top_level(text, '.'):
        st = st.strip()
        if not st:
            continue
        toks = []
        for m in TOK.finditer(st):
            if m.group('iri'):
                toks.append(m.group('iri'))
            elif m.group('lit'):
                toks.append(m.group('lit'))
            elif m.group('var'):
                toks.append(m.group('var'))
            elif m.group('kw'):
                toks.append('a')
        if len(toks) == 3:
            s, p, o = toks
            if p == 'a':
                p = '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>'
            out.append((s, p, o))
        elif len(toks) == 1 and toks[0].startswith('?'):
            out.append(('BIND', toks[0], None))
    return out


def match_triple(data, kg, s_term, p_term, o_term, row):
    """Extend `row` with all bindings matching one triple pattern.

    A row is {var -> single value}. Each matching binding produces a new row.
    """
    idx = data.index.get(kg)
    if idx is None:
        return []
    p_key = p_term.strip('<>')

    # ---- candidate subjects ----
    if s_term.startswith('?'):
        bound = row.get(s_term)
        if bound is not None:
            cand = [bound] if (bound, p_key) in idx else []
        else:
            cand = [k for k in data.n[kg] if (k, p_key) in idx]
    else:
        kind, val, dt = data.literal(s_term)
        cand = [val] if (kind == 'iri' and (val, p_key) in idx) else []

    results = []
    for s in cand:
        for o in idx.get((s, p_key), []):
            kind, val, dt = data.literal(o)
            if o_term.startswith('?'):
                prev = row.get(o_term)
                if prev is not None and prev != val:
                    continue
                newrow = dict(row)
                newrow[o_term] = val
                if s_term.startswith('?'):
                    newrow[s_term] = s
                results.append(newrow)
            else:
                kind2, val2, dt2 = data.literal(o_term)
                matched = False
                if kind2 == 'iri':
                    matched = (kind == 'iri' and val2 == val)
                else:
                    if val == val2:
                        matched = True
                    elif dt2 and dt and dt != dt2:
                        try:
                            matched = float(val) == float(val2)
                        except ValueError:
                            matched = False
                if matched:
                    # the subject must still be bound even when the object is a literal
                    newrow = dict(row)
                    if s_term.startswith('?'):
                        newrow[s_term] = s
                    results.append(newrow)
    return results


def eval_patterns(data, kg, pats, rows):
    for pat in pats:
        if isinstance(pat, tuple) and pat[0] == 'UNION':
            left = eval_patterns(data, kg, pat[1], rows)
            right = eval_patterns(data, kg, pat[2], rows)
            rows = left + right
        elif isinstance(pat, tuple) and pat[0] == 'OPTIONAL':
            newrows = []
            for r in rows:
                opt = eval_patterns(data, kg, pat[1], [r])
                newrows.extend(opt if opt else [r])
            rows = newrows
        elif pat[0] == 'BIND':
            continue
        else:
            s, p, o = pat
            nxt = []
            for r in rows:
                nxt.extend(match_triple(data, kg, s, p, o, r))
            rows = nxt
    return rows


# ---------------------------------------------------------------- filters
FILTER_HEAD = re.compile(r'FILTER\s*\(', re.I)

# YEAR() is only meaningful on a date, but the row values carry no datatype
# (match_triple stores the bare lexical form), so the year is recovered from
# the ISO shape of the literal.
ISO_DATE = re.compile(r'^(-?\d{4})-(\d{2})-(\d{2})')
ISO_YEAR = re.compile(r'^(-?\d{4})$')


def filter_spans(text):
    """Bodies of every FILTER(...) in `text`, honouring nested parens so
    YEAR(?x) survives."""
    out = []
    for m in FILTER_HEAD.finditer(text):
        depth = 1
        i = m.end()
        while i < len(text) and depth:
            if text[i] == '(':
                depth += 1
            elif text[i] == ')':
                depth -= 1
            i += 1
        out.append(text[m.end():i - 1].strip())
    return out


def _as_number(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _year_of(v):
    s = str(v)
    m = ISO_DATE.match(s) or ISO_YEAR.match(s)
    return int(m.group(1)) if m else None


def _compare(a, op, b):
    na, nb = _as_number(a), _as_number(b)
    if na is not None and nb is not None:
        x, y = na, nb
    elif (na is None) != (nb is None):
        return None                      # type error, as in real SPARQL
    else:
        x, y = str(a), str(b)            # ISO dates compare lexicographically
    return {'>': x > y, '<': x < y, '>=': x >= y, '<=': x <= y,
            '=': x == y, '==': x == y, '!=': x != y}[op]


def eval_filter_expr(expr, data, row):
    """Evaluate one FILTER body against a row.

    True/False, or None when the expression errors (unbound variable,
    unsupported function, type mismatch) -- an erroring filter drops the
    row, same as SPARQL.
    """
    expr = expr.strip()
    # FUNC(ARG) OP LITERAL
    m = re.match(r'^([A-Za-z]+)\s*\(\s*(\?\w+)\s*\)\s*(>=|<=|!=|==|=|>|<)\s*(.+)$',
                 expr)
    if m:
        fn, var, op, lit = m.group(1), m.group(2), m.group(3), m.group(4)
        if fn.upper() != 'YEAR':
            return None
        v = row.get(var)
        if v is None:
            return None
        y = _year_of(v)
        if y is None:
            return None
        return _compare(y, op, _literal_value(lit, data))
    # VAR OP LITERAL
    m = re.match(r'^(\?\w+)\s*(>=|<=|!=|==|=|>|<)\s*(.+)$', expr)
    if m:
        var, op, lit = m.group(1), m.group(2), m.group(3)
        v = row.get(var)
        if v is None:
            return None
        return _compare(v, op, _literal_value(lit, data))
    return None


def _literal_value(lit, data):
    lit = lit.strip()
    if not (lit.startswith('"') or lit.startswith('<')):
        lit = '"' + lit + '"'
    _, val, _ = data.literal(lit)
    return val


def apply_filter_exprs(data, exprs, rows):
    for expr in exprs:
        out = []
        for r in rows:
            if eval_filter_expr(expr, data, r) is True:
                out.append(r)
        rows = out
    return rows


def apply_filters(data, body, rows):
    return apply_filter_exprs(data, filter_spans(body), rows)


def execute(sparql, data=None):
    data = data or Data()
    q = Query(sparql, data)
    rows = [{}]
    for kind, kg, inner in q.blocks:
        if kind == 'service':
            pats = parse_triples(inner)
            rows = eval_patterns(data, kg, pats, rows)
        else:
            pats = parse_triples(inner)
            rows = eval_patterns(data, 'dbpedia', pats, rows)
        rows = apply_filters(data, inner, rows)
        if not rows:
            break
    # a FILTER outside every SERVICE block scopes over the whole join
    rows = apply_filter_exprs(data, q.outer_filters, rows)
    out = []
    if q.agg:
        fn, src, dst = q.agg
        vals = set()
        for r in rows:
            if src in r:
                vals.add(r[src])
        if fn == 'COUNT':
            v = len(vals)
        elif fn == 'MIN':
            v = min(vals) if vals else None
        elif fn == 'MAX':
            v = max(vals) if vals else None
        else:
            v = sum(float(x) for x in vals) if vals else 0
        out.append({dst: str(v)})
        rows = out
    res = []
    for r in rows:
        rec = {}
        for v in q.select_vars:
            rec[v] = r.get(v)
        res.append(rec)
    if q.distinct:
        seen, dd = set(), []
        for rec in res:
            k = tuple(sorted((kk, vv) for kk, vv in rec.items()))
            if k not in seen:
                seen.add(k)
                dd.append(rec)
        res = dd
    for var, desc in q.order:
        res.sort(key=lambda rec: (rec.get(var) is None, rec.get(var)), reverse=desc)
    if q.limit:
        res = res[:q.limit]
    return res
