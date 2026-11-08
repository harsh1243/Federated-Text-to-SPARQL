"""LLM-B: the final stage -- grounded plans become federated SPARQL.

WHY THIS STAGE EXISTS
---------------------
The retrievers emit, per slot, a ranked list of candidate IRIs *per KG*.  The
constraint solver turns those lists into ranked GROUNDED PLANS: which KG serves
each triple, which URI binds each slot, which identity edge carries the join,
and what constant a shared variable must take.  This module is the compiler
that writes those decisions out as SPARQL.

The compiler is DETERMINISTIC.  It does not call a language model, and that is
the point of the architecture:

    LLM-A  ->  retrieve  ->  CONSTRAIN  ->  COMPILE
    (skeleton)  (candidates)  (grounded     (federated
                    per slot)   plans)        SPARQL)

A conventional text-to-SPARQL pipeline ends with an LLM that rewrites a
candidate list into a query, and it is exactly that rewrite step where a
dbpedia predicate gets paired with an lmdb entity and the query dies.  Here the
constraint layer has already decided every binding, so the last stage has
nothing left to guess.  Every triple it writes is one the constraint layer
checked against the dump.

It also emits ONE QUERY PER SURVIVING PLAN.  The solver deliberately returns
several ranked plans because the constraints narrow the space but do not
always collapse it to one; each surviving plan is a legal, executable query,
and the caller takes the first one it likes.  (The user asked for exactly
this: if several combinations pass the constraints, write them all.)

WHAT A PLAN MEANS, AND HOW IT BECOMES SPARQL
---------------------------------------------
A plan carries:

  kg_of_triple   {triple index -> KG}      which endpoint serves each triple
  properties     {slot -> {uri, kg}}        the predicate bound to each slot
  classes        {slot -> {uri, kg}}        the class bound to each slot
  entities       {var -> {kg -> {uri,...}}}  where each logical entity lives
  bridges        [{kind, entities, via}]     the identity edges used
  shared_var_bindings {var -> constant}      what a shared variable binds to

The skeleton triples from LLM-A are KG-agnostic: `?trainspotting a ?film_type`,
`?trainspotting ?subject_category ?shared_category`.  A slot's variable name is
a placeholder for a real URI; the plan says which URI and which KG.  So the
compiler walks the skeleton triples, looks up each triple's KG and each slot's
URI, and emits the same triple with the placeholders replaced.

THE FOUR STRUCTURAL RULES
--------------------------
1. THE NAME TRIPLE.  A federated query identifies an entity by its NAME in
   exactly one endpoint -- the one that holds the label.  The name predicate
   is per endpoint and per entity class, decided by the dump:
   dbpedia and drugbank use rdfs:label, and lmdb uses dcterms:title for
   films but rdfs:label for actors (lmdb films carry both, with identical
   values; actors carry only rdfs:label).

2. THE IDENTITY EDGE.  Entities joined across endpoints are linked by
   owl:sameAs -- and every sameAs edge in these three dumps points INTO
   dbpedia (lmdb->dbpedia 30,354, drugbank->dbpedia 1,891, lmdb->drugbank 0).
   So the edge triple is PHYSICALLY STORED in the leaf endpoint, and it must
   be written there: `?lmdbThing owl:sameAs ?dbpediaThing`.  dbpedia itself
   carries no outgoing sameAs at all, so a query that puts the edge in the
   dbpedia block would match zero triples.

3. THE BRIDGE.  Two entities joined through a shared variable are bridged by
   a shared dbpedia category.  The category IRI is written into the dbpedia
   block as a literal constant -- that constant is the join key, and it is
   what multiplies the two SERVICE blocks together.  (There is no
   lmdb->drugbank sameAs edge, so a three-KG query cannot chain lmdb straight
   to drugbank; it routes through dbpedia via the category.)

4. THE OPTIONAL CLASS.  A class triple is a CONSTRAINT, not content: `?x a
   ?film_type`.  The plan says which class URI binds it and which KG serves
   it.  When the solver dropped the class triple (kg_of_triple is None) the
   compiler omits it -- a type constraint that was judged unnecessary must not
   reappear as an unrequested filter.

VARIABLE NAMING
---------------
The skeleton variable names are reused where they are unambiguous, and renamed
where two things collide:

  * An entity realised in several KGs needs one variable PER ENDPOINT, because
    each endpoint holds a different URI for it.  The dbpedia side keeps the
    skeleton's name (?trainspotting); the leaf side is prefixed
    (?lmdb_trainspotting, ?dbk_heroin) -- the same convention the benchmark's
    own gold queries use.
  * An ANSWER variable (the object of a property triple) is renamed after the
    predicate that produces it, so ?date_of_birth becomes ?dbp_birthDate.
    That both matches the gold convention and dodges the degenerate case where
    LLM-A reuses the slot name in the object position (`?ronin ?release_date
    ?release_date`).
"""
import json
import os
import re
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import constraint_solver as CS

# The public endpoints.  These are the SPARQL service URLs of the three dumps.
ENDPOINTS = {
    'dbpedia': 'https://dbpedia.org/sparql',
    'lmdb': 'http://data.linkedmdb.org/sparql',
    'drugbank': 'http://www4.wiwiss.fu-berlin.de/drugbank/sparql',
}

RDFS_LABEL = 'http://www.w3.org/2000/01/rdf-schema#label'
DCT_TITLE = 'http://purl.org/dc/terms/title'
OWL_SAMEAS = 'http://www.w3.org/2002/07/owl#sameAs'
RDF_TYPE = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'

# The prefix each endpoint's variables carry.  Matches the benchmark's gold
# queries (?dbp_birthDate, ?dbk_indication, ?lmdb_language).
VAR_PREFIX = {'dbpedia': 'dbp', 'drugbank': 'dbk', 'lmdb': 'lmdb'}

# Order the SERVICE blocks appear in.
KG_ORDER = ('dbpedia', 'lmdb', 'drugbank')

# The endpoint every sameAs edge points INTO.  It is the hub of the identity
# graph, so it is the one endpoint that never carries an outgoing sameAs.
HUB = 'dbpedia'


def _esc(s):
    """Escape a string for an N-Triples-style literal."""
    return (s.replace('\\', '\\\\')
             .replace('"', '\\"')
             .replace('\n', '\\n')
             .replace('\r', '\\r')
             .replace('\t', '\\t'))


def _lit(s):
    return '"%s"' % _esc(s)


def _sanitize(name):
    """'?trainspotting' -> 'trainspotting'."""
    return re.sub(r'[^A-Za-z0-9_]', '_', name.lstrip('?'))


def _pred_local(uri):
    """The local name of a predicate: 'dbo:birthDate' -> 'birthDate'."""
    tail = uri.rsplit('#', 1)[-1]
    tail = tail.rsplit('/', 1)[-1]
    return tail


def _pred_var(kg, pred_uri):
    """An answer variable named after the predicate that produces it."""
    return '?%s_%s' % (VAR_PREFIX[kg], _pred_local(pred_uri))


class Compiler:
    """Turn one grounded plan into one federated SPARQL query."""

    # ------------------------------------------------------------ name forms
    def _name_predicate(self, kg, cls):
        """Which predicate carries this entity's name in this KG?

        Checked against the dump rather than assumed.  dbpedia and drugbank
        use rdfs:label for everything (dbpedia has no dcterms:title anywhere
        in the subset; drugbank has none either).  lmdb uses dcterms:title
        for films -- which carry BOTH dcterms:title and rdfs:label with
        identical values -- and rdfs:label for actors, which carry only
        rdfs:label.  So the choice is per entity class, not per KG.
        """
        if kg == 'lmdb' and cls == 'film':
            return DCT_TITLE
        return RDFS_LABEL

    # ------------------------------------------------------------- compiling
    def compile(self, plan, parsed):
        """Compile one plan into a federated SPARQL query.

        Returns a dict: {'sparql', 'kgs', 'warnings'}.  Raises ValueError if
        the plan cannot be compiled, which would mean the solver handed over
        something internally inconsistent.
        """
        triples = parsed['triples']
        kg_of = plan['kg_of_triple']
        props = plan.get('properties', {}) or {}
        classes = plan.get('classes', {}) or {}
        entities = plan.get('entities', {}) or {}
        shared = plan.get('shared_var_bindings', {}) or {}
        optional = set(plan.get('optional_slots', []) or [])

        warnings = []

        # ---- 1. the variable each (entity, endpoint) pair is written with.
        # An entity realised in several endpoints has a different URI in each,
        # so each needs its own variable.  The dbpedia side keeps the
        # skeleton's name; the leaf sides are prefixed.
        ent_var = {}
        for var, per_kg in entities.items():
            base = _sanitize(var)
            for kg in per_kg:
                if kg == HUB:
                    ent_var[(var, kg)] = '?' + base
                else:
                    ent_var[(var, kg)] = '?%s_%s' % (VAR_PREFIX[kg], base)

        # ---- 2. the answer variables.  The object of a property triple is a
        # value, not a name, so it is renamed after the predicate that
        # produces it.  A shared variable is not a value at all: it is bound
        # to a constant, so it is substituted rather than renamed.
        ans_var = {}
        used_names = set()
        for i, t in enumerate(triples):
            if t['kind'] != 'prop':
                continue
            o = t['o']
            if o in entities or o in shared or o in ans_var:
                continue
            kg = kg_of.get(str(i))
            if kg is None:
                continue
            pred = props.get(t['slot'], {}).get('uri')
            if not pred:
                continue
            name = _pred_var(kg, pred)
            # two different value variables must not collide
            k = 2
            while name in used_names:
                name = '%s_%d' % (_pred_var(kg, pred), k)
                k += 1
            used_names.add(name)
            ans_var[o] = name

        # ---- 3. which endpoints does this query need?
        kgs = set(k for k in kg_of.values() if k)
        for per_kg in entities.values():
            kgs |= set(per_kg)

        # ---- 4. group the skeleton triples by the endpoint that serves them
        by_kg = {kg: [] for kg in kgs}
        for i, t in enumerate(triples):
            kg = kg_of.get(str(i))
            if kg is None:
                continue            # dropped (optional class constraint)
            if kg not in by_kg:
                raise ValueError('triple %d assigned to unknown KG %r' % (i, kg))
            by_kg[kg].append((i, t))

        # ---- 5. per endpoint, build its block
        blocks = {}
        for kg in KG_ORDER:
            if kg not in by_kg:
                continue
            lines = []
            lines += self._name_lines(kg, entities, ent_var, warnings)
            lines += self._edge_lines(kg, entities, ent_var, warnings)
            lines += self._content_lines(kg, by_kg[kg], triples, props, classes,
                                         entities, ent_var, ans_var, shared,
                                         parsed, optional, warnings)
            if lines:
                blocks[kg] = lines

        if not blocks:
            raise ValueError('plan produced no triples')

        # ---- 6. the SELECT list
        select = self._select_vars(parsed, ans_var, entities, ent_var, shared)

        # ---- 7. assemble
        out = ['SELECT DISTINCT %s WHERE {'
               % ' '.join('?' + v.lstrip('?') for v in select)]
        for kg in KG_ORDER:
            if kg not in blocks:
                continue
            out.append('  SERVICE <%s> {' % ENDPOINTS[kg])
            for ln in blocks[kg]:
                out.append('    ' + ln)
            out.append('  }')
        out.append('}')

        return {'sparql': '\n'.join(out),
                'kgs': [kg for kg in KG_ORDER if kg in blocks],
                'warnings': warnings}

    # ------------------------------------------------------------- anchoring
    def _name_lines(self, kg, entities, ent_var, warnings):
        """The name triple for this endpoint.

        A federated query names an entity in exactly one endpoint -- the one
        whose realisation was matched by name, which the plan records as the
        realisation whose source is not 'sameAs'.  In the other endpoints the
        entity is reached through owl:sameAs, so those blocks get the identity
        edge instead of the label.
        """
        lines = []
        for var, per_kg in entities.items():
            if kg not in per_kg:
                continue
            me = per_kg[kg]
            if me.get('source') == 'sameAs':
                continue            # reached through the edge, not named here
            label = me.get('label')
            if label is None:
                warnings.append('entity %s in %s has no label to anchor on'
                                % (var, kg))
                continue
            pred = self._name_predicate(kg, me.get('class'))
            lines.append('%s <%s> %s .'
                         % (ent_var[(var, kg)], pred, _lit(label)))
        return lines

    # --------------------------------------------------------- identity edge
    def _edge_lines(self, kg, entities, ent_var, warnings):
        """The owl:sameAs edges this endpoint serves.

        Every sameAs edge in these dumps points INTO dbpedia, so the edge
        triple is stored in the LEAF endpoint and must be written there:
        the leaf's variable is the subject, the dbpedia variable the object.

        dbpedia therefore never carries an outgoing sameAs -- which is why
        the identity of a dbpedia-side entity is established by the LEAF
        blocks, not by dbpedia's own.
        """
        if kg == HUB:
            return []              # the hub carries no outgoing sameAs
        lines = []
        for var, per_kg in entities.items():
            if kg not in per_kg:
                continue
            if HUB not in per_kg:
                # an entity in two leaf endpoints would need an
                # lmdb->drugbank edge, which does not exist
                if len(per_kg) > 1:
                    warnings.append('entity %s spans leaf endpoints %s with no '
                                    'identity edge' % (var, sorted(per_kg)))
                continue
            lines.append('%s <%s> %s .'
                         % (ent_var[(var, kg)], OWL_SAMEAS,
                            ent_var[(var, HUB)]))
        return lines

    # -------------------------------------------------------------- content
    def _content_lines(self, kg, idx_triples, triples, props, classes,
                       entities, ent_var, ans_var, shared, parsed, optional,
                       warnings):
        """The content triples this endpoint serves."""
        lines = []
        for i, t in idx_triples:
            slot = t['slot']
            if t['kind'] == 'class':
                # `?x a ?class_slot` -- the class slot binds a class URI.
                h = classes.get(slot)
                if not h:
                    continue
                subj = self._term(t['s'], kg, entities, ent_var, ans_var,
                                   shared)
                if subj is None:
                    continue
                lines.append('%s <%s> <%s> .' % (subj, RDF_TYPE, h['uri']))
                continue

            # a property triple: ?s ?slot ?o
            h = props.get(slot)
            if not h:
                continue
            subj = self._term(t['s'], kg, entities, ent_var, ans_var, shared)
            if subj is None:
                warnings.append('triple %d has no subject in %s' % (i, kg))
                continue
            obj = self._term(t['o'], kg, entities, ent_var, ans_var, shared)
            if obj is None:
                warnings.append('triple %d has no object in %s' % (i, kg))
                continue
            line = '%s <%s> %s .' % (subj, h['uri'], obj)
            if slot in optional:
                # The fact does not exist for this entity (A09's Peter
                # O'Toole has no deathDate).  A REQUIRED triple would
                # eliminate the whole solution, so it is written as
                # OPTIONAL: the question's answerable part still gets
                # answered, and the missing part comes back unbound --
                # which is what the gold queries do too.
                lines.append('OPTIONAL { %s }' % line)
            else:
                lines.append(line)
        return lines

    # --------------------------------------------------------------- terms
    def _term(self, var, kg, entities, ent_var, ans_var, shared):
        """The term a variable is written as in one endpoint.

        Three cases:
          * a NAMED entity  -> its variable in this endpoint
          * a SHARED variable -> the bridge constant it is bound to
          * an ANSWER variable -> its renamed answer variable
        """
        if var in shared:
            return '<%s>' % shared[var]
        if var in entities:
            v = ent_var.get((var, kg))
            if v is None:
                return None
            return v
        return ans_var.get(var)

    # --------------------------------------------------------------- SELECT
    def _select_vars(self, parsed, ans_var, entities, ent_var, shared):
        """The variables the query projects.

        The answer variables are what the question asks for, so they come
        from LLM-A's own SELECT clause -- that is the question's intent, and
        the skeleton says which variable holds it.  A shared variable is a
        join key, not an answer, so it is dropped.  An entity variable is
        projected through its dbpedia side, which is where the name and the
        content live.
        """
        out = []
        for v in parsed.get('select_vars', []) or []:
            if v in shared:
                continue
            if v in entities:
                name = ent_var.get((v, HUB)) or ent_var.get((v, 'lmdb'))
                if name is None:
                    continue
            elif v in ans_var:
                name = ans_var[v]
            else:
                continue
            if name not in out:
                out.append(name)
        if not out:
            # no usable SELECT clause: project every answer variable
            for name in ans_var.values():
                if name not in out:
                    out.append(name)
        return out


# ======================================================================= main
def compile_plan(plan, parsed, compiler=None):
    """Compile one plan.  Returns None if the plan is not compilable."""
    compiler = compiler or Compiler()
    try:
        return compiler.compile(plan, parsed)
    except ValueError:
        return None


def compile_solutions(solutions, parsed, compiler=None):
    """Compile every surviving plan.

    `solutions` is the solver's output: {'plans': [...], 'stats': {...}}.

    Returns one compiled query per surviving plan, in the solver's ranking
    order.  The user asked for all of them: if several combinations pass the
    constraints, each is a legal query and each is written out.
    """
    compiler = compiler or Compiler()
    out = []
    for n, plan in enumerate(solutions.get('plans', [])):
        try:
            res = compiler.compile(plan, parsed)
        except ValueError as e:
            res = {'sparql': None, 'kgs': [], 'warnings': [str(e)]}
        res['rank'] = n
        res['score'] = plan.get('score')
        res['plan'] = plan
        out.append(res)
    return out


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
    solver = CS.ConstraintSolver()
    res = solver.solve(llm_a_text)
    parsed = CS.parse_llm_a(llm_a_text)
    for r in compile_solutions(res, parsed):
        print('=' * 76)
        print('rank %d  score %s  kgs %s' % (r['rank'], r['score'], r['kgs']))
        print(r['sparql'])
