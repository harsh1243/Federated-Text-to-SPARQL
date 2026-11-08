# OELP — Text-to-Federated-SPARQL over dbpedia + lmdb + drugbank

The code that turns a natural-language question into a federated SPARQL query
spanning three knowledge graphs, and executes it.

```
question
  -> LLM-A            llm_A.ipynb / run_llm_a_live.py
                       a KG-agnostic SPARQL skeleton + a description per slot
  -> retriever        retriever_final.py  (properties + classes)
                       retriever_entity.py (entities)
                       candidate IRIs per slot, per graph
  -> CONSTRAINTS      constraint_solver.py
                       ranked grounded plans; every binding checked on the dump
  -> compiler         llm_b.py
                       one federated SPARQL query per surviving plan
  -> execute          sparql_local.py, triple_index.py
                       run against the local dumps, compare answer rows
```

## Results (50 gold queries, execution accuracy against the gold's rows)

| metric | with the constraint layer | without it |
|---|---|---|
| query executes == gold answers | **48/50 (96.0%)** | 0/50 (0.0%) |
| plan found | 48/50 | 37/50 |
| endpoint set matches gold | 47/50 | 6/50 |

"executes == gold answers" means *some* surviving plan does; the top-scored
plan alone reaches 37/50 (74.0%).

## Files

| file | role |
|---|---|
| `llm_A.ipynb` | the LLM-A prompt (skeleton + slot descriptions) |
| `_llm_a_prompt.txt` | the same prompt as plain text, for scripted runs |
| `run_llm_a_live.py` | live LLM-A calls — validated, retried, output hashed |
| `retriever_final.py` | properties + classes (dense + BM25, class-aware) |
| `retriever_entity.py` | entities (BM25 + exact + `owl:sameAs` expansion) |
| `constraint_solver.py` | **the contribution** — grounded plan enumeration |
| `llm_b.py` | deterministic plan → federated SPARQL compiler |
| `sparql_local.py` | mini SPARQL engine over the local neighbourhood |
| `triple_index.py` | (subject, predicate) existence index, per KG |
| `run_pipeline.py` | run the pipeline over all 50 → `_final_queries.json` |
| `eval_end_to_end.py` | the end-to-end harness |
| `run_benchmark_final.py` | the benchmark harness: exact match + P/R/F1 |
| `eval_constraint_solver.py` | plan-found / endpoint-set + the unconstrained ablation |

## Data this code needs at runtime

The code is here; the data is not, because it is large and derived from
third-party dumps that cannot be redistributed. Supply these in the working
directory:

| path | size | what it is |
|---|---|---|
| `kg_embeddings_class.zip` | 6.8 MB | property + class embeddings (MiniLM, 384-dim) |
| `sameas_joins.json` | 12 MB | the `owl:sameAs` join surface |
| `_bridge_graph.json` | 11 MB | sameAs + dbpedia category-hub membership |
| `_class_predicates.json` | 264 KB | which predicates fire on each class |
| `_predicate_ranges.json` | 92 KB | what kind of object each predicate points at |
| `_entity_bm25/` | 629 MB | BM25 index over 4,468,704 entity labels |
| `_triple_index/` | 639 MB | triple-existence index, per graph |
| `_nbhd.json`, `_nbhd_labels.json` | 4 MB | the harvested query neighbourhood |
| `benchmark_50.json` | 244 KB | the 50 gold questions and their answers |
| `_llm_a_runs.json` | 36 KB | the frozen LLM-A output the numbers were measured on |

## Install and run

```bash
pip install -r requirements.txt

export ANTHROPIC_API_KEY=...        # and ANTHROPIC_BASE_URL if you use a proxy

# live LLM-A over the benchmark, then the full pipeline
python run_llm_a_live.py --model claude-sonnet-5-5
python run_benchmark_final.py --runs _llm_a_live_claude_sonnet_5_5.json --out _bench_sonnet.json

# or the end-to-end harness against the frozen LLM-A output
python eval_end_to_end.py

# the pipeline proper: every surviving plan, compiled and executed
python run_pipeline.py
```

## Reproducibility

The LLM-A call is stochastic, and the endpoint is **not** deterministic even at
`temperature=0` (measured: 6 identical calls produced 6 distinct responses).
The reproducibility guarantee is therefore the **frozen artefact**
(`_llm_a_runs.json`), not the sampling temperature — anyone re-running the
pipeline on that file gets byte-identical numbers. `run_llm_a_live.py` records
the SHA-256 of the 50 outputs it produces.

## Two notes before you read the code

**The skeleton parser rejects anything that is not a triple.** A line counts
as a triple only if the *predicate position* holds a variable, the `a`
keyword, or an IRI. The LLM-A prompt explicitly allows `FILTER`, and a
`FILTER` over a string containing a space splits into three whitespace
tokens — so a naive parser turns it into a triple whose predicate is a
fragment of the string literal. See `retriever_final.split_triple_line`.

**The constraint layer retrieves at `top_k = 10`**, the retriever's validated
operating point (`retriever_final.TOP_K = 10`; 218/221 property pairs
retrievable). The solver previously used `top_k=5` — half that — which cost a
query, because a paraphrase of a slot description only has to push the gold
predicate out of a short list to break it.
