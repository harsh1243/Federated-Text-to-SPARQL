"""
Retriever for LLM-A -> LLM-B stage.
Takes raw LLM-A output text, returns structured matches.
"""

import json
import zipfile
import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_ZIP = "kg_embeddings.zip"
MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"
TOP_K = 5

_kgs = None
_model = None


def _load():
    """Lazy-load model + embeddings once."""
    global _kgs, _model
    if _kgs is not None:
        return _kgs, _model

    _model = SentenceTransformer(MODEL_NAME)
    _kgs = {}

    with zipfile.ZipFile(EMBED_ZIP) as z:
        for name in z.namelist():
            if not name.endswith(".json"):
                continue
            data = json.loads(z.read(name))
            kg = data["kg"]

            prop_matrix = np.array([p["embedding"] for p in data["properties"]], dtype=np.float32)
            class_matrix = np.array([c["embedding"] for c in data["classes"]], dtype=np.float32)
            prop_matrix /= (np.linalg.norm(prop_matrix, axis=1, keepdims=True) + 1e-12)
            class_matrix /= (np.linalg.norm(class_matrix, axis=1, keepdims=True) + 1e-12)

            _kgs[kg] = {
                "props_vec": prop_matrix,
                "classes_vec": class_matrix,
                "props_meta": data["properties"],
                "classes_meta": data["classes"],
            }
    return _kgs, _model


def _parse_llm_a(llm_text):
    """Strip fences, return the JSON block as a dict."""
    s = llm_text.strip()
    if "```" in s:
        s = s.split("```", 2)[1]
        if s.startswith("json"):
            s = s[4:]
        s = s.rsplit("```", 1)[0]
    return json.loads(s.strip())


def _topk(query_vec, kgs, slot, top_k):
    """Return top-k across all KGs for the given slot ('props' or 'classes')."""
    hits = []
    for kg, d in kgs.items():
        matrix = d[f"{slot}_vec"]
        meta = d[f"{slot}_meta"]
        scores = matrix @ query_vec
        top_idx = np.argsort(-scores)[:top_k]
        for i in top_idx:
            hits.append({
                "kg": kg,
                "uri": meta[i]["uri"],
                "label": meta[i]["label"],
                "score": round(float(scores[i]), 4),
            })
    hits.sort(key=lambda x: -x["score"])
    return hits[:top_k]


def retrieve(llm_a_output, top_k=TOP_K):
    """
    Input  : raw LLM-A text (with or without ```json fences)
    Output : dict ready to feed into LLM-B stage
        {
          "properties": {
              "?authored": [{"kg", "uri", "label", "score"}, ...top_k...],
              ...
          },
          "classes": {
              "?paper_type": [{"kg", "uri", "label", "score"}, ...top_k...],
              ...
          },
          "entities": { ... untouched from LLM-A ... }
        }
    """
    kgs, model = _load()
    parsed = _parse_llm_a(llm_a_output)

    props = parsed.get("properties", {})
    classes = parsed.get("classes", {})

    out = {"properties": {}, "classes": {}, "entities": parsed.get("entities", {})}

    if props:
        descs = list(props.values())
        vecs = model.encode(descs, normalize_embeddings=True, convert_to_numpy=True)
        for var, desc in props.items():
            q = vecs[descs.index(desc)]
            out["properties"][var] = _topk(q, kgs, "props", top_k)

    if classes:
        descs = list(classes.values())
        vecs = model.encode(descs, normalize_embeddings=True, convert_to_numpy=True)
        for var, desc in classes.items():
            q = vecs[descs.index(desc)]
            out["classes"][var] = _topk(q, kgs, "classes", top_k)

    return out


if __name__ == "__main__":
 
    llm_a_text = """
```sparql
SELECT ?paper WHERE {
  ?paper a ?paper_type .
  ?einstein ?authored ?paper .
  ?einstein ?worked_at ?princeton .
}
```

```json
{
  "properties": {
    "?authored": "links a person (subject) to a paper they authored (object)",
    "?worked_at": "links a person (subject) to an organization they worked at (object)"
  },
  "classes": {
    "?paper_type": "academic paper or scholarly publication"
  },
  "entities": {
    "?einstein": "Albert Einstein, theoretical physicist",
    "?princeton": "Princeton University"
  }
}
```
"""
    result = retrieve(llm_a_text, top_k=5)
    print(json.dumps(result, indent=2))
