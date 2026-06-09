# TrustPropRAG

Core implementation of TrustPropRAG.

## Pipeline

| Stage | Module |
|-------|--------|
| Document relation graph | `graph/` |
| Trust score optimization | `trust/optimizer.py` |
| Trust-aware rescoring | `retrieval/trust_rerank.py` |
| Trust-aware generation | `generation/trust_prompting.py` |

## Structure

```
src/
├── pipeline/trustprop_pipeline.py
├── graph/
├── trust/
├── feedback/
├── retrieval/
├── generation/
└── utils/
```

