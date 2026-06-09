from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def load_tau_json(tau_path: str | Path) -> Dict[str, float]:
    with open(tau_path, "r", encoding="utf-8") as f:
        tau = json.load(f)
    return {str(k): float(v) for k, v in tau.items()}


def _minmax_normalize(values: list[float]) -> list[float]:
    mn, mx = min(values), max(values)
    if mx <= mn:
        return [0.0 for _ in values]
    return [(v - mn) / (mx - mn) for v in values]


def fuse_query_scores(
    doc_scores: Dict[str, float],
    tau: Dict[str, float],
    alpha: float = 0.6,
    default_tau: float = 0.5,
) -> Dict[str, float]:
    doc_ids = list(doc_scores.keys())
    scores_norm = _minmax_normalize([doc_scores[d] for d in doc_ids])

    fused: Dict[str, float] = {}
    for doc_id, s_i in zip(doc_ids, scores_norm):
        t_i = tau.get(doc_id, default_tau)
        fused[doc_id] = alpha * t_i + (1.0 - alpha) * s_i
    return fused


def trust_rerank_results(
    results: Dict[str, Dict[str, float]],
    tau: Dict[str, float],
    alpha: float = 0.6,
    default_tau: float = 0.5,
    top_k: int | None = None,
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}

    for qid, doc_scores in results.items():
        fused = fuse_query_scores(doc_scores, tau, alpha=alpha, default_tau=default_tau)
        if top_k is not None:
            fused = dict(sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k])
        out[qid] = fused

    return out
