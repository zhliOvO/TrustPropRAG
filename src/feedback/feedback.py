from __future__ import annotations

import random
from typing import Dict, List

from trustprop_rag.types import HumanFeedback


def generate_synthetic_feedback(
    contradiction_log: List[Dict],
    corpus: Dict,
    ratio: float,
    seed: int = 42,
) -> List[HumanFeedback]:
    random.seed(seed)
    fact_ids = [item["doc_id"] for item in contradiction_log if item["doc_id"] in corpus]
    contra_ids = [
        f"{item['doc_id']}_contra"
        for item in contradiction_log
        if f"{item['doc_id']}_contra" in corpus
    ]

    feedback: List[HumanFeedback] = []
    n_facts = int(len(fact_ids) * ratio)
    for doc_id in random.sample(fact_ids, min(n_facts, len(fact_ids))):
        feedback.append(HumanFeedback(doc_id, "good", random.uniform(0.7, 1.0)))

    n_contras = int(len(contra_ids) * ratio)
    for doc_id in random.sample(contra_ids, min(n_contras, len(contra_ids))):
        feedback.append(HumanFeedback(doc_id, "bad", random.uniform(0.7, 1.0)))

    return feedback
