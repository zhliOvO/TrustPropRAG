from __future__ import annotations

from typing import Dict, List

from trustprop_rag.graph.nli_labeler import NLILabeler, load_labeled_edges, save_labeled_edges
from trustprop_rag.types import LabeledEdge


def build_document_relation_graph(
    corpus: Dict[str, Dict],
    nli_model: str = "cross-encoder/nli-deberta-v3-small",
    nli_batch_size: int = 16,
) -> List[LabeledEdge]:
    nli_labeler = NLILabeler(model_name=nli_model)
    return nli_labeler.label_all_pairs(corpus, batch_size=nli_batch_size)
