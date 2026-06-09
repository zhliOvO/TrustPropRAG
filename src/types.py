from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HumanFeedback:
    doc_id: str
    rating: str
    confidence: float = 1.0


@dataclass
class DocumentEdge:
    src: str
    dst: str
    label: int
    confidence: float
    similarity: float = 1.0


LabeledEdge = tuple[str, str, int, float, float]
