from dataclasses import dataclass
from typing import Optional


@dataclass
class RetrievalResult:
    query_id: str
    doc_id: str
    score: float
    rank: int

    def __repr__(self) -> str:
        return f"RetrievalResult(query={self.query_id}, doc={self.doc_id}, score={self.score:.4f}, rank={self.rank})"


@dataclass
class RetrievalConfig:
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    normalize_embeddings: bool = True
    show_progress: bool = True
    top_k: int = 100
    device: Optional[str] = None
    use_fp16: bool = True
    index_type: str = "flat"


@dataclass
class IndexConfig:
    index_type: str = "flat"
    metric: str = "ip"
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 128
    save_path: Optional[str] = None
