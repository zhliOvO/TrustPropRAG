from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from rich.console import Console

from trustprop_rag.feedback.feedback import HumanFeedback
from trustprop_rag.generation.trust_prompting import generate_trust_aware_answer
from trustprop_rag.graph import build_document_relation_graph
from trustprop_rag.graph.nli_labeler import load_labeled_edges, save_labeled_edges
from trustprop_rag.retrieval.dense import DenseRetriever
from trustprop_rag.retrieval.trust_rerank import trust_rerank_results
from trustprop_rag.trust.export import export_tau_json
from trustprop_rag.trust.optimizer import TrustOptimizerConfig, TrustPropOptimizer
from trustprop_rag.types import LabeledEdge

console = Console()


@dataclass
class TrustPropRAGConfig:
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    retrieval_top_k: int = 100
    rerank_alpha: float = 0.6
    generation_top_k: int = 5
    trust_optimizer: TrustOptimizerConfig = field(default_factory=TrustOptimizerConfig)


class TrustPropRAGPipeline:
    def __init__(self, config: Optional[TrustPropRAGConfig] = None):
        self.config = config or TrustPropRAGConfig()
        self.corpus: Optional[Dict] = None
        self.labeled_edges: Optional[List[LabeledEdge]] = None
        self.tau: Optional[Dict[str, float]] = None
        self.retriever: Optional[DenseRetriever] = None

    def build_graph(
        self,
        corpus: Dict,
        cache_path: Optional[Path] = None,
    ) -> List[LabeledEdge]:
        self.corpus = corpus

        if cache_path and cache_path.exists():
            console.print(f"[dim]Loading cached labeled edges from {cache_path}[/dim]")
            self.labeled_edges = load_labeled_edges(cache_path)
            return self.labeled_edges

        console.print("[bold cyan]Stage 1: Document Relation Graph Construction[/bold cyan]")
        self.labeled_edges = build_document_relation_graph(
            corpus,
            nli_model=self.config.nli_model,
        )

        if cache_path:
            save_labeled_edges(self.labeled_edges, cache_path)

        return self.labeled_edges

    def optimize_trust_scores(
        self,
        feedback: Optional[List[HumanFeedback]] = None,
        output_path: Optional[Path] = None,
    ) -> Dict[str, float]:
        if self.labeled_edges is None or self.corpus is None:
            raise RuntimeError("Call build_graph() before optimize_trust_scores()")

        console.print("[bold cyan]Stage 2: Trust Score Optimization (PGD)[/bold cyan]")
        optimizer = TrustPropOptimizer(self.config.trust_optimizer)
        result = optimizer.optimize(
            self.labeled_edges,
            feedback=feedback,
            all_doc_ids=sorted(self.corpus.keys()),
        )
        self.tau = result.tau

        if output_path:
            export_tau_json(result.tau, output_path)

        return result.tau

    def retrieve(
        self,
        queries: Dict,
        top_k: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        if self.corpus is None:
            raise RuntimeError("Corpus not loaded")

        top_k = top_k or self.config.retrieval_top_k
        self.retriever = DenseRetriever(encoder_name=self.config.encoder_name)
        self.retriever.encode_corpus(self.corpus)
        self.retriever.build_index()

        query_texts = {
            qid: (q.get("text", str(q)) if isinstance(q, dict) else str(q))
            for qid, q in queries.items()
        }
        query_ids, query_embeddings = self.retriever.encode_queries(query_texts)
        return self.retriever.search(query_embeddings, query_ids, top_k=top_k)

    def trust_aware_rescore(
        self,
        retrieval_results: Dict[str, Dict[str, float]],
        tau: Optional[Dict[str, float]] = None,
        alpha: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        tau = tau or self.tau
        if tau is None:
            raise RuntimeError("Trust scores τ not available. Run optimize_trust_scores() first.")

        console.print("[bold cyan]Stage 3: Trust-aware Rescoring[/bold cyan]")
        return trust_rerank_results(
            retrieval_results,
            tau,
            alpha=alpha if alpha is not None else self.config.rerank_alpha,
            top_k=top_k or self.config.generation_top_k,
        )

    def generate_answer(
        self,
        query: str,
        reranked_results: Dict[str, float],
        tau: Optional[Dict[str, float]] = None,
        llm_generate: Optional[Callable] = None,
    ) -> str:
        if self.corpus is None:
            raise RuntimeError("Corpus not loaded")

        tau = tau or self.tau or {}
        top_docs = sorted(reranked_results.items(), key=lambda x: x[1], reverse=True)[
            : self.config.generation_top_k
        ]

        contexts, doc_ids = [], []
        for doc_id, _ in top_docs:
            if doc_id not in self.corpus:
                continue
            doc = self.corpus[doc_id]
            title = doc.get("title", "")
            text = doc.get("text", "")
            contexts.append(f"{title}\n{text}".strip() if title else text)
            doc_ids.append(doc_id)

        console.print("[bold cyan]Stage 4: Trust-aware Answer Generation[/bold cyan]")
        return generate_trust_aware_answer(
            query,
            contexts,
            doc_ids,
            tau,
            llm_generate=llm_generate,
        )

    def run_query(
        self,
        query_id: str,
        query_text: str,
        retrieval_results: Dict[str, Dict[str, float]],
        llm_generate: Optional[Callable] = None,
    ) -> str:
        reranked = self.trust_aware_rescore(retrieval_results)[query_id]
        return self.generate_answer(query_text, reranked, llm_generate=llm_generate)
