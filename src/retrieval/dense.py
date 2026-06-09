from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import torch
from rich.console import Console
from sentence_transformers import SentenceTransformer

console = Console()


class DenseRetriever:
    def __init__(
        self,
        encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        use_fp16: bool = True,
    ):
        self.encoder_name = encoder_name
        self.use_fp16 = use_fp16

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        console.print(f"[cyan]Using device: {self.device}[/cyan]")
        console.print(f"[cyan]Loading encoder: {encoder_name}[/cyan]")
        self.encoder = SentenceTransformer(encoder_name, device=self.device)

        self.index: Optional[faiss.Index] = None
        self.doc_ids: Optional[list[str]] = None
        self.corpus_embeddings: Optional[np.ndarray] = None

    def encode_corpus(
        self,
        corpus: dict,
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = True,
    ) -> np.ndarray:
        self.doc_ids = list(corpus.keys())
        texts = []

        for doc_id in self.doc_ids:
            doc = corpus[doc_id]
            title = doc.get("title", "")
            text = doc.get("text", "")
            combined = f"{title} {text}".strip() if title else text
            texts.append(combined)

        console.print(f"[cyan]Encoding {len(texts)} documents...[/cyan]")

        embeddings = self.encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )

        self.corpus_embeddings = embeddings.astype(np.float32)
        console.print(f"[green]Encoded corpus: shape {self.corpus_embeddings.shape}[/green]")

        return self.corpus_embeddings

    def encode_queries(
        self,
        queries: dict,
        batch_size: int = 32,
        normalize: bool = True,
    ) -> tuple[list[str], np.ndarray]:
        query_ids = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]

        console.print(f"[cyan]Encoding {len(query_texts)} queries...[/cyan]")

        embeddings = self.encoder.encode(
            query_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )

        return query_ids, embeddings.astype(np.float32)

    def build_index(
        self,
        embeddings: Optional[np.ndarray] = None,
        index_type: str = "flat",
        metric: str = "ip",
    ):
        if embeddings is None:
            embeddings = self.corpus_embeddings

        if embeddings is None:
            raise ValueError("No embeddings provided. Call encode_corpus first.")

        dim = embeddings.shape[1]
        console.print(f"[cyan]Building {index_type} index (dim={dim}, metric={metric})...[/cyan]")

        if metric == "ip":
            if index_type == "flat":
                self.index = faiss.IndexFlatIP(dim)
            elif index_type == "hnsw":
                self.index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
                self.index.hnsw.efConstruction = 200
                self.index.hnsw.efSearch = 128
            else:
                raise ValueError(f"Unknown index type: {index_type}")
        else:
            if index_type == "flat":
                self.index = faiss.IndexFlatL2(dim)
            elif index_type == "hnsw":
                self.index = faiss.IndexHNSWFlat(dim, 32)
                self.index.hnsw.efConstruction = 200
                self.index.hnsw.efSearch = 128
            else:
                raise ValueError(f"Unknown index type: {index_type}")

        self.index.add(embeddings)
        console.print(f"[green]Built index with {self.index.ntotal} vectors[/green]")

    def search(
        self,
        query_embeddings: np.ndarray,
        query_ids: list[str],
        top_k: int = 100,
    ) -> dict[str, dict[str, float]]:
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")

        if self.doc_ids is None:
            raise ValueError("Document IDs not set. Call encode_corpus first.")

        console.print(f"[cyan]Searching for top-{top_k} results...[/cyan]")

        scores, indices = self.index.search(query_embeddings, top_k)

        results = {}
        for i, query_id in enumerate(query_ids):
            results[query_id] = {}
            for j in range(top_k):
                idx = indices[i, j]
                if idx >= 0:
                    doc_id = self.doc_ids[idx]
                    score = float(scores[i, j])
                    results[query_id][doc_id] = score

        console.print(f"[green]Retrieved results for {len(results)} queries[/green]")
        return results

    def retrieve(
        self,
        corpus: dict,
        queries: dict,
        top_k: int = 100,
        batch_size: int = 32,
        index_type: str = "flat",
    ) -> dict[str, dict[str, float]]:
        if self.corpus_embeddings is None:
            self.encode_corpus(corpus, batch_size=batch_size)

        if self.index is None:
            self.build_index(index_type=index_type)

        query_ids, query_embeddings = self.encode_queries(queries, batch_size=batch_size)
        return self.search(query_embeddings, query_ids, top_k=top_k)

    def save_embeddings(self, path: str | Path):
        if self.corpus_embeddings is None:
            raise ValueError("No embeddings to save")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        np.save(path, self.corpus_embeddings)

        ids_path = path.with_suffix(".ids.txt")
        with open(ids_path, "w") as f:
            for doc_id in self.doc_ids:
                f.write(f"{doc_id}\n")

        console.print(f"[green]Saved embeddings to {path}[/green]")

    def load_embeddings(self, path: str | Path):
        path = Path(path)

        self.corpus_embeddings = np.load(path)

        ids_path = path.with_suffix(".ids.txt")
        with open(ids_path, "r") as f:
            self.doc_ids = [line.strip() for line in f]

        console.print(f"[green]Loaded embeddings: shape {self.corpus_embeddings.shape}[/green]")

    def save_index(self, path: str | Path):
        if self.index is None:
            raise ValueError("No index to save")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path))
        console.print(f"[green]Saved index to {path}[/green]")

    def load_index(self, path: str | Path):
        path = Path(path)
        self.index = faiss.read_index(str(path))
        console.print(f"[green]Loaded index with {self.index.ntotal} vectors[/green]")
