from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

console = Console()


class GraphBuilder:
    def __init__(
        self,
        encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ):
        import torch
        from sentence_transformers import SentenceTransformer

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        console.print(f"[cyan]GraphBuilder using device: {self.device}[/cyan]")
        console.print(f"[cyan]Loading encoder: {encoder_name}[/cyan]")
        self.encoder = SentenceTransformer(encoder_name, device=self.device)

        self.doc_ids: Optional[List[str]] = None
        self.embeddings: Optional[np.ndarray] = None
        self.index = None

    def encode_corpus(
        self,
        corpus: Dict[str, Dict],
        batch_size: int = 32,
    ) -> np.ndarray:
        self.doc_ids = list(corpus.keys())
        texts = []

        for doc_id in self.doc_ids:
            doc = corpus[doc_id]
            title = doc.get("title", "")
            text = doc.get("text", "")
            combined = f"{title} {text}".strip() if title else text
            texts.append(combined)

        console.print(f"[cyan]Encoding {len(texts)} documents for graph...[/cyan]")

        embeddings = self.encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        self.embeddings = embeddings.astype(np.float32)
        return self.embeddings

    def build_index(self):
        import faiss

        if self.embeddings is None:
            raise ValueError("No embeddings. Call encode_corpus first.")

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

        console.print(f"[green]Built index with {self.index.ntotal} vectors[/green]")

    def find_neighbors(
        self,
        n_neighbors: int = 50,
        exclude_self: bool = True,
    ) -> Dict[str, List[str]]:
        if self.index is None:
            self.build_index()

        k = n_neighbors + 1 if exclude_self else n_neighbors

        console.print(f"[cyan]Finding {n_neighbors} neighbors per document...[/cyan]")

        scores, indices = self.index.search(self.embeddings, k)

        neighbors = {}
        for i, doc_id in enumerate(self.doc_ids):
            neighbor_ids = []
            for j in range(k):
                idx = indices[i, j]
                if exclude_self and idx == i:
                    continue
                if idx >= 0 and len(neighbor_ids) < n_neighbors:
                    neighbor_ids.append(self.doc_ids[idx])
            neighbors[doc_id] = neighbor_ids

        return neighbors

    def build_edge_list(
        self,
        n_neighbors: int = 50,
    ) -> List[Tuple[str, str, float]]:
        if self.index is None:
            self.build_index()

        k = n_neighbors + 1

        console.print(f"[cyan]Building edge list (top-{n_neighbors} per node)...[/cyan]")

        scores, indices = self.index.search(self.embeddings, k)

        edges = []
        for i, doc_id in enumerate(self.doc_ids):
            for j in range(k):
                idx = indices[i, j]
                if idx == i or idx < 0:
                    continue
                neighbor_id = self.doc_ids[idx]
                sim_score = float(scores[i, j])
                edges.append((doc_id, neighbor_id, sim_score))

        console.print(f"[green]Built {len(edges)} edges[/green]")
        return edges

    def save_edges(
        self,
        edges: List[Tuple[str, str, float]],
        output_path: str | Path,
    ):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        edge_data = [
            {"src": src, "dst": dst, "similarity": score}
            for src, dst, score in edges
        ]

        with open(output_path, "w") as f:
            json.dump(edge_data, f, indent=2)

        console.print(f"[green]Saved {len(edges)} edges to {output_path}[/green]")

    def load_edges(self, input_path: str | Path) -> List[Tuple[str, str, float]]:
        with open(input_path, "r") as f:
            edge_data = json.load(f)

        return [
            (e["src"], e["dst"], e["similarity"])
            for e in edge_data
        ]


def build_neighbor_graph(
    corpus: Dict[str, Dict],
    n_neighbors: int = 50,
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    device: Optional[str] = None,
) -> Tuple[Dict[str, List[str]], List[Tuple[str, str, float]]]:
    builder = GraphBuilder(encoder_name=encoder_name, device=device)
    builder.encode_corpus(corpus, batch_size=batch_size)

    neighbors = builder.find_neighbors(n_neighbors=n_neighbors)
    edges = builder.build_edge_list(n_neighbors=n_neighbors)

    return neighbors, edges
