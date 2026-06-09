from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

from trustprop_rag.types import HumanFeedback, LabeledEdge

console = Console()


@dataclass
class TrustOptimizerConfig:
    lambda_feedback: float = 1.0
    tau_positive: float = 0.9
    tau_negative: float = 0.1
    tau_min: float = 0.3
    tau_max: float = 0.7
    learning_rate: float = 0.01
    max_iters: int = 1000
    convergence_tol: float = 1e-6
    log_every: int = 100


@dataclass
class TrustOptimizationResult:
    tau: Dict[str, float]
    final_loss: float
    loss_history: List[float]
    converged: bool
    n_iters: int


class TrustPropOptimizer:
    def __init__(self, config: Optional[TrustOptimizerConfig] = None):
        self.config = config or TrustOptimizerConfig()

    def optimize(
        self,
        labeled_edges: List[LabeledEdge],
        feedback: Optional[List[HumanFeedback]] = None,
        all_doc_ids: Optional[List[str]] = None,
    ) -> TrustOptimizationResult:
        feedback = feedback or []
        doc_ids = self._collect_doc_ids(labeled_edges, feedback, all_doc_ids)
        if not doc_ids:
            return TrustOptimizationResult({}, 0.0, [], True, 0)

        doc_to_idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}
        n_docs = len(doc_ids)

        edge_src, edge_dst, edge_label, edge_weight = self._build_edge_arrays(
            labeled_edges, doc_to_idx
        )
        fb_idx, fb_target = self._build_feedback_arrays(feedback, doc_to_idx)

        console.print(f"[cyan]Optimizing trust for {n_docs} documents (PGD)...[/cyan]")
        console.print(
            f"[dim]{len(edge_src)} relation edges, {len(fb_idx)} feedback anchors[/dim]"
        )

        tau = self._initialize_tau(n_docs, edge_src, edge_dst, edge_label, edge_weight)

        loss_history: List[float] = []
        prev_loss = float("inf")
        converged = False

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("PGD", total=self.config.max_iters)

            for iteration in range(self.config.max_iters):
                loss, grad = self._loss_and_gradient(
                    tau, edge_src, edge_dst, edge_label, edge_weight, fb_idx, fb_target
                )
                loss_history.append(loss)

                if abs(prev_loss - loss) < self.config.convergence_tol:
                    console.print(f"[green]Converged at iteration {iteration}[/green]")
                    converged = True
                    break

                prev_loss = loss
                tau -= self.config.learning_rate * grad
                tau = np.clip(tau, 0.0, 1.0)
                progress.update(task, advance=1)

                if (iteration + 1) % self.config.log_every == 0:
                    progress.console.print(f"[dim]Iter {iteration + 1}: loss={loss:.6f}[/dim]")

        tau_dict = {doc_ids[i]: float(tau[i]) for i in range(n_docs)}
        return TrustOptimizationResult(
            tau=tau_dict,
            final_loss=loss_history[-1] if loss_history else 0.0,
            loss_history=loss_history,
            converged=converged,
            n_iters=len(loss_history),
        )

    def _collect_doc_ids(
        self,
        labeled_edges: List[LabeledEdge],
        feedback: List[HumanFeedback],
        all_doc_ids: Optional[List[str]],
    ) -> List[str]:
        doc_ids_set = set(all_doc_ids or [])
        for src, dst, label, _, _ in labeled_edges:
            if label != 0:
                doc_ids_set.add(src)
                doc_ids_set.add(dst)
        for fb in feedback:
            doc_ids_set.add(fb.doc_id)
        return sorted(doc_ids_set)

    def _build_edge_arrays(
        self,
        labeled_edges: List[LabeledEdge],
        doc_to_idx: Dict[str, int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        edge_src, edge_dst, edge_label, edge_weight = [], [], [], []
        for src, dst, label, nli_conf, sim in labeled_edges:
            if label == 0 or src not in doc_to_idx or dst not in doc_to_idx:
                continue
            edge_src.append(doc_to_idx[src])
            edge_dst.append(doc_to_idx[dst])
            edge_label.append(label)
            edge_weight.append(nli_conf)

        return (
            np.array(edge_src, dtype=np.int32),
            np.array(edge_dst, dtype=np.int32),
            np.array(edge_label, dtype=np.float32),
            np.array(edge_weight, dtype=np.float32),
        )

    def _build_feedback_arrays(
        self,
        feedback: List[HumanFeedback],
        doc_to_idx: Dict[str, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        fb_idx, fb_target = [], []
        cfg = self.config
        for fb in feedback:
            if fb.doc_id not in doc_to_idx:
                continue
            fb_idx.append(doc_to_idx[fb.doc_id])
            target = cfg.tau_positive if fb.rating == "good" else cfg.tau_negative
            fb_target.append(target)
        return np.array(fb_idx, dtype=np.int32), np.array(fb_target, dtype=np.float32)

    def _initialize_tau(
        self,
        n_docs: int,
        edge_src: np.ndarray,
        edge_dst: np.ndarray,
        edge_label: np.ndarray,
        edge_weight: np.ndarray,
    ) -> np.ndarray:
        node_score = np.zeros(n_docs, dtype=np.float32)
        for i in range(len(edge_src)):
            signed = edge_label[i] * edge_weight[i]
            node_score[edge_src[i]] += signed
            node_score[edge_dst[i]] += signed

        score_range = node_score.max() - node_score.min()
        if score_range > 0:
            normalized = (node_score - node_score.min()) / score_range
            tau = self.config.tau_min + (self.config.tau_max - self.config.tau_min) * normalized
        else:
            tau = np.full(n_docs, 0.5, dtype=np.float32)

        return tau.astype(np.float32)

    def _loss_and_gradient(
        self,
        tau: np.ndarray,
        edge_src: np.ndarray,
        edge_dst: np.ndarray,
        edge_label: np.ndarray,
        edge_weight: np.ndarray,
        fb_idx: np.ndarray,
        fb_target: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        grad = np.zeros_like(tau)
        loss = 0.0

        if len(edge_src) > 0:
            sup_mask = edge_label > 0
            if sup_mask.any():
                src, dst, w = edge_src[sup_mask], edge_dst[sup_mask], edge_weight[sup_mask]
                diff = tau[src] - tau[dst]
                loss += np.sum(w * diff**2)
                np.add.at(grad, src, 2 * w * diff)
                np.add.at(grad, dst, -2 * w * diff)

            con_mask = edge_label < 0
            if con_mask.any():
                src, dst, w = edge_src[con_mask], edge_dst[con_mask], edge_weight[con_mask]
                diff = tau[src] + tau[dst] - 1.0
                loss += np.sum(w * diff**2)
                np.add.at(grad, src, 2 * w * diff)
                np.add.at(grad, dst, 2 * w * diff)

        if len(fb_idx) > 0:
            fb_diff = tau[fb_idx] - fb_target
            loss += self.config.lambda_feedback * np.sum(fb_diff**2)
            np.add.at(grad, fb_idx, self.config.lambda_feedback * 2 * fb_diff)

        return float(loss), grad
