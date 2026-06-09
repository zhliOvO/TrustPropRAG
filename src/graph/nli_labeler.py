from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

console = Console()

SUPPORTIVE = 1
CONTRADICTORY = -1
UNRELATED = 0


class NLILabeler:
    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        device: Optional[str] = None,
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device

        console.print(f"[cyan]Loading NLI model: {model_name}[/cyan]")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.label_map = self._detect_label_map(model_name)

    def _detect_label_map(self, model_name: str) -> Dict[str, int]:
        if hasattr(self.model.config, "id2label"):
            id2label = self.model.config.id2label
            label_map = {}
            for idx, label in id2label.items():
                label_lower = label.lower()
                if "entail" in label_lower:
                    label_map["entailment"] = int(idx)
                elif "contra" in label_lower:
                    label_map["contradiction"] = int(idx)
                elif "neutral" in label_lower:
                    label_map["neutral"] = int(idx)
            if len(label_map) == 3:
                return label_map

        return {
            "contradiction": 0,
            "neutral": 1,
            "entailment": 2,
        }

    def predict_batch(
        self,
        premise_hypothesis_pairs: List[Tuple[str, str]],
        batch_size: int = 16,
        max_length: int = 512,
    ) -> List[Tuple[int, float]]:
        import torch

        results = []

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("NLI inference", total=len(premise_hypothesis_pairs))

            for i in range(0, len(premise_hypothesis_pairs), batch_size):
                batch = premise_hypothesis_pairs[i:i + batch_size]

                premises = [p for p, _ in batch]
                hypotheses = [h for _, h in batch]

                inputs = self.tokenizer(
                    premises,
                    hypotheses,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)

                for j in range(len(batch)):
                    prob = probs[j].cpu().numpy()
                    label_idx = int(prob.argmax())
                    confidence = float(prob[label_idx])

                    if label_idx == self.label_map["entailment"]:
                        results.append((SUPPORTIVE, confidence))
                    elif label_idx == self.label_map["contradiction"]:
                        results.append((CONTRADICTORY, confidence))
                    else:
                        results.append((UNRELATED, confidence))

                progress.update(task, advance=len(batch))

        return results

    def label_all_pairs(
        self,
        corpus: Dict[str, Dict],
        batch_size: int = 16,
    ) -> List[Tuple[str, str, int, float, float]]:
        doc_ids = sorted(corpus.keys())
        pairs: List[Tuple[str, str]] = []
        pair_ids: List[Tuple[str, str]] = []

        for i, src in enumerate(doc_ids):
            for j, dst in enumerate(doc_ids):
                if i >= j:
                    continue
                pairs.append((self._get_text(corpus[src]), self._get_text(corpus[dst])))
                pair_ids.append((src, dst))

        if not pairs:
            return []

        console.print(f"[cyan]Labeling {len(pairs)} document pairs with NLI...[/cyan]")
        nli_results = self.predict_batch(pairs, batch_size=batch_size)

        labeled_edges = [
            (src, dst, label, confidence, 1.0)
            for (src, dst), (label, confidence) in zip(pair_ids, nli_results)
        ]

        n_support = sum(1 for _, _, l, _, _ in labeled_edges if l == SUPPORTIVE)
        n_contra = sum(1 for _, _, l, _, _ in labeled_edges if l == CONTRADICTORY)
        n_neutral = sum(1 for _, _, l, _, _ in labeled_edges if l == UNRELATED)
        console.print(
            f"[green]NLI edges: {n_support} supportive, {n_contra} contradictory, {n_neutral} neutral[/green]"
        )

        return labeled_edges

    def _get_text(self, doc: Dict) -> str:
        title = doc.get("title", "")
        text = doc.get("text", "")
        combined = f"{title} {text}".strip() if title else text
        return combined[:500]


def save_labeled_edges(
    edges: List[Tuple[str, str, int, float, float]],
    output_path: str | Path,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    edge_data = [
        {
            "src": src,
            "dst": dst,
            "label": label,
            "nli_confidence": nli_conf,
            "similarity": sim,
        }
        for src, dst, label, nli_conf, sim in edges
    ]

    with open(output_path, "w") as f:
        json.dump(edge_data, f, indent=2)

    console.print(f"[green]Saved {len(edges)} labeled edges to {output_path}[/green]")


def load_labeled_edges(
    input_path: str | Path,
) -> List[Tuple[str, str, int, float, float]]:
    with open(input_path, "r") as f:
        edge_data = json.load(f)

    return [
        (e["src"], e["dst"], e["label"], e["nli_confidence"], e["similarity"])
        for e in edge_data
    ]
