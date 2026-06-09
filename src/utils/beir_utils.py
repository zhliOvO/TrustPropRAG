from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()


def load_beir_dataset(
    dataset_path: str | Path,
    split: str = "test",
    preserve_metadata: bool = True,
) -> tuple[dict, dict, dict]:
    from beir.datasets.data_loader import GenericDataLoader

    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    console.print(f"[cyan]Loading dataset from {dataset_path} (split={split})...[/cyan]")

    corpus, queries, qrels = GenericDataLoader(str(dataset_path)).load(split=split)

    if preserve_metadata:
        queries = _load_queries_with_metadata(dataset_path)

    console.print(f"[green]Loaded {len(corpus)} documents, {len(queries)} queries[/green]")

    return corpus, queries, qrels


def _load_queries_with_metadata(dataset_path: Path) -> dict:
    import json
    from tqdm import tqdm

    queries_path = dataset_path / "queries.jsonl"

    if not queries_path.exists():
        return {}

    answers_map = {}
    answers_path = dataset_path / "answers.json"
    if answers_path.exists():
        try:
            with open(answers_path, "r") as f:
                answers_map = json.load(f)
            console.print(f"[dim]Loaded {len(answers_map)} ground-truth answers from {answers_path.name}[/dim]")
        except Exception as e:
            console.print(f"[yellow]Could not load {answers_path}: {e}[/yellow]")

    queries = {}
    with open(queries_path, "r") as f:
        for line in tqdm(f, desc="Loading queries with metadata"):
            data = json.loads(line.strip())
            qid = data.get("_id", "")
            text = data.get("text", "")
            metadata = dict(data.get("metadata", {}))

            if qid in answers_map:
                metadata["answer"] = answers_map[qid]

            if metadata:
                queries[qid] = {"text": text, "metadata": metadata}
            else:
                queries[qid] = text

    return queries


def get_corpus_texts(corpus: dict) -> list[str]:
    texts = []
    for doc_id, doc in corpus.items():
        title = doc.get("title", "")
        text = doc.get("text", "")

        if title:
            combined = f"{title} {text}"
        else:
            combined = text

        texts.append(combined)

    return texts


def get_corpus_ids(corpus: dict) -> list[str]:
    return list(corpus.keys())


def convert_results_to_beir_format(
    results: dict[str, list[tuple[str, float]]]
) -> dict[str, dict[str, float]]:
    beir_results = {}
    for query_id, doc_scores in results.items():
        beir_results[query_id] = {doc_id: score for doc_id, score in doc_scores}
    return beir_results


def evaluate_results(
    qrels: dict,
    results: dict[str, dict[str, float]],
    k_values: list[int] = [1, 3, 5, 10, 20, 50, 100],
) -> dict[str, float]:
    from beir.retrieval.evaluation import EvaluateRetrieval

    evaluator = EvaluateRetrieval()

    ndcg, map_score, recall, precision = evaluator.evaluate(
        qrels, results, k_values
    )

    all_metrics = {}

    for k in k_values:
        if f"NDCG@{k}" in ndcg:
            all_metrics[f"NDCG@{k}"] = ndcg[f"NDCG@{k}"]
        if f"MAP@{k}" in map_score:
            all_metrics[f"MAP@{k}"] = map_score[f"MAP@{k}"]
        if f"Recall@{k}" in recall:
            all_metrics[f"Recall@{k}"] = recall[f"Recall@{k}"]
        if f"P@{k}" in precision:
            all_metrics[f"P@{k}"] = precision[f"P@{k}"]

    return all_metrics


def print_metrics(metrics: dict[str, float], title: str = "Evaluation Results"):
    from rich.table import Table

    table = Table(title=title)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    metric_groups = {}
    for metric, value in metrics.items():
        if "@" in metric:
            name, k = metric.split("@")
            if name not in metric_groups:
                metric_groups[name] = []
            metric_groups[name].append((int(k), value))

    for name in ["NDCG", "MAP", "Recall", "P"]:
        if name in metric_groups:
            for k, value in sorted(metric_groups[name]):
                table.add_row(f"{name}@{k}", f"{value:.4f}")

    console.print(table)


def save_beir_runfile(
    results: dict[str, dict[str, float]],
    output_path: str | Path,
    run_name: str = "beir_run",
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for query_id, doc_scores in results.items():
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            for rank, (doc_id, score) in enumerate(sorted_docs, 1):
                f.write(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {run_name}\n")

    console.print(f"[green]Saved runfile: {output_path}[/green]")
