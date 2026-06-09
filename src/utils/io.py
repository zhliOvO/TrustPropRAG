import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from rich.console import Console

console = Console()


def load_runfile(runfile_path: str | Path) -> dict[str, dict[str, float]]:
    runfile_path = Path(runfile_path)

    if not runfile_path.exists():
        raise FileNotFoundError(f"Runfile not found: {runfile_path}")

    results = {}

    with open(runfile_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                query_id = parts[0]
                doc_id = parts[2]
                score = float(parts[4])

                if query_id not in results:
                    results[query_id] = {}
                results[query_id][doc_id] = score

    return results


def save_runfile(
    results: dict[str, dict[str, float]],
    output_path: str | Path,
    run_name: str = "beir_rag_run",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for query_id, doc_scores in results.items():
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

            for rank, (doc_id, score) in enumerate(sorted_docs, 1):
                f.write(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {run_name}\n")

    console.print(f"[green]Saved runfile to {output_path}[/green]")
    return output_path


def load_results_json(results_path: str | Path) -> dict:
    results_path = Path(results_path)

    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    with open(results_path, "r") as f:
        return json.load(f)


def save_results_json(
    results: dict,
    output_path: str | Path,
    metadata: Optional[dict] = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }

    if metadata:
        output_data["metadata"] = metadata

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    console.print(f"[green]Saved results to {output_path}[/green]")
    return output_path


def get_run_paths(
    dataset_name: str,
    base_dir: str = "data/runs",
    encoder_name: Optional[str] = None,
    reranker_name: Optional[str] = None,
) -> dict[str, Path]:
    base_path = Path(base_dir) / dataset_name

    encoder_suffix = ""
    if encoder_name:
        encoder_suffix = "_" + encoder_name.replace("/", "_").replace("-", "_")

    reranker_suffix = ""
    if reranker_name:
        reranker_suffix = "_" + reranker_name.replace("/", "_").replace("-", "_")

    return {
        "retrieval_runfile": base_path / f"retrieval{encoder_suffix}.run",
        "reranked_runfile": base_path / f"reranked{encoder_suffix}{reranker_suffix}.run",
        "retrieval_results": base_path / f"retrieval{encoder_suffix}_results.json",
        "reranked_results": base_path / f"reranked{encoder_suffix}{reranker_suffix}_results.json",
        "embeddings_dir": Path("data/embeddings") / dataset_name / encoder_name.replace("/", "_") if encoder_name else None,
        "index_path": Path("data/indexes") / dataset_name / f"{encoder_name.replace('/', '_')}.index" if encoder_name else None,
    }


def format_metrics_table(metrics: dict[str, float], title: str = "Evaluation Results") -> str:
    from rich.table import Table

    table = Table(title=title)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for metric, value in sorted(metrics.items()):
        table.add_row(metric, f"{value:.4f}")

    return table
