from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.table import Table

console = Console()


def export_tau_json(
    tau: Dict[str, float],
    output_path: str | Path,
    metadata: Optional[Dict[str, Any]] = None,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tau_native = {str(k): float(v) for k, v in tau.items()}

    with open(output_path, "w") as f:
        json.dump(tau_native, f, indent=2)

    console.print(f"[green]Exported {len(tau)} tau scores to {output_path}[/green]")

    if metadata:
        meta_path = output_path.with_suffix(".meta.json")
        metadata["timestamp"] = datetime.now().isoformat()
        metadata["n_documents"] = len(tau)

        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)


def export_diagnostics(
    tau: Dict[str, float],
    partition: Optional[Dict[str, int]],
    loss_history: List[float],
    output_dir: str | Path,
    labeled_edges: Optional[List] = None,
    cluster_stats: Optional[Dict] = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    export_tau_json(tau, output_dir / "tau.json")

    if partition is not None:
        with open(output_dir / "communities.json", "w") as f:
            json.dump({
                "partition": partition,
                "n_communities": len(set(partition.values())),
            }, f, indent=2)
        n_communities = len(set(partition.values()))
    else:
        n_communities = None

    loss_history_native = [float(x) for x in loss_history]
    with open(output_dir / "loss_history.json", "w") as f:
        json.dump({"loss_history": loss_history_native}, f, indent=2)

    if cluster_stats:
        stats_json = {
            f"{p}_{q}": stats
            for (p, q), stats in cluster_stats.items()
        }
        with open(output_dir / "cluster_stats.json", "w") as f:
            json.dump(stats_json, f, indent=2)

    tau_values = [float(v) for v in tau.values()]
    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_documents": len(tau),
        "n_communities": n_communities,
        "tau_stats": {
            "min": float(min(tau_values)),
            "max": float(max(tau_values)),
            "mean": float(sum(tau_values) / len(tau_values)),
            "median": float(sorted(tau_values)[len(tau_values) // 2]),
        },
        "optimization": {
            "final_loss": float(loss_history[-1]) if loss_history else None,
            "n_iterations": len(loss_history),
        },
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    console.print(f"[green]Exported diagnostics to {output_dir}[/green]")


def print_tau_summary(tau: Dict[str, float], top_k: int = 10):
    tau_values = [float(v) for v in tau.values()]

    console.print("\n[bold cyan]Trust Score Summary[/bold cyan]")
    console.print(f"  Documents: {len(tau)}")
    console.print(f"  Min τ: {min(tau_values):.4f}")
    console.print(f"  Max τ: {max(tau_values):.4f}")
    console.print(f"  Mean τ: {sum(tau_values)/len(tau_values):.4f}")
    console.print(f"  Median τ: {sorted(tau_values)[len(tau_values)//2]:.4f}")

    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    hist = [0] * (len(bins) - 1)
    for v in tau_values:
        for i in range(len(bins) - 1):
            if bins[i] <= v < bins[i+1] or (i == len(bins)-2 and v == bins[i+1]):
                hist[i] += 1
                break

    console.print("\n[bold]Distribution:[/bold]")
    for i in range(len(bins) - 1):
        pct = hist[i] / len(tau_values) * 100
        bar = "█" * int(pct / 2)
        console.print(f"  [{bins[i]:.1f}-{bins[i+1]:.1f}): {hist[i]:5d} ({pct:5.1f}%) {bar}")

    sorted_tau = sorted([(k, float(v)) for k, v in tau.items()], key=lambda x: x[1], reverse=True)

    table = Table(title=f"Top {top_k} Most Trusted Documents")
    table.add_column("Doc ID", style="cyan")
    table.add_column("τ", style="green")

    for doc_id, score in sorted_tau[:top_k]:
        table.add_row(doc_id, f"{score:.4f}")

    console.print(table)

    table = Table(title=f"Top {top_k} Least Trusted Documents")
    table.add_column("Doc ID", style="cyan")
    table.add_column("τ", style="red")

    for doc_id, score in sorted_tau[-top_k:]:
        table.add_row(doc_id, f"{score:.4f}")

    console.print(table)


def load_tau_json(tau_path: str | Path) -> Dict[str, float]:
    tau_path = Path(tau_path)

    if not tau_path.exists():
        raise FileNotFoundError(f"Tau file not found: {tau_path}")

    with open(tau_path, "r") as f:
        tau = json.load(f)

    return {str(k): float(v) for k, v in tau.items()}
