"""Funcoes de avaliacao, metricas e artefatos visuais."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable




def compute_confusion_matrix(y_true, y_pred):
    from sklearn.metrics import confusion_matrix

    return confusion_matrix(y_true, y_pred)


def save_confusion_matrix_plot(
    y_true,
    y_pred,
    output_path: str | Path,
    title: str,
    display_labels: Iterable[str],
):
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    cm = compute_confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(display_labels))

    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax)
    plt.title(title)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


def build_pipeline_summary(
    pipeline_name: str,
    scores: np.ndarray,
    mean_accuracy: float,
    std_accuracy: float,
    laser: int,
    labels: Iterable[str],
    n_splits: int,
) -> dict[str, Any]:
    return {
        "pipeline": pipeline_name,
        "laser": laser,
        "labels": [str(label) for label in labels],
        "n_splits": n_splits,
        "accuracy_scores": [float(score) for score in scores],
        "mean_accuracy": float(mean_accuracy),
        "std_accuracy": float(std_accuracy),
    }


def save_pipeline_summary(output_path: str | Path, summary: dict[str, Any]):
    from src.utils import save_pipeline_summary_json

    return save_pipeline_summary_json(output_path, summary)


def save_consolidated_summary(
    output_path: str | Path,
    experiment: str,
    laser: int,
    labels: Iterable[str],
    n_splits: int,
    pipeline_summaries: list[dict[str, Any]],
):
    summary = {
        "experiment": experiment,
        "laser": laser,
        "labels": [str(label) for label in labels],
        "n_splits": n_splits,
        "pipelines": pipeline_summaries,
    }
    from src.utils import save_pipeline_summary_json

    return save_pipeline_summary_json(output_path, summary)


def plot_ranking(
    results: dict[str, float],
    output_path: str | Path,
    title: str,
    xlabel: str = "Pipeline",
    ylabel: str = "Accuracy",
):
    if not results:
        return None

    import matplotlib.pyplot as plt

    names = list(results.keys())
    values = list(results.values())

    plt.figure(figsize=(14, 6))
    bars = plt.bar(names, values)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f"{height:.3f}",
            ha="center",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


__all__ = [
    "build_pipeline_summary",
    "compute_confusion_matrix",
    "plot_ranking",
    "save_confusion_matrix_plot",
    "save_consolidated_summary",
    "save_pipeline_summary",
]