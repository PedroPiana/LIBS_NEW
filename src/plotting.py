from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def save_figure(fig: plt.Figure, output_path: str | Path) -> Path:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)
    return output_file


def save_confusion_matrix_plot(
    y_true: Iterable[int] | np.ndarray,
    y_pred: Iterable[int] | np.ndarray,
    output_path: str | Path,
    title: str | None = None,
    display_labels: Iterable[str] | None = None,
):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    if display_labels is not None:
        ax.set_yticklabels(list(display_labels))
        ax.set_xticklabels(list(display_labels))
    if title:
        ax.set_title(title)

    return save_figure(fig, output_path)


def plot_ranking(results: dict[str, float], output_path: str | Path, title: str | None = None):
    names = list(results.keys())
    scores = [results[k] for k in names]

    # Ordena por valor decrescente
    order = np.argsort(scores)[::-1]
    names_sorted = [names[i] for i in order]
    scores_sorted = [scores[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(names_sorted))))
    sns.barplot(x=scores_sorted, y=names_sorted, ax=ax)
    ax.set_xlabel("Score")
    if title:
        ax.set_title(title)

    return save_figure(fig, output_path)


def plot_spectra(
    X,
    wavelengths: Iterable[float] | None = None,
    labels: Iterable[str] | None = None,
    output_path: str | Path | None = None,
    title: str | None = None,
):
    fig, ax = plt.subplots(figsize=(8, 4))
    X = np.asarray(X)

    if wavelengths is None:
        x = np.arange(X.shape[1])
    else:
        x = np.asarray(list(wavelengths))

    if X.ndim == 1:
        ax.plot(x, X, label=(labels[0] if labels else None))
    else:
        for i in range(X.shape[0]):
            label = (list(labels)[i] if labels is not None else None)
            ax.plot(x, X[i, :], label=label, alpha=0.8)

    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Intensity")
    if title:
        ax.set_title(title)
    if labels is not None:
        ax.legend()

    if output_path is not None:
        return save_figure(fig, output_path)

    return fig


__all__ = [
    "save_confusion_matrix_plot",
    "plot_ranking",
    "plot_spectra",
    "save_figure",
]
