"""Funcoes de treino e comparacao de modelos."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline

from evaluation import (
    build_pipeline_summary,
    plot_ranking,
    save_confusion_matrix_plot,
    save_consolidated_summary,
    save_pipeline_summary,
)


def build_pipeline(preprocess_steps, model):
    return Pipeline([
        *preprocess_steps,
        ("model", model),
    ])


def evaluate_pipeline(pipe, X, y, cv, groups=None, scoring: str = "accuracy"):
    if groups is None:
        scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring)
        y_pred = cross_val_predict(pipe, X, y, cv=cv)
    else:
        scores = cross_val_score(pipe, X, y, cv=cv, groups=groups, scoring=scoring)
        y_pred = cross_val_predict(pipe, X, y, cv=cv, groups=groups)

    return scores, y_pred


def save_trained_pipeline(pipe, output_path: str | Path):
    joblib.dump(pipe, output_path)
    return output_path


def run_experiment(
    X,
    y,
    preprocessamentos: dict[str, list[tuple[str, Any]]],
    modelos: dict[str, Any],
    cv,
    output_models_dir: str | Path,
    output_plots_dir: str | Path,
    laser: int,
    labels,
    experiment_name: str,
    display_labels,
    groups=None,
    summary_filename: str = "all_pipelines_summary.json",
    ranking_filename: str = "ranking_final.png",
):
    output_models_dir = Path(output_models_dir)
    output_plots_dir = Path(output_plots_dir)
    output_models_dir.mkdir(parents=True, exist_ok=True)
    output_plots_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, float] = {}
    pipeline_summaries: list[dict[str, Any]] = []

    for prep_name, prep_steps in preprocessamentos.items():
        for model_name, model in modelos.items():
            pipeline_name = f"{prep_name}_{model_name}"
            pipe = build_pipeline(prep_steps, model)

            try:
                scores, y_pred = evaluate_pipeline(pipe, X, y, cv, groups=groups)
            except Exception:
                continue

            mean_acc = float(scores.mean())
            std_acc = float(scores.std())
            results[pipeline_name] = mean_acc

            try:
                pipe.fit(X, y)
                save_trained_pipeline(pipe, output_models_dir / f"{pipeline_name}.joblib")
            except Exception:
                pass

            try:
                save_confusion_matrix_plot(
                    y,
                    y_pred,
                    output_plots_dir / f"{pipeline_name}_cm.png",
                    f"Matriz de Confusão - {pipeline_name}",
                    display_labels=display_labels,
                )
            except Exception:
                pass

            pipeline_summaries.append(
                build_pipeline_summary(
                    pipeline_name,
                    scores,
                    mean_acc,
                    std_acc,
                    laser,
                    labels,
                    cv.get_n_splits(groups=groups) if groups is not None else cv.get_n_splits(),
                )
            )

    summary_path = output_plots_dir / summary_filename
    save_consolidated_summary(
        summary_path,
        experiment_name,
        laser,
        labels,
        cv.get_n_splits(groups=groups) if groups is not None else cv.get_n_splits(),
        pipeline_summaries,
    )

    ranking_path = output_plots_dir / ranking_filename
    plot_ranking(
        results,
        ranking_path,
        title=f"Comparacao dos Pipelines - Laser {laser}",
    )

    return {
        "results": results,
        "pipeline_summaries": pipeline_summaries,
        "summary_path": summary_path,
        "ranking_path": ranking_path,
    }


__all__ = [
    "build_pipeline",
    "evaluate_pipeline",
    "run_experiment",
    "save_trained_pipeline",
]