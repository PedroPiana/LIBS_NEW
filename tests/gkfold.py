#!/usr/bin/env python3
"""Script de execução do experimento equivalente à célula principal de `notebooks/gkfold.ipynb`.

Este arquivo contém apenas o trecho necessário para preparar os dados
e executar `run_experiment` a partir dos módulos em `src/`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple


def find_project_root(start: Path = Path.cwd()) -> Path:
    for p in [start] + list(start.parents):
        if (p / "src").is_dir():
            return p
    return start


def prepare_dataset(laser: int, data_root: Path) -> Tuple:
    import numpy as np
    import pandas as pd

    from src.utils import extrair_numero

    file_path = data_root / f"laser{laser}.dat"
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path, sep=";", header=None)
    y_raw = df.iloc[0, 1:].values.astype(str)
    X_full = df.iloc[1:, 1:].values.astype(float).T

    extra_a_labels_266 = {
        "Pilao_sample",
        "Evolutto_sample",
        "Do_Ponto_sample",
        "America_sample",
        "Coamo_sample",
    }

    X = []
    y = []
    groups = []

    for xi, yi in zip(X_full, y_raw):
        num = extrair_numero(yi)
        if laser == 266:
            if (yi.startswith("A") and num is not None and num <= 12) or yi in extra_a_labels_266:
                X.append(xi)
                y.append("A")
                groups.append(str(num) if num is not None else yi)
            elif yi.startswith("R") and num is not None and num <= 9:
                X.append(xi)
                y.append("R")
                groups.append(str(num))
        elif laser == 532:
            if yi.startswith("A") and num is not None and num <= 8:
                X.append(xi)
                y.append("A")
                groups.append(str(num))
            elif yi.startswith("R") and num is not None and num <= 8:
                X.append(xi)
                y.append("R")
                groups.append(str(num))
        else:
            if yi.startswith("A") and num is not None and num <= 12:
                X.append(xi)
                y.append("A")
                groups.append(str(num))
            elif yi.startswith("R") and num is not None and num <= 9:
                X.append(xi)
                y.append("R")
                groups.append(str(num))

    X = np.array(X)
    y = np.array(y)
    groups = np.array(groups)

    return X, y, groups


def main() -> None:
    proj_root = find_project_root()
    if str(proj_root) not in sys.path:
        sys.path.insert(0, str(proj_root))

    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold

    from src.preprocessing import make_savgol, make_snv
    from src.models import PLSDAClassifier, make_random_forest, make_svm_linear
    from src.training import run_experiment

    parser = argparse.ArgumentParser()
    parser.add_argument("--laser", type=int, default=1064, help="Laser: 266, 532 or 1064")
    parser.add_argument("--data-root", type=str, default=str(proj_root / "dataset"))
    parser.add_argument("--models-out", type=str, default=str(proj_root / "models_groupkfold"))
    parser.add_argument("--plots-out", type=str, default=str(proj_root / "plots_groupkfold"))
    args = parser.parse_args()

    laser = int(args.laser)
    if laser not in [266, 532, 1064]:
        raise ValueError("O valor deve ser 266, 532 ou 1064")

    X, y, groups = prepare_dataset(laser, Path(args.data_root))

    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("É necessário pelo menos 2 grupos para usar StratifiedGroupKFold.")

    n_splits = min(8, len(unique_groups))
    sgkf = StratifiedGroupKFold(n_splits=n_splits)

    preprocessamentos = {
        "none": [],
        "snv": [make_snv()],
        "savgol_snv": [make_savgol(), make_snv()],
    }

    modelos = {
        "svm_linear": make_svm_linear(),
        "random_forest": make_random_forest(),
        "pls_da": PLSDAClassifier(n_components=2),
    }

    out_models = Path(args.models_out) / f"laser{laser}"
    out_plots = Path(args.plots_out) / f"laser{laser}"
    out_models.mkdir(parents=True, exist_ok=True)
    out_plots.mkdir(parents=True, exist_ok=True)

    results_data = run_experiment(
        X=X,
        y=y,
        preprocessamentos=preprocessamentos,
        modelos=modelos,
        cv=sgkf,
        output_models_dir=out_models,
        output_plots_dir=out_plots,
        laser=laser,
        labels=np.unique(y),
        experiment_name="stratified_groupkfold_binary",
        display_labels=["Arabica", "Robusta"],
        groups=groups,
        summary_filename="all_pipelines_summary.json",
        ranking_filename="ranking_final_groupkfold.png",
    )

    results = results_data["results"]
    print("==============================")
    print("RANKING FINAL")
    print("==============================")
    for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True):
        print(f"{k}: {v:.4f}")

    print(f"Resumo consolidado salvo em: {results_data['summary_path']}")
    print(f"Plot final salvo em: {results_data['ranking_path']}")


if __name__ == "__main__":
    main()
