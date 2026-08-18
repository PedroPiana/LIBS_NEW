from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def ensure_parent_dir(path: str | Path) -> Path:
    p = Path(path)
    p_parent = p.parent
    p_parent.mkdir(parents=True, exist_ok=True)
    return p


def read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def save_json(obj: Any, output_path: str | Path) -> Path:
    output_file = ensure_parent_dir(output_path)
    with Path(output_file).open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
    return Path(output_file)


def load_json(input_path: str | Path) -> Any:
    with Path(input_path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_joblib(obj: Any, output_path: str | Path) -> Path:
    output_file = ensure_parent_dir(output_path)
    joblib.dump(obj, output_file)
    return Path(output_file)


def load_joblib(path: str | Path) -> Any:
    return joblib.load(path)


def save_keras_model(model, output_path: str | Path, save_format: str | None = None) -> Path:
    """Salva um modelo Keras/TensorFlow. Se o pacote não estiver disponível, lança ImportError."""
    try:
        from tensorflow import keras
    except Exception as e:
        raise ImportError("TensorFlow/Keras não disponível: %s" % e)

    output_file = ensure_parent_dir(output_path)
    # Keras deteta o formato pelo sufixo .keras/.h5 ou pelo argumento save_format
    model.save(output_file, save_format=save_format)  # type: ignore[arg-type]
    return Path(output_file)


def load_keras_model(path: str | Path):
    try:
        from tensorflow import keras
    except Exception as e:
        raise ImportError("TensorFlow/Keras não disponível: %s" % e)

    return keras.models.load_model(path)


__all__ = [
    "ensure_parent_dir",
    "read_csv",
    "save_json",
    "load_json",
    "save_joblib",
    "load_joblib",
    "save_keras_model",
    "load_keras_model",
]
