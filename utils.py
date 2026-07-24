from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np


# Extrai o primeiro número encontrado em uma string (ex.: "A12" -> 12).
def extrair_numero(label: str | None) -> int | None:
    if label is None:
        return None

    match = re.search(r"\d+", str(label))
    return int(match.group()) if match else None


# Converte objetos do NumPy para tipos nativos do Python,
# permitindo que sejam salvos em arquivos JSON.
def normalize_missing_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: normalize_missing_values(item) for key, item in value.items()}

    if isinstance(value, list):
        return [normalize_missing_values(item) for item in value]

    if isinstance(value, tuple):
        return tuple(normalize_missing_values(item) for item in value)

    if isinstance(value, set):
        return {normalize_missing_values(item) for item in value}

    if isinstance(value, np.ndarray):
        return [normalize_missing_values(item) for item in value.tolist()]

    if isinstance(value, np.generic):
        return value.item()

    return value


# Garante que a pasta de destino exista antes de salvar um arquivo.
def ensure_parent_dir(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


# Salva qualquer estrutura de dados em um arquivo JSON.
def save_json(output_path: str | Path, payload: Any) -> Path:
    output_file = ensure_parent_dir(output_path)
    normalized_payload = normalize_missing_values(payload)

    with output_file.open("w", encoding="utf-8") as json_file:
        json.dump(normalized_payload, json_file, ensure_ascii=False, indent=2)

    return output_file


# Carrega e retorna o conteúdo de um arquivo JSON.
def load_json(input_path: str | Path) -> Any:
    with Path(input_path).open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


# Função específica para salvar resumos (summary) de pipelines em JSON.
def save_pipeline_summary_json(output_path: str | Path, payload: Any) -> Path:
    return save_json(output_path, payload)


# Define quais funções podem ser importadas com "from modulo import *".
__all__ = [
    "ensure_parent_dir",
    "extrair_numero",
    "load_json",
    "normalize_missing_values",
    "save_json",
    "save_pipeline_summary_json",
]