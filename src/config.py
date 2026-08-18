from __future__ import annotations

from pathlib import Path
from typing import Iterable


# Diretórios base do projeto (assume layout: workspace root contains `dataset`, `models`, `plots`, etc.)
BASE_DIR: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = BASE_DIR / "dataset"
MODELS_DIR: Path = BASE_DIR / "models"
PLOTS_DIR: Path = BASE_DIR / "plots"
MODELS_GROUPKFOLD_DIR: Path = BASE_DIR / "models_groupkfold"

# Configurações e constantes padrão
RANDOM_SEED: int = 42
LASERS: tuple[int, ...] = (266, 532, 1064)


def ensure_dirs(dirs: Iterable[Path] | None = None) -> None:
    """Garante que diretórios importantes existam.

    Args:
        dirs: iterável de Paths a criar. Se None, cria os diretórios padrão.
    """
    if dirs is None:
        dirs = (DATA_DIR, MODELS_DIR, PLOTS_DIR, MODELS_GROUPKFOLD_DIR)

    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def list_dirs():
    print("BASE_DIR: ", BASE_DIR, "\nDATA_DIR: ",DATA_DIR, "\nMODELS_DIR: ", MODELS_DIR, 
          "\nPLOTS_DIR:", PLOTS_DIR)


__all__ = ["BASE_DIR", "DATA_DIR", "MODELS_DIR", "PLOTS_DIR", "RANDOM_SEED", "LASERS", "list_dirs","ensure_dirs"]
