from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


# Implementa o filtro Savitzky-Golay 
class SavgolFilter(BaseEstimator, TransformerMixin):
    def __init__(self, window_length: int = 11, polyorder: int = 2, deriv: int = 0):
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv

    # Não realiza treinamento, apenas retorna a própria instância.
    def fit(self, X, y=None):
        return self

    # Aplica o filtro Savitzky-Golay aos espectros.
    def transform(self, X):
        return savgol_filter(
            X,
            window_length=self.window_length,
            polyorder=self.polyorder,
            deriv=self.deriv,
            axis=1,
        )


# Implementa a normalização SNV 
class SNV(BaseEstimator, TransformerMixin):
    # Não realiza treinamento, apenas retorna a própria instância.
    def fit(self, X, y=None):
        return self

    # Centraliza e normaliza cada espectro individualmente.
    def transform(self, X):
        mean = np.mean(X, axis=1, keepdims=True)
        std = np.std(X, axis=1, ddof=1, keepdims=True)
        std[std == 0] = 1
        return (X - mean) / std


def make_savgol(window_length: int = 11, polyorder: int = 2):
    return ("savgol", SavgolFilter(window_length=window_length, polyorder=polyorder))


def make_snv():
    return ("snv", SNV())


def make_scaler():
    return ("scaler", StandardScaler())


# Define quais classes e funções podem ser importadas com "from modulo import *".
__all__ = [
    "SNV",
    "SavgolFilter",
    "make_savgol",
    "make_scaler",
    "make_snv",
]