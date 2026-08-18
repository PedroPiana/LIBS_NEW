"""Modelos customizados reutilizaveis extraidos dos notebooks."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import (
        Conv1D,
        Dense,
        Dropout,
        Flatten,
        InputLayer,
        MaxPooling1D,
    )

    TENSORFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover - depende do ambiente local
    tf = None
    Sequential = None
    EarlyStopping = None
    Conv1D = None
    Dense = None
    Dropout = None
    Flatten = None
    InputLayer = None
    MaxPooling1D = None
    TENSORFLOW_AVAILABLE = False


class PLSDAClassifier(BaseEstimator, ClassifierMixin):
    """PLS-DA binario usando PLSRegression e limiar em 0.5."""

    def __init__(self, n_components: int = 2):
        self.n_components = n_components

    def fit(self, X, y):
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        if len(self.label_encoder_.classes_) != 2:
            raise ValueError("PLS-DA nesta implementacao espera apenas duas classes.")

        self.pls_ = PLSRegression(n_components=self.n_components)
        self.pls_.fit(X, y_encoded)
        return self

    def predict(self, X):
        scores = self.pls_.predict(X).ravel()
        predicted = (scores >= 0.5).astype(int)
        return self.label_encoder_.inverse_transform(predicted)


class PLSDAMulticlass(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components: int = 10):
        self.n_components = n_components

    def fit(self, X, y):
        self.lb_ = LabelBinarizer()
        Y = self.lb_.fit_transform(y)

        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        self.classes_ = self.lb_.classes_
        self.models_ = []

        for i in range(Y.shape[1]):
            pls = PLSRegression(
                n_components=min(
                    self.n_components,
                    X.shape[0] - 1,
                    X.shape[1],
                )
            )
            pls.fit(X, Y[:, i])
            self.models_.append(pls)

        return self

    def predict(self, X):
        scores = np.column_stack([model.predict(X).ravel() for model in self.models_])
        idx = np.argmax(scores, axis=1)
        return self.classes_[idx]

    def predict_proba(self, X):
        scores = np.column_stack([model.predict(X).ravel() for model in self.models_])
        scores = np.maximum(scores, 0)

        row_sum = scores.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1

        return scores / row_sum


class CNN1DClassifier(BaseEstimator, ClassifierMixin):
    """Wrapper scikit-learn para uma CNN 1D com TensorFlow/Keras."""

    def __init__(
        self,
        epochs: int = 30,
        batch_size: int = 16,
        validation_split: float = 0.2,
        verbose: int = 0,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.verbose = verbose

    def _build_model(self, input_length: int):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError(
                "TensorFlow nao esta instalado. Instale tensorflow para usar o modelo 1D-CNN."
            )

        model = Sequential(
            [
                InputLayer(shape=(input_length, 1)),
                Conv1D(16, kernel_size=5, activation="relu", padding="same"),
                MaxPooling1D(pool_size=2),
                Conv1D(32, kernel_size=3, activation="relu", padding="same"),
                MaxPooling1D(pool_size=2),
                Flatten(),
                Dense(64, activation="relu"),
                Dropout(0.3),
                Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def fit(self, X, y):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError(
                "TensorFlow nao esta instalado. Instale tensorflow para usar o modelo 1D-CNN."
            )

        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y).astype(np.float32)
        if len(self.label_encoder_.classes_) != 2:
            raise ValueError("O modelo 1D-CNN nesta implementacao espera apenas duas classes.")

        X_cnn = X.astype(np.float32)[..., np.newaxis]
        self.model_ = self._build_model(X_cnn.shape[1])
        callback = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        self.model_.fit(
            X_cnn,
            y_encoded,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            verbose=self.verbose,
            callbacks=[callback],
        )
        return self

    def predict(self, X):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError(
                "TensorFlow nao esta instalado. Instale tensorflow para usar o modelo 1D-CNN."
            )

        X_cnn = X.astype(np.float32)[..., np.newaxis]
        probabilities = self.model_.predict(X_cnn, verbose=0).ravel()
        predicted = (probabilities >= 0.5).astype(int)
        return self.label_encoder_.inverse_transform(predicted)


def make_svm_linear():
    return SVC(kernel="linear", gamma="scale")


def make_svm_rbf():
    return SVC(kernel="rbf", gamma="scale")


def make_svm_poly():
    return SVC(kernel="poly", gamma="scale")


def make_svm_sigmoid():
    return SVC(kernel="sigmoid", gamma="scale")


def make_random_forest():
    return RandomForestClassifier(n_estimators=200, random_state=42)


def build_binary_model_registry(include_cnn: bool = False):
    models = {
        "svm_linear": make_svm_linear(),
        "svm_rbf": make_svm_rbf(),
        "svm_poly": make_svm_poly(),
        "svm_sigmoid": make_svm_sigmoid(),
        "random_forest": make_random_forest(),
        "pls_da": PLSDAClassifier(n_components=2),
    }

    if include_cnn and TENSORFLOW_AVAILABLE:
        models["cnn_1d"] = CNN1DClassifier(
            epochs=30,
            batch_size=16,
            validation_split=0.2,
            verbose=0,
        )

    return models


def build_multiclass_model_registry(include_cnn: bool = False):
    models = {
        "pls_da": PLSDAMulticlass(n_components=10),
    }

    if include_cnn and TENSORFLOW_AVAILABLE:
        models["cnn_1d"] = CNN1DClassifier(
            epochs=30,
            batch_size=16,
            validation_split=0.2,
            verbose=0,
        )

    return models


__all__ = [
    "CNN1DClassifier",
    "build_binary_model_registry",
    "build_multiclass_model_registry",
    "make_random_forest",
    "make_svm_linear",
    "make_svm_poly",
    "make_svm_rbf",
    "make_svm_sigmoid",
    "PLSDAClassifier",
    "PLSDAMulticlass",
    "TENSORFLOW_AVAILABLE",
]