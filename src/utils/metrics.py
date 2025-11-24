from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


@dataclass
class ClassificationMetrics:
    accuracy: float
    f1_macro: float
    f1_weighted: float
    precision_macro: float
    recall_macro: float
    confusion: np.ndarray


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ClassificationMetrics:
    """
    Calcule un ensemble standard de métriques de classification
    pour la comparaison DQN vs PPO vs baseline.
    """
    acc = accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec_mac = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_mac = recall_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return ClassificationMetrics(
        accuracy=acc,
        f1_macro=f1_mac,
        f1_weighted=f1_w,
        precision_macro=prec_mac,
        recall_macro=rec_mac,
        confusion=cm,
    )


def metrics_to_markdown_row(name: str, m: ClassificationMetrics) -> str:
    """
    Formate une ligne de tableau Markdown pour le rapport de comparaison.
    """
    return (
        f"| {name} | "
        f"{m.accuracy:.4f} | "
        f"{m.f1_macro:.4f} | "
        f"{m.f1_weighted:.4f} | "
        f"{m.precision_macro:.4f} | "
        f"{m.recall_macro:.4f} |\n"
    )