"""
models/classifier.py
--------------------
Module A: Multi-class crop classifier using Random Forest.

Mathematical Basis — Bagging (Bootstrap Aggregating)
------------------------------------------------------
Given a training dataset D = {(x_i, y_i)}_{i=1}^n and B trees:

  For b = 1, ..., B:
    1. Draw bootstrap sample D_b ~ D with replacement (|D_b| = n)
    2. Grow decision tree T_b on D_b, selecting m = sqrt(p) features
       randomly at each split node (feature randomization)
    3. Aggregate: f(x) = argmax_c sum_{b=1}^B I(T_b(x) = c)  [majority vote]

Variance Reduction (key insight):
  Let σ² = variance of a single tree, ρ = avg pairwise tree correlation.
  Var(f_bag) = ρ·σ² + (1−ρ)/B · σ²

  As B -> ∞:   Var(f_bag) -> ρ·σ²    (irreducible floor)
  Feature randomization drives ρ -> 0, so Var(f_bag) -> 0.
  Bias unchanged (deep trees are low-bias). Net: optimal bias-variance.

Feature Importance:
  Impurity-based: avg decrease in Gini impurity from splits on feature j,
  weighted by the fraction of samples reaching each split node.
  Gini(t) = 1 − sum_c p_{tc}²
  This is used in the profit engine to identify the most influential
  soil/weather factors for each recommended crop.
"""

import logging
import numpy as np
import joblib
from pathlib import Path
from typing import Tuple, List, Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, accuracy_score,
                              f1_score, confusion_matrix)

logger = logging.getLogger(__name__)


class CropClassifier:

    def __init__(self, cfg: dict, n_classes: int):
        """
        Parameters
        ----------
        cfg : dict   classifier section from config.yaml
        n_classes : int  total number of unique crops in the dataset
        """
        self.cfg      = cfg
        self.n_classes = n_classes
        self.model    = RandomForestClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            max_features=cfg["max_features"],   # sqrt(p) for decorrelation
            min_samples_leaf=cfg["min_samples_leaf"],
            class_weight=cfg["class_weight"],   # handles imbalanced classes
            bootstrap=True,                      # enables bagging
            oob_score=True,                      # out-of-bag estimate (free CV)
            random_state=42,
            n_jobs=-1,
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fit the Random Forest on training data.
        OOB score is computed automatically and logged as a free
        cross-validation estimate without requiring a separate split.
        """
        logger.info("Training Random Forest: %d estimators, max_depth=%s",
                    self.cfg["n_estimators"], self.cfg["max_depth"])
        self.model.fit(X_train, y_train)
        logger.info("OOB accuracy estimate: %.4f", self.model.oob_score_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict_top_k(self, X: np.ndarray, k: int = 2) -> List[List[int]]:
        """
        Returns top-k class indices sorted by posterior probability P(y|x).
        Used by the profit engine to evaluate top-k candidate crops.
        """
        proba = self.model.predict_proba(X)  # (n_samples, n_classes)
        return [np.argsort(row)[::-1][:k].tolist() for row in proba]

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                 class_names: List[str]) -> Dict:
        """
        Computes and returns evaluation metrics.

        Metrics
        -------
        - Accuracy: fraction of correctly classified samples
        - Weighted F1: F1 averaged over classes, weighted by support
          F1_c = 2·(P_c·R_c)/(P_c+R_c)
          Weighted F1 is preferred over macro for imbalanced classes.
        - Per-class precision, recall, F1 (full classification report)
        - Confusion matrix
        """
        preds = self.model.predict(X_test)
        acc   = accuracy_score(y_test, preds)
        f1    = f1_score(y_test, preds, average="weighted")
        report = classification_report(y_test, preds,
                                       target_names=class_names, output_dict=True)
        cm = confusion_matrix(y_test, preds)
        logger.info("[Module A] Test Accuracy: %.4f | Weighted F1: %.4f", acc, f1)
        return {
            "accuracy":       acc,
            "f1_weighted":    f1,
            "oob_score":      self.model.oob_score_,
            "report":         report,
            "confusion_matrix": cm,
            "feature_importance": self.model.feature_importances_,
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info("Classifier saved -> %s", path)

    @classmethod
    def load(cls, path: str, cfg: dict, n_classes: int) -> "CropClassifier":
        obj = cls(cfg, n_classes)
        obj.model = joblib.load(path)
        logger.info("Classifier loaded <- %s", path)
        return obj
