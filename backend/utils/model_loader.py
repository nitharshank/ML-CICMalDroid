"""
Model loader + ensemble inference.
Mirrors the pipeline from the Streamlit app: scaler -> variance threshold ->
[RF, XGB, MLP] -> mean probability.
"""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np

log = logging.getLogger(__name__)


class EnsembleModel:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.features: list[str] = []
        self.scaler = None
        self.vt = None
        self.rf = None
        self.xgb = None
        self.mlp = None

    def load(self) -> None:
        proc = self.base_dir / "processed"
        models = self.base_dir / "models"

        with open(proc / "drebin_features.json") as f:
            self.features = json.load(f)

        self.scaler = joblib.load(proc / "drebin_scaler.joblib")
        self.vt = joblib.load(proc / "drebin_variance_selector.joblib")
        self.rf = joblib.load(models / "rf_drebin.joblib")
        self.xgb = joblib.load(models / "xgb_drebin.joblib")
        self.mlp = joblib.load(models / "mlp_drebin.joblib")

        log.info("Loaded ensemble. Feature count: %d", len(self.features))

    def predict(self, vec: np.ndarray) -> dict:
        if self.scaler is None:
            raise RuntimeError("Models not loaded. Call .load() first.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x = vec.astype(np.float32).reshape(1, -1)
            x_sc = self.scaler.transform(x)
            x_vt = self.vt.transform(x_sc)
            p_rf = float(self.rf.predict_proba(x_vt)[0, 1])
            p_xgb = float(self.xgb.predict_proba(x_vt)[0, 1])
            p_mlp = float(self.mlp.predict_proba(x_vt)[0, 1])
            p_ens = (p_rf + p_xgb + p_mlp) / 3.0

        return {
            "rf": p_rf,
            "xgb": p_xgb,
            "mlp": p_mlp,
            "ensemble": p_ens,
        }
