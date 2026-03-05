"""Démonstration d'optimisation ONNX: export + benchmark de latence.

Usage:
    python -m scripts.onnx_optimization_demo
    python -m scripts.onnx_optimization_demo --runs 500 --batch-size 1
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
from onnxmltools import convert_lightgbm
from onnxconverter_common.data_types import FloatTensorType


def _load_bundle(path: Path) -> dict:
    artifact = joblib.load(path)
    if not isinstance(artifact, dict):
        raise ValueError("Le modèle doit être un bundle dict avec model/feature_names/medians.")
    required = {"model", "feature_names", "medians"}
    if not required.issubset(artifact.keys()):
        raise ValueError(f"Bundle incomplet. Clés requises: {required}")
    return artifact


def _build_input_frame(feature_names: list[str], medians: dict[str, float], batch_size: int) -> pd.DataFrame:
    row = {name: float(medians.get(name, 0.0)) for name in feature_names}
    frame = pd.DataFrame([row for _ in range(batch_size)], columns=feature_names)
    return frame


def _export_to_onnx(model, n_features: int, output_path: Path) -> None:
    initial_types = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_lightgbm(model, initial_types=initial_types)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(onnx_model.SerializeToString())


def _benchmark_sklearn(model, frame: pd.DataFrame, runs: int) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    probs = None
    for _ in range(runs):
        probs = model.predict_proba(frame)[:, 1]
    elapsed_ms = (time.perf_counter() - start) * 1000
    return probs, elapsed_ms / runs


def _benchmark_onnx(onnx_path: Path, frame: pd.DataFrame, runs: int) -> tuple[np.ndarray, float]:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [out.name for out in session.get_outputs()]

    payload = frame.to_numpy(dtype=np.float32)

    def _extract_positive_proba(outputs_obj):
        for out in outputs_obj:
            if isinstance(out, np.ndarray):
                if out.ndim == 2 and out.shape[1] >= 2:
                    return out[:, 1]
                if out.ndim == 1:
                    return out
            if isinstance(out, list) and out and isinstance(out[0], dict):
                return np.asarray([float(item.get(1, 0.0)) for item in out], dtype=float)
        raise RuntimeError("Impossible d'extraire la probabilité positive depuis la sortie ONNX.")

    start = time.perf_counter()
    probs = None
    for _ in range(runs):
        outputs = session.run(output_names, {input_name: payload})
        probs = _extract_positive_proba(outputs)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return np.asarray(probs, dtype=float), elapsed_ms / runs


def main(model_path: str, onnx_path: str, runs: int, batch_size: int) -> None:
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Modèle introuvable: {model_file}")

    bundle = _load_bundle(model_file)
    model = bundle["model"]
    feature_names = list(bundle["feature_names"])
    medians = dict(bundle["medians"])

    frame = _build_input_frame(feature_names, medians, batch_size=batch_size)

    output_file = Path(onnx_path)
    _export_to_onnx(model, n_features=len(feature_names), output_path=output_file)

    skl_probs, skl_ms = _benchmark_sklearn(model, frame, runs=runs)
    onnx_probs, onnx_ms = _benchmark_onnx(output_file, frame, runs=runs)

    max_abs_diff = float(np.max(np.abs(skl_probs - onnx_probs)))
    speedup = (skl_ms / onnx_ms) if onnx_ms > 0 else float("inf")

    print("=== ONNX Optimization Demo ===")
    print(f"Model path           : {model_file}")
    print(f"ONNX path            : {output_file}")
    print(f"Features             : {len(feature_names)}")
    print(f"Batch size           : {batch_size}")
    print(f"Runs                 : {runs}")
    print(f"Sklearn avg latency  : {skl_ms:.4f} ms")
    print(f"ONNX avg latency     : {onnx_ms:.4f} ms")
    print(f"Speedup (x)          : {speedup:.2f}")
    print(f"Max |proba diff|     : {max_abs_diff:.8f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Démo d'optimisation ONNX pour le modèle notebook")
    parser.add_argument("--model-path", type=str, default="models/notebook_model.joblib")
    parser.add_argument("--onnx-path", type=str, default="models/notebook_model.onnx")
    parser.add_argument("--runs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    main(
        model_path=args.model_path,
        onnx_path=args.onnx_path,
        runs=args.runs,
        batch_size=args.batch_size,
    )
