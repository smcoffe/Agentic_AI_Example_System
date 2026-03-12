"""
data_loader.py — Dataset loading, splitting, and normalisation.

Supports:
  • scikit-learn built-in datasets  (iris, digits, breast_cancer, wine)
  • Local CSV files                 (auto-detects label column)
  • Local NumPy .npz files          (keys: 'X' and 'y')
"""

from __future__ import annotations

import csv
import logging
import os
from typing import List, Tuple

import numpy as np

from config import DataConfig, SKLEARN_DATASETS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_dataset(cfg: DataConfig) -> Tuple[
    np.ndarray, np.ndarray,   # X_train, y_train
    np.ndarray, np.ndarray,   # X_val,   y_val
    np.ndarray, np.ndarray,   # X_test,  y_test
    List[str],                # class_names
    dict,                     # dataset_info
]:
    """Load, split, and (optionally) normalise a dataset.

    Returns six arrays and two metadata objects:
        X_train, y_train, X_val, y_val, X_test, y_test, class_names, info
    """
    if cfg.source == "sklearn":
        X, y, class_names = _load_sklearn(cfg.sklearn_dataset)
    elif cfg.source == "local":
        X, y, class_names = _load_local(cfg.local_path, cfg.label_column)
    else:
        raise ValueError(f"Unknown data source '{cfg.source}'. Use 'sklearn' or 'local'.")

    logger.info("Dataset loaded: %d samples, %d features, %d classes",
                X.shape[0], X.shape[1], len(class_names))

    X_train, y_train, X_val, y_val, X_test, y_test = _split(
        X, y, cfg.test_size, cfg.val_size, cfg.random_state
    )

    if cfg.normalize:
        X_train, X_val, X_test = _normalize(X_train, X_val, X_test)

    info = {
        "n_samples":       X.shape[0],
        "n_features":      X.shape[1],
        "n_classes":       len(class_names),
        "class_names":     class_names,
        "train_size":      X_train.shape[0],
        "val_size":        X_val.shape[0],
        "test_size":       X_test.shape[0],
        "class_balance":   _class_balance(y, class_names),
    }

    logger.info("Split → train=%d  val=%d  test=%d",
                info["train_size"], info["val_size"], info["test_size"])

    return X_train, y_train, X_val, y_val, X_test, y_test, class_names, info


# ---------------------------------------------------------------------------
# scikit-learn datasets
# ---------------------------------------------------------------------------

def _load_sklearn(name: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if name not in SKLEARN_DATASETS:
        raise ValueError(
            f"Unknown sklearn dataset '{name}'. "
            f"Available: {list(SKLEARN_DATASETS.keys())}"
        )
    try:
        from sklearn import datasets as sk_datasets
    except ImportError:
        raise ImportError(
            "scikit-learn is required for built-in datasets. "
            "Install with: pip install scikit-learn"
        )

    loader_name = SKLEARN_DATASETS[name]
    loader = getattr(sk_datasets, loader_name)
    data   = loader()

    X = data.data.astype(np.float64)
    y = data.target.astype(np.int64)

    class_names = [str(c) for c in data.target_names]
    logger.info("Loaded sklearn dataset '%s'", name)
    return X, y, class_names


# ---------------------------------------------------------------------------
# Local dataset loaders
# ---------------------------------------------------------------------------

def _load_local(path: str, label_column: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext == ".npz":
        return _load_npz(path)
    elif ext == ".npy":
        return _load_npy(path)
    elif ext in (".csv", ".tsv", ".txt"):
        delimiter = "\t" if ext == ".tsv" else ","
        return _load_csv(path, label_column, delimiter)
    else:
        # Try CSV as default
        logger.warning("Unknown extension '%s' — attempting CSV parse.", ext)
        return _load_csv(path, label_column, ",")


def _load_npz(path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    data = np.load(path, allow_pickle=True)
    if "X" not in data or "y" not in data:
        raise ValueError("NPZ file must contain arrays named 'X' and 'y'.")
    X = data["X"].astype(np.float64)
    y = data["y"].astype(np.int64)
    class_names = [str(i) for i in np.unique(y)]
    logger.info("Loaded .npz dataset from '%s'", path)
    return X, y, class_names


def _load_npy(path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Single .npy: treat last column as label."""
    arr = np.load(path, allow_pickle=True)
    X   = arr[:, :-1].astype(np.float64)
    raw = arr[:, -1]
    y, class_names = _encode_labels(raw)
    logger.info("Loaded .npy dataset from '%s'", path)
    return X, y, class_names


def _load_csv(path: str, label_column: str, delimiter: str = ","
              ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    rows: List[List[str]] = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh, delimiter=delimiter)
        header = next(reader)
        for row in reader:
            if row:
                rows.append(row)

    if not rows:
        raise ValueError("CSV file is empty (no data rows).")

    # Determine label column index
    if label_column and label_column in header:
        label_idx = header.index(label_column)
    else:
        label_idx = len(header) - 1   # default: last column
        logger.info("Using last column ('%s') as label.", header[label_idx])

    feature_indices = [i for i in range(len(header)) if i != label_idx]
    raw_X = [[row[i] for i in feature_indices] for row in rows]
    raw_y = [row[label_idx] for row in rows]

    X = np.array(raw_X, dtype=np.float64)
    y, class_names = _encode_labels(np.array(raw_y))

    logger.info("Loaded CSV dataset from '%s' — features: %s, label: '%s'",
                path, [header[i] for i in feature_indices], header[label_idx])
    return X, y, class_names


# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------

def _encode_labels(raw: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Convert arbitrary string / float labels to 0-indexed integer classes."""
    # Try numeric first
    try:
        numeric = raw.astype(np.float64)
        unique  = np.sort(np.unique(numeric))
        mapping = {v: i for i, v in enumerate(unique)}
        y       = np.array([mapping[v] for v in numeric], dtype=np.int64)
        names   = [str(v) for v in unique]
    except (ValueError, TypeError):
        unique  = sorted(set(raw))
        mapping = {v: i for i, v in enumerate(unique)}
        y       = np.array([mapping[v] for v in raw], dtype=np.int64)
        names   = [str(v) for v in unique]
    return y, names


# ---------------------------------------------------------------------------
# Train / Val / Test split
# ---------------------------------------------------------------------------

def _split(X: np.ndarray, y: np.ndarray,
           test_size: float, val_size: float,
           random_state: int):
    rng = np.random.default_rng(random_state)
    n   = X.shape[0]

    idx = rng.permutation(n)
    n_test = max(1, int(n * test_size))
    n_val  = max(1, int(n * val_size))

    test_idx  = idx[:n_test]
    val_idx   = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    return (X[train_idx], y[train_idx],
            X[val_idx],   y[val_idx],
            X[test_idx],  y[test_idx])


# ---------------------------------------------------------------------------
# Normalisation (fit on train, apply to val/test)
# ---------------------------------------------------------------------------

def _normalize(X_train: np.ndarray,
               X_val:   np.ndarray,
               X_test:  np.ndarray,
               eps: float = 1e-8):
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0)
    std[std < eps] = 1.0          # avoid division by zero for constant features

    X_train = (X_train - mean) / std
    X_val   = (X_val   - mean) / std
    X_test  = (X_test  - mean) / std
    return X_train, X_val, X_test


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _class_balance(y: np.ndarray, class_names: List[str]) -> dict:
    counts = {class_names[c]: int(np.sum(y == c)) for c in range(len(class_names))}
    return counts


def list_sklearn_datasets() -> List[str]:
    return list(SKLEARN_DATASETS.keys())
