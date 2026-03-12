"""
neural_network.py — Pure-NumPy feedforward neural network.

Architecture
------------
  Input → [Hidden layer × N] → Softmax output

Features
--------
  • Configurable hidden layers (depth and width)
  • Activations:  relu | sigmoid | tanh
  • Optimisers:   adam | sgd
  • L2 regularisation
  • Inverted dropout on hidden layers
  • He / Xavier weight initialisation
  • Full training loop with mini-batch SGD
  • Per-epoch metrics and early-stopping signals

Everything is plain NumPy — no deep-learning framework required.
"""

from __future__ import annotations

import copy
import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import HyperParameters

logger = logging.getLogger(__name__)


# ===========================================================================
# Neural Network class
# ===========================================================================

class NeuralNetwork:
    """
    Fully-connected feedforward network for multi-class classification.

    Parameters
    ----------
    input_size   : int                  — number of input features
    hidden_layers: List[int]            — neurons in each hidden layer
    num_classes  : int                  — number of output classes
    hyperparams  : HyperParameters      — see config.py
    random_state : int                  — for reproducible weight init
    """

    def __init__(
        self,
        input_size:    int,
        hidden_layers: List[int],
        num_classes:   int,
        hyperparams:   HyperParameters,
        random_state:  int = 42,
    ) -> None:
        np.random.seed(random_state)

        self.hp          = hyperparams
        self.input_size  = input_size
        self.num_classes = num_classes

        self._build_layers(input_size, hidden_layers, num_classes)
        self._init_adam_state()

        self._cache: dict = {}   # populated during forward

        logger.debug(
            "Network built: %d→%s→%d | act=%s | opt=%s | lr=%.5f | dropout=%.2f",
            input_size, hidden_layers, num_classes,
            hyperparams.activation, hyperparams.optimizer,
            hyperparams.learning_rate, hyperparams.dropout_rate,
        )

    # -----------------------------------------------------------------------
    # Initialisation helpers
    # -----------------------------------------------------------------------

    def _build_layers(
        self, input_size: int, hidden_layers: List[int], num_classes: int
    ) -> None:
        layer_sizes = [input_size] + list(hidden_layers) + [num_classes]
        self.weights: List[np.ndarray] = []
        self.biases:  List[np.ndarray] = []

        for i in range(len(layer_sizes) - 1):
            fan_in  = layer_sizes[i]
            fan_out = layer_sizes[i + 1]

            # He init for relu, Xavier for sigmoid/tanh
            if self.hp.activation == "relu":
                std = np.sqrt(2.0 / fan_in)
            else:
                std = np.sqrt(1.0 / fan_in)

            W = np.random.randn(fan_in, fan_out) * std
            b = np.zeros((1, fan_out))
            self.weights.append(W)
            self.biases.append(b)

        self.n_layers = len(self.weights)

    def _init_adam_state(self) -> None:
        if self.hp.optimizer == "adam":
            self._m_W = [np.zeros_like(W) for W in self.weights]
            self._m_b = [np.zeros_like(b) for b in self.biases]
            self._v_W = [np.zeros_like(W) for W in self.weights]
            self._v_b = [np.zeros_like(b) for b in self.biases]
            self._adam_t = 0

    # -----------------------------------------------------------------------
    # Activation functions
    # -----------------------------------------------------------------------

    def _activation(self, Z: np.ndarray, deriv: bool = False) -> np.ndarray:
        name = self.hp.activation
        if name == "relu":
            if deriv:
                return (Z > 0).astype(np.float64)
            return np.maximum(0.0, Z)
        elif name == "sigmoid":
            sig = 1.0 / (1.0 + np.exp(-np.clip(Z, -500, 500)))
            if deriv:
                return sig * (1.0 - sig)
            return sig
        elif name == "tanh":
            if deriv:
                return 1.0 - np.tanh(Z) ** 2
            return np.tanh(Z)
        raise ValueError(f"Unknown activation '{name}'")

    @staticmethod
    def _softmax(Z: np.ndarray) -> np.ndarray:
        Z_s   = Z - Z.max(axis=1, keepdims=True)   # numerical stability
        exp_Z = np.exp(Z_s)
        return exp_Z / exp_Z.sum(axis=1, keepdims=True)

    # -----------------------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------------------

    def forward(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Run a forward pass.

        During training, dropout masks are applied to hidden-layer outputs
        and stored in self._cache for use in backpropagation.

        Returns the output probability matrix  (n_samples × n_classes).
        """
        self._cache = {"A": [X], "Z": [], "masks": {}}
        A = X

        for i in range(self.n_layers):
            Z = A @ self.weights[i] + self.biases[i]
            self._cache["Z"].append(Z)

            if i == self.n_layers - 1:
                # Output layer → softmax
                A = self._softmax(Z)
            else:
                # Hidden layer → activation + optional dropout
                A = self._activation(Z)
                if training and self.hp.dropout_rate > 0.0:
                    p    = self.hp.dropout_rate
                    mask = (np.random.rand(*A.shape) > p) / (1.0 - p)
                    self._cache["masks"][i] = mask
                    A = A * mask

            self._cache["A"].append(A)

        return A  # probabilities

    # -----------------------------------------------------------------------
    # Loss
    # -----------------------------------------------------------------------

    def compute_loss(
        self, probs: np.ndarray, y_true: np.ndarray
    ) -> float:
        """Cross-entropy loss with L2 regularisation."""
        m   = y_true.shape[0]
        eps = 1e-15
        ce  = -np.mean(np.log(
            np.clip(probs[np.arange(m), y_true], eps, 1.0)
        ))
        l2  = (self.hp.l2_lambda / (2.0 * m)) * sum(
            np.sum(W ** 2) for W in self.weights
        )
        return float(ce + l2)

    # -----------------------------------------------------------------------
    # Backward pass  (reverse-mode autodiff, hand-coded)
    # -----------------------------------------------------------------------

    def backward(
        self, y_true: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Compute gradients via backpropagation.

        Uses the combined softmax + cross-entropy gradient at the output
        layer, then chains through hidden layers with activation derivatives
        and dropout mask reversal.

        Returns (grad_W, grad_b) — lists matching self.weights / self.biases.
        """
        m  = y_true.shape[0]
        L  = self.n_layers
        cache = self._cache

        # ── Output layer: d(loss)/d(Z[L-1]) via softmax+cross-entropy ──────
        dZ = cache["A"][-1].copy()
        dZ[np.arange(m), y_true] -= 1.0
        dZ /= m

        grad_W: List[Optional[np.ndarray]] = [None] * L
        grad_b: List[Optional[np.ndarray]] = [None] * L

        for i in reversed(range(L)):
            A_prev = cache["A"][i]                # input to layer i
            W_i    = self.weights[i]

            # Parameter gradients for layer i
            grad_W[i] = A_prev.T @ dZ + (self.hp.l2_lambda / m) * W_i
            grad_b[i] = dZ.sum(axis=0, keepdims=True)

            if i > 0:
                # Propagate gradient to previous layer
                #   dA = d(loss)/d(cache['A'][i])
                #       = d(loss)/d(A_drop_{i-1}) since A[i] = activation(Z[i-1]) * mask_{i-1}
                dA = dZ @ W_i.T

                # Reverse dropout (multiply by same mask used in forward)
                if (i - 1) in cache["masks"]:
                    dA = dA * cache["masks"][i - 1]

                # Activation derivative for layer i-1
                dZ = dA * self._activation(cache["Z"][i - 1], deriv=True)

        return grad_W, grad_b   # type: ignore[return-value]

    # -----------------------------------------------------------------------
    # Parameter update
    # -----------------------------------------------------------------------

    def update(
        self,
        grad_W: List[np.ndarray],
        grad_b: List[np.ndarray],
        lr:     float,
    ) -> None:
        if self.hp.optimizer == "adam":
            self._adam_update(grad_W, grad_b, lr)
        else:
            self._sgd_update(grad_W, grad_b, lr)

    def _sgd_update(
        self, grad_W: List[np.ndarray], grad_b: List[np.ndarray], lr: float
    ) -> None:
        for i in range(self.n_layers):
            self.weights[i] -= lr * grad_W[i]
            self.biases[i]  -= lr * grad_b[i]

    def _adam_update(
        self,
        grad_W: List[np.ndarray],
        grad_b: List[np.ndarray],
        lr:     float,
        beta1:  float = 0.9,
        beta2:  float = 0.999,
        eps:    float = 1e-8,
    ) -> None:
        self._adam_t += 1
        t = self._adam_t
        for i in range(self.n_layers):
            # First moment
            self._m_W[i] = beta1 * self._m_W[i] + (1 - beta1) * grad_W[i]
            self._m_b[i] = beta1 * self._m_b[i] + (1 - beta1) * grad_b[i]
            # Second moment
            self._v_W[i] = beta2 * self._v_W[i] + (1 - beta2) * grad_W[i] ** 2
            self._v_b[i] = beta2 * self._v_b[i] + (1 - beta2) * grad_b[i] ** 2
            # Bias-corrected
            mW_hat = self._m_W[i] / (1 - beta1 ** t)
            mb_hat = self._m_b[i] / (1 - beta1 ** t)
            vW_hat = self._v_W[i] / (1 - beta2 ** t)
            vb_hat = self._v_b[i] / (1 - beta2 ** t)
            # Update
            self.weights[i] -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
            self.biases[i]  -= lr * mb_hat / (np.sqrt(vb_hat) + eps)

    # -----------------------------------------------------------------------
    # Inference helpers
    # -----------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.forward(X, training=False)
        return np.argmax(probs, axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X, training=False)

    # -----------------------------------------------------------------------
    # Snapshot / restore (for best-model tracking)
    # -----------------------------------------------------------------------

    def get_state(self) -> dict:
        return {
            "weights": [W.copy() for W in self.weights],
            "biases":  [b.copy() for b in self.biases],
        }

    def set_state(self, state: dict) -> None:
        self.weights = [W.copy() for W in state["weights"]]
        self.biases  = [b.copy() for b in state["biases"]]


# ===========================================================================
# Training loop
# ===========================================================================

def train(
    model:     NeuralNetwork,
    X_train:   np.ndarray,
    y_train:   np.ndarray,
    X_val:     np.ndarray,
    y_val:     np.ndarray,
    hp:        HyperParameters,
    verbose:   bool = True,
    log_every: int  = 10,
) -> dict:
    """
    Train *model* for hp.epochs epochs using mini-batch gradient descent.

    Returns a history dict:
        {
          'train_loss': [...], 'val_loss':  [...],
          'train_acc':  [...], 'val_acc':   [...],
          'epoch_times': [...],
          'best_val_acc': float, 'best_epoch': int,
        }
    """
    m  = X_train.shape[0]
    lr = hp.learning_rate
    rng = np.random.default_rng(42)

    history: Dict[str, list] = {
        "train_loss":  [],
        "val_loss":    [],
        "train_acc":   [],
        "val_acc":     [],
        "epoch_times": [],
    }

    best_val_acc   = -np.inf
    best_state     = model.get_state()
    best_epoch     = 0

    for epoch in range(hp.epochs):
        t_start = time.perf_counter()

        # ── Mini-batch loop ──────────────────────────────────────────────
        idx       = rng.permutation(m)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, m, hp.batch_size):
            batch_idx = idx[start: start + hp.batch_size]
            X_b = X_train[batch_idx]
            y_b = y_train[batch_idx]

            probs = model.forward(X_b, training=True)
            loss  = model.compute_loss(probs, y_b)
            gW, gb = model.backward(y_b)
            model.update(gW, gb, lr)

            epoch_loss += loss
            n_batches  += 1

        epoch_loss /= n_batches

        # ── Metrics ──────────────────────────────────────────────────────
        train_acc = model.accuracy(X_train, y_train)
        val_probs = model.forward(X_val, training=False)
        val_loss  = model.compute_loss(val_probs, y_val)
        val_acc   = model.accuracy(X_val, y_val)

        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["epoch_times"].append(time.perf_counter() - t_start)

        # ── Track best model ──────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = model.get_state()
            best_epoch   = epoch + 1

        # ── Logging ───────────────────────────────────────────────────────
        if verbose and ((epoch + 1) % log_every == 0 or epoch == 0):
            logger.info(
                "  Epoch %4d/%d | train_loss=%.4f  val_loss=%.4f | "
                "train_acc=%.4f  val_acc=%.4f",
                epoch + 1, hp.epochs,
                epoch_loss, val_loss,
                train_acc, val_acc,
            )

    history["best_val_acc"] = best_val_acc
    history["best_epoch"]   = best_epoch
    history["best_state"]   = best_state

    logger.info(
        "Training complete. Best val_acc=%.4f at epoch %d",
        best_val_acc, best_epoch
    )

    # Restore best weights
    model.set_state(best_state)

    return history


# ===========================================================================
# Metrics helpers
# ===========================================================================

def compute_metrics(
    model:       NeuralNetwork,
    X:           np.ndarray,
    y:           np.ndarray,
    class_names: List[str],
) -> dict:
    """Compute a full set of evaluation metrics for a dataset split."""
    preds = model.predict(X)
    probs = model.predict_proba(X)
    loss  = model.compute_loss(probs, y)
    acc   = float(np.mean(preds == y))
    n_cls = len(class_names)

    # Per-class metrics
    precision = np.zeros(n_cls)
    recall    = np.zeros(n_cls)
    f1        = np.zeros(n_cls)

    for c in range(n_cls):
        tp = int(np.sum((preds == c) & (y == c)))
        fp = int(np.sum((preds == c) & (y != c)))
        fn = int(np.sum((preds != c) & (y == c)))
        precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[c]    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1[c]        = (2 * precision[c] * recall[c] /
                        (precision[c] + recall[c])
                        if (precision[c] + recall[c]) > 0 else 0.0)

    # Confusion matrix
    conf = np.zeros((n_cls, n_cls), dtype=int)
    for true, pred in zip(y, preds):
        conf[true, pred] += 1

    return {
        "loss":           loss,
        "accuracy":       acc,
        "macro_precision": float(precision.mean()),
        "macro_recall":    float(recall.mean()),
        "macro_f1":        float(f1.mean()),
        "per_class_precision": precision.tolist(),
        "per_class_recall":    recall.tolist(),
        "per_class_f1":        f1.tolist(),
        "confusion_matrix":    conf.tolist(),
    }


def build_network(
    input_size:    int,
    num_classes:   int,
    hp:            HyperParameters,
    random_state:  int = 42,
) -> NeuralNetwork:
    """Convenience factory used by main.py and agent.py."""
    return NeuralNetwork(
        input_size    = input_size,
        hidden_layers = hp.hidden_layers,
        num_classes   = num_classes,
        hyperparams   = hp,
        random_state  = random_state,
    )
