"""
config.py — Configuration dataclasses for the Agentic Neural Network system.

All tunable hyperparameters, LLM settings, agent behaviour settings, and
data-loading settings live here so they can be serialised to / from JSON
and passed cleanly between modules.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional


# ---------------------------------------------------------------------------
# Hyperparameter space
# ---------------------------------------------------------------------------

VALID_ACTIVATIONS = ("relu", "sigmoid", "tanh")
VALID_OPTIMIZERS  = ("adam", "sgd")

SKLEARN_DATASETS = {
    "iris":          "load_iris",
    "digits":        "load_digits",
    "breast_cancer": "load_breast_cancer",
    "wine":          "load_wine",
}


@dataclass
class HyperParameters:
    """All tunable hyperparameters for a single training run."""

    learning_rate: float       = 0.001
    hidden_layers: List[int]   = field(default_factory=lambda: [64, 32])
    activation:    str         = "relu"
    optimizer:     str         = "adam"
    batch_size:    int         = 32
    epochs:        int         = 50
    l2_lambda:     float       = 1e-4
    dropout_rate:  float       = 0.0

    # ---- validation -------------------------------------------------------

    def validate(self) -> None:
        assert self.learning_rate > 0,             "learning_rate must be > 0"
        assert all(n > 0 for n in self.hidden_layers), "all layer sizes must be > 0"
        assert self.activation in VALID_ACTIVATIONS, \
            f"activation must be one of {VALID_ACTIVATIONS}"
        assert self.optimizer in VALID_OPTIMIZERS, \
            f"optimizer must be one of {VALID_OPTIMIZERS}"
        assert self.batch_size >= 1,   "batch_size must be >= 1"
        assert self.epochs >= 1,       "epochs must be >= 1"
        assert self.l2_lambda >= 0,    "l2_lambda must be >= 0"
        assert 0.0 <= self.dropout_rate < 1.0, "dropout_rate must be in [0, 1)"

    # ---- serialisation helpers --------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "HyperParameters":
        # Tolerate extra keys (e.g., from LLM that returns more fields)
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered   = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def diff(self, other: "HyperParameters") -> dict:
        """Return a dict of fields that changed from self → other."""
        changes = {}
        for key, val in self.to_dict().items():
            other_val = getattr(other, key)
            if val != other_val:
                changes[key] = {"from": val, "to": other_val}
        return changes


# ---------------------------------------------------------------------------
# LLM configuration
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    """Connection settings for any OpenAI-compatible LLM endpoint."""

    api_key:    str   = ""
    base_url:   str   = "https://api.openai.com/v1"
    model:      str   = "gpt-4o-mini"
    max_tokens: int   = 2048
    temperature: float = 0.3

    def is_configured(self) -> bool:
        return bool(self.api_key.strip())

    def to_dict(self) -> dict:
        d = asdict(self)
        d["api_key"] = "***" if self.api_key else ""   # Never log the key
        return d


# ---------------------------------------------------------------------------
# Agentic loop configuration
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    """Controls how the agentic optimisation loop behaves."""

    max_iterations:   int   = 10
    patience:         int   = 3          # stop if no improvement for N iters
    target_accuracy:  float = 0.97       # stop if val accuracy >= this
    min_improvement:  float = 0.002      # minimum Δacc to count as improvement
    use_heuristic_fallback: bool = True  # rule-based agent when no LLM key
    log_dir:          str   = "logs"
    results_dir:      str   = "results"

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Where to find data and how to split / normalise it."""

    source:          str   = "sklearn"   # "sklearn" or "local"
    sklearn_dataset: str   = "iris"      # key in SKLEARN_DATASETS
    local_path:      str   = ""          # path to CSV or .npz file
    label_column:    str   = ""          # CSV column name; "" = last column
    test_size:       float = 0.20
    val_size:        float = 0.10
    normalize:       bool  = True
    random_state:    int   = 42

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Full session configuration
# ---------------------------------------------------------------------------

@dataclass
class SessionConfig:
    """Top-level bundle passed throughout the whole pipeline."""

    hyperparams:  HyperParameters = field(default_factory=HyperParameters)
    llm:          LLMConfig       = field(default_factory=LLMConfig)
    agent:        AgentConfig     = field(default_factory=AgentConfig)
    data:         DataConfig      = field(default_factory=DataConfig)

    def to_dict(self) -> dict:
        return {
            "hyperparams": self.hyperparams.to_dict(),
            "llm":         self.llm.to_dict(),
            "agent":       self.agent.to_dict(),
            "data":        self.data.to_dict(),
        }

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "SessionConfig":
        with open(path) as fh:
            d = json.load(fh)
        cfg = cls()
        cfg.hyperparams = HyperParameters.from_dict(d.get("hyperparams", {}))
        # LLM: preserve api_key from environment if the saved one is masked
        llm_d = d.get("llm", {})
        if llm_d.get("api_key") == "***":
            llm_d.pop("api_key")
        cfg.llm = LLMConfig(**{k: v for k, v in llm_d.items()
                               if k in LLMConfig.__dataclass_fields__})
        cfg.agent = AgentConfig(**{k: v for k, v in d.get("agent", {}).items()
                                   if k in AgentConfig.__dataclass_fields__})
        cfg.data  = DataConfig(**{k: v for k, v in d.get("data", {}).items()
                                  if k in DataConfig.__dataclass_fields__})
        return cfg
