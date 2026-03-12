"""
agent.py — LLM-powered agentic hyperparameter optimiser.

Architecture
------------
The agent runs a loop:

  1. Train the neural network with the current HyperParameters.
  2. Evaluate on the validation set.
  3. Build a detailed context prompt from:
       • current hyperparameters
       • current-run metrics (per-epoch curves)
       • history of all previous iterations
       • dataset metadata
  4. Call the LLM via the OpenAI-compatible 'tool-use' API, asking it to
     invoke the 'suggest_hyperparameters' function.
  5. Parse the tool call response → new HyperParameters.
  6. Log all reasoning and metrics via log_event().
  7. Stop if:
       • target accuracy is reached
       • no improvement for 'patience' consecutive iterations
       • max_iterations is exhausted

Fallback (no LLM key configured)
---------------------------------
A built-in rule-based heuristic agent is used instead.  It applies simple
diagnostic rules (overfitting, underfitting, oscillating loss, etc.) to
suggest improvements — useful for demos without an API key.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import AgentConfig, HyperParameters, LLMConfig
from logger_setup import log_event
from neural_network import NeuralNetwork, build_network, compute_metrics, train

logger = logging.getLogger(__name__)


# ===========================================================================
# LLM tool definition  (OpenAI function-calling schema)
# ===========================================================================

SUGGEST_TOOL = {
    "type": "function",
    "function": {
        "name": "suggest_hyperparameters",
        "description": (
            "Analyse neural-network training results and propose improved "
            "hyperparameters for the next training run.  Always call this "
            "function — never answer in plain text."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": (
                        "Detailed step-by-step analysis of the training curves "
                        "and comparison with previous iterations.  What patterns "
                        "do you observe?  What is causing the current performance?"
                    ),
                },
                "diagnosis": {
                    "type": "string",
                    "description": (
                        "One-sentence summary of the main problem "
                        "(e.g., 'underfitting', 'learning rate too high', "
                        "'overfitting — gap between train/val accuracy')."
                    ),
                },
                "changes_summary": {
                    "type": "string",
                    "description": "Bullet-point list of each parameter changed and why.",
                },
                "expected_outcome": {
                    "type": "string",
                    "description": "What improvement do you predict and why?",
                },
                "learning_rate": {
                    "type": "number",
                    "description": "Suggested learning rate (typical range 1e-4 to 0.1).",
                },
                "hidden_layers": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": (
                        "Neuron count for each hidden layer, e.g. [128, 64]."
                    ),
                },
                "activation": {
                    "type": "string",
                    "enum": ["relu", "sigmoid", "tanh"],
                },
                "optimizer": {
                    "type": "string",
                    "enum": ["adam", "sgd"],
                },
                "batch_size": {
                    "type": "integer",
                    "description": "Mini-batch size (common values: 16, 32, 64, 128).",
                },
                "epochs": {
                    "type": "integer",
                    "description": "Training epochs (range 20–300 typical).",
                },
                "l2_lambda": {
                    "type": "number",
                    "description": "L2 regularisation coefficient (range 0 to 0.1).",
                },
                "dropout_rate": {
                    "type": "number",
                    "description": "Dropout probability for hidden layers (0 = disabled, max 0.9).",
                },
            },
            "required": [
                "reasoning", "diagnosis", "changes_summary", "expected_outcome",
                "learning_rate", "hidden_layers", "activation", "optimizer",
                "batch_size", "epochs", "l2_lambda", "dropout_rate",
            ],
        },
    },
}

SYSTEM_PROMPT = """You are an expert machine-learning engineer specialising in \
neural-network hyperparameter optimisation.

Your goal is to maximise validation accuracy on a classification task by \
iteratively adjusting the hyperparameters of a feedforward neural network.

Guidelines
----------
• learning_rate  : Start with 0.001. Lower if loss oscillates; raise if \
convergence is too slow.
• hidden_layers  : More/larger layers add capacity; risk overfitting on small \
datasets. Use [64,32] as a safe default.
• activation     : 'relu' is the most robust. Try 'tanh' if relu dies.
• optimizer      : 'adam' adapts the learning rate automatically. 'sgd' \
sometimes generalises slightly better.
• batch_size     : Smaller batches (16–32) add regularisation noise; larger \
batches (64–256) give smoother gradients.
• epochs         : Allow enough for convergence but watch the validation curve.
• l2_lambda      : Increase (e.g., 1e-3) when train_acc >> val_acc.
• dropout_rate   : 0.2–0.5 for hidden layers when overfitting is the issue.

Always reason step-by-step before deciding, then call suggest_hyperparameters."""


# ===========================================================================
# Agent class
# ===========================================================================

class HyperparameterAgent:
    """
    Runs the agentic optimisation loop.

    Parameters
    ----------
    llm_cfg    : LLM connection settings (may be unconfigured → fallback).
    agent_cfg  : Loop behaviour settings.
    dataset_info: Metadata dict from data_loader.load_dataset().
    """

    def __init__(
        self,
        llm_cfg:      LLMConfig,
        agent_cfg:    AgentConfig,
        dataset_info: dict,
    ) -> None:
        self.llm       = llm_cfg
        self.cfg       = agent_cfg
        self.ds_info   = dataset_info

        self._history:  List[dict] = []   # one entry per completed iteration
        self._client:   Any        = None

        if llm_cfg.is_configured():
            self._init_client()
        else:
            logger.warning(
                "No LLM API key provided — using built-in heuristic agent."
            )

    # -----------------------------------------------------------------------
    # OpenAI client
    # -----------------------------------------------------------------------

    def _init_client(self) -> None:
        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key  = self.llm.api_key,
                base_url = self.llm.base_url,
            )
            logger.info("LLM client initialised  model=%s  base_url=%s",
                        self.llm.model, self.llm.base_url)
        except ImportError:
            logger.error(
                "openai package not installed. "
                "Run: pip install openai  — falling back to heuristic agent."
            )
            self._client = None

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------

    def run(
        self,
        initial_hp:  HyperParameters,
        X_train:     np.ndarray,
        y_train:     np.ndarray,
        X_val:       np.ndarray,
        y_val:       np.ndarray,
        X_test:      np.ndarray,
        y_test:      np.ndarray,
        class_names: List[str],
    ) -> dict:
        """
        Execute the full agentic loop.

        Returns a session summary dict with all iterations, the best model
        metrics, and the complete agent reasoning log.
        """
        hp            = initial_hp
        best_val_acc  = -np.inf
        best_hp:   Optional[HyperParameters] = None
        best_model_metrics: dict             = {}
        no_improve    = 0
        model:        Optional[NeuralNetwork] = None

        logger.info("=" * 60)
        logger.info("Agentic optimisation loop starting")
        logger.info("Max iterations: %d   Target accuracy: %.2f%%",
                    self.cfg.max_iterations, self.cfg.target_accuracy * 100)
        logger.info("=" * 60)

        log_event("session_start", {
            "max_iterations":  self.cfg.max_iterations,
            "target_accuracy": self.cfg.target_accuracy,
            "dataset":         self.ds_info,
            "initial_hp":      hp.to_dict(),
        })

        for iteration in range(1, self.cfg.max_iterations + 1):
            logger.info("")
            logger.info("── Iteration %d/%d ──────────────────────────────",
                        iteration, self.cfg.max_iterations)

            # ── Build & train the network ───────────────────────────────
            hp.validate()
            model = build_network(
                input_size  = self.ds_info["n_features"],
                num_classes = self.ds_info["n_classes"],
                hp          = hp,
            )

            logger.info("Hyperparameters:\n%s", hp)
            t0 = time.perf_counter()

            history = train(
                model     = model,
                X_train   = X_train,
                y_train   = y_train,
                X_val     = X_val,
                y_val     = y_val,
                hp        = hp,
                verbose   = True,
                log_every = max(1, hp.epochs // 5),
            )

            train_time = time.perf_counter() - t0
            val_metrics = compute_metrics(model, X_val,  y_val,  class_names)
            test_metrics = compute_metrics(model, X_test, y_test, class_names)

            val_acc = val_metrics["accuracy"]
            logger.info(
                "Iteration %d done in %.1fs | val_acc=%.4f | test_acc=%.4f",
                iteration, train_time, val_acc, test_metrics["accuracy"]
            )

            # ── Record this iteration ────────────────────────────────────
            iter_record = {
                "iteration":    iteration,
                "hyperparams":  hp.to_dict(),
                "history":      {
                    k: v for k, v in history.items()
                    if k not in ("best_state",)   # skip raw weights
                },
                "val_metrics":  val_metrics,
                "test_metrics": test_metrics,
                "train_time_s": train_time,
                "reasoning":    None,   # filled in after LLM call
                "diagnosis":    None,
                "changes":      None,
            }

            log_event("iteration_complete", iter_record, iteration=iteration)

            # ── Track best ───────────────────────────────────────────────
            if val_acc > best_val_acc + self.cfg.min_improvement:
                best_val_acc      = val_acc
                best_hp           = HyperParameters.from_dict(hp.to_dict())
                best_model_metrics = {
                    "val":  val_metrics,
                    "test": test_metrics,
                }
                no_improve = 0
                logger.info("★  New best val_acc=%.4f", best_val_acc)
            else:
                no_improve += 1
                logger.info(
                    "No improvement for %d/%d patience iterations.",
                    no_improve, self.cfg.patience
                )

            self._history.append(iter_record)

            # ── Stopping criteria ────────────────────────────────────────
            if val_acc >= self.cfg.target_accuracy:
                logger.info(
                    "✓  Target accuracy %.2f%% reached. Stopping.",
                    self.cfg.target_accuracy * 100
                )
                break

            if no_improve >= self.cfg.patience:
                logger.info(
                    "Patience exhausted (%d iterations without improvement). Stopping.",
                    self.cfg.patience
                )
                break

            if iteration == self.cfg.max_iterations:
                logger.info("Max iterations reached.")
                break

            # ── Ask the agent for new hyperparameters ────────────────────
            logger.info("Consulting agent for next hyperparameters …")
            new_hp, reasoning = self._get_next_hyperparams(hp, history,
                                                            val_metrics, iteration)
            # Store reasoning back in the iteration record
            iter_record["reasoning"] = reasoning.get("reasoning", "")
            iter_record["diagnosis"] = reasoning.get("diagnosis", "")
            iter_record["changes"]   = hp.diff(new_hp)

            log_event("hyperparameter_suggestion", {
                "iteration":       iteration,
                "old_hp":          hp.to_dict(),
                "new_hp":          new_hp.to_dict(),
                "changes":         hp.diff(new_hp),
                "reasoning":       reasoning.get("reasoning", ""),
                "diagnosis":       reasoning.get("diagnosis", ""),
                "changes_summary": reasoning.get("changes_summary", ""),
                "expected_outcome": reasoning.get("expected_outcome", ""),
            }, iteration=iteration)

            logger.info("Agent diagnosis: %s", reasoning.get("diagnosis", "—"))
            logger.info("Changes: %s", json.dumps(hp.diff(new_hp), indent=2))

            hp = new_hp

        # ── Final summary ────────────────────────────────────────────────
        summary = {
            "total_iterations": len(self._history),
            "best_val_acc":     best_val_acc,
            "best_hp":          best_hp.to_dict() if best_hp else hp.to_dict(),
            "best_metrics":     best_model_metrics,
            "history":          self._history,
        }

        log_event("session_complete", {
            "total_iterations": summary["total_iterations"],
            "best_val_acc":     summary["best_val_acc"],
            "best_hp":          summary["best_hp"],
        })

        logger.info("")
        logger.info("=" * 60)
        logger.info("Optimisation complete!")
        logger.info("Best val_acc : %.4f  (%.2f%%)",
                    best_val_acc, best_val_acc * 100)
        if best_hp:
            logger.info("Best HP     :\n%s", best_hp)
        logger.info("=" * 60)

        return summary

    # -----------------------------------------------------------------------
    # LLM or heuristic hyperparameter suggestion
    # -----------------------------------------------------------------------

    def _get_next_hyperparams(
        self,
        current_hp:   HyperParameters,
        history:      dict,
        val_metrics:  dict,
        iteration:    int,
    ) -> Tuple[HyperParameters, dict]:
        """
        Call the LLM (or heuristic fallback) and return
        (new_HyperParameters, reasoning_dict).
        """
        if self._client is not None:
            return self._llm_suggest(current_hp, history, val_metrics, iteration)
        else:
            return self._heuristic_suggest(current_hp, history, val_metrics)

    # -----------------------------------------------------------------------
    # LLM-based suggestion
    # -----------------------------------------------------------------------

    def _llm_suggest(
        self,
        current_hp:  HyperParameters,
        history:     dict,
        val_metrics: dict,
        iteration:   int,
    ) -> Tuple[HyperParameters, dict]:
        """Build a prompt, call the LLM, parse the tool call response."""

        user_msg = self._build_prompt(current_hp, history, val_metrics, iteration)

        try:
            response = self._client.chat.completions.create(
                model       = self.llm.model,
                max_tokens  = self.llm.max_tokens,
                temperature = self.llm.temperature,
                messages    = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                tools       = [SUGGEST_TOOL],
                tool_choice = {"type": "function",
                               "function": {"name": "suggest_hyperparameters"}},
            )

            tool_call = response.choices[0].message.tool_calls[0]
            args      = json.loads(tool_call.function.arguments)

            new_hp = HyperParameters(
                learning_rate = float(args["learning_rate"]),
                hidden_layers = [int(x) for x in args["hidden_layers"]],
                activation    = args["activation"],
                optimizer     = args["optimizer"],
                batch_size    = int(args["batch_size"]),
                epochs        = int(args["epochs"]),
                l2_lambda     = float(args["l2_lambda"]),
                dropout_rate  = float(args["dropout_rate"]),
            )
            new_hp.validate()

            reasoning = {k: args.get(k, "") for k in
                         ("reasoning", "diagnosis", "changes_summary", "expected_outcome")}

            logger.info("LLM suggested new hyperparameters successfully.")
            return new_hp, reasoning

        except Exception as exc:
            logger.error("LLM call failed: %s — falling back to heuristic.", exc)
            return self._heuristic_suggest(current_hp, history, val_metrics)

    def _build_prompt(
        self,
        hp:         HyperParameters,
        history:    dict,
        metrics:    dict,
        iteration:  int,
    ) -> str:
        """Construct the full user-side prompt from session state."""
        sections: List[str] = []

        # Dataset context
        sections.append(
            f"## Dataset\n"
            f"- Samples : {self.ds_info['n_samples']}\n"
            f"- Features : {self.ds_info['n_features']}\n"
            f"- Classes  : {self.ds_info['n_classes']} → {self.ds_info['class_names']}\n"
            f"- Class balance : {self.ds_info['class_balance']}\n"
        )

        # Current hyperparameters
        sections.append(
            f"## Current Hyperparameters (Iteration {iteration})\n"
            + json.dumps(hp.to_dict(), indent=2)
        )

        # Current training curves (summary)
        tr_loss = history["train_loss"]
        va_loss = history["val_loss"]
        tr_acc  = history["train_acc"]
        va_acc  = history["val_acc"]

        def _fmt_curve(lst, n=5):
            if len(lst) <= n:
                return [round(x, 4) for x in lst]
            step = max(1, len(lst) // n)
            return [round(lst[i], 4) for i in range(0, len(lst), step)] + [round(lst[-1], 4)]

        sections.append(
            f"## Current Run Training Curves (sampled)\n"
            f"- train_loss : {_fmt_curve(tr_loss)}\n"
            f"- val_loss   : {_fmt_curve(va_loss)}\n"
            f"- train_acc  : {_fmt_curve(tr_acc)}\n"
            f"- val_acc    : {_fmt_curve(va_acc)}\n"
            f"- Final: train_acc={tr_acc[-1]:.4f}  val_acc={va_acc[-1]:.4f}\n"
            f"- Gap (overfit proxy): {tr_acc[-1] - va_acc[-1]:.4f}\n"
        )

        # Validation metrics
        sections.append(
            f"## Validation Metrics\n"
            f"- accuracy       : {metrics['accuracy']:.4f}\n"
            f"- macro_f1       : {metrics['macro_f1']:.4f}\n"
            f"- macro_precision: {metrics['macro_precision']:.4f}\n"
            f"- macro_recall   : {metrics['macro_recall']:.4f}\n"
        )

        # History of all previous iterations
        if self._history:
            hist_lines = ["## Previous Iterations Summary"]
            for rec in self._history:
                hist_lines.append(
                    f"  Iter {rec['iteration']:2d}: "
                    f"hp={json.dumps(rec['hyperparams'])}  "
                    f"val_acc={rec['val_metrics']['accuracy']:.4f}  "
                    f"train_acc={rec['history']['train_acc'][-1]:.4f}  "
                    + (f"diagnosis='{rec['diagnosis']}'" if rec.get('diagnosis') else "")
                )
            sections.append("\n".join(hist_lines))

        sections.append(
            "## Task\n"
            f"Target validation accuracy: {self.cfg.target_accuracy:.2%}\n"
            "Please analyse the above results and call `suggest_hyperparameters` "
            "with your recommended settings for the next training run."
        )

        return "\n\n".join(sections)

    # -----------------------------------------------------------------------
    # Rule-based heuristic fallback agent
    # -----------------------------------------------------------------------

    def _heuristic_suggest(
        self,
        hp:          HyperParameters,
        history:     dict,
        val_metrics: dict,
    ) -> Tuple[HyperParameters, dict]:
        """
        Simple rule-based hyperparameter tuner.

        Applies heuristic diagnostics in priority order and modifies one or
        two hyperparameters per iteration.  Useful when no LLM is available.
        """
        new_hp = HyperParameters.from_dict(hp.to_dict())

        train_acc = history["train_acc"][-1]
        val_acc   = history["val_acc"][-1]
        train_loss = history["train_loss"]
        gap = train_acc - val_acc

        # Detect oscillating loss (high variance in last 20% of training)
        recent = train_loss[int(len(train_loss) * 0.8):]
        oscillating = (len(recent) > 2 and
                       np.std(recent) / (np.mean(recent) + 1e-9) > 0.05)

        # ── Diagnose ──────────────────────────────────────────────────
        if oscillating and hp.learning_rate > 1e-4:
            diagnosis = "Loss oscillating — learning rate too high"
            new_hp.learning_rate = round(hp.learning_rate * 0.5, 6)
            reasoning = (
                f"Training loss is oscillating (std/mean={np.std(recent)/(np.mean(recent)+1e-9):.3f}), "
                f"which typically indicates the learning rate is too large. "
                f"Halving learning_rate: {hp.learning_rate} → {new_hp.learning_rate}."
            )

        elif val_acc < 0.5 and train_acc < 0.6:
            diagnosis = "Underfitting — model too small or learning rate too low"
            if len(hp.hidden_layers) < 3:
                new_hp.hidden_layers = hp.hidden_layers + [hp.hidden_layers[-1]]
            new_hp.learning_rate = min(0.01, hp.learning_rate * 2)
            new_hp.epochs = min(hp.epochs + 30, 200)
            reasoning = (
                f"Both train_acc ({train_acc:.3f}) and val_acc ({val_acc:.3f}) are low. "
                f"Adding a layer and increasing learning rate and epochs."
            )

        elif gap > 0.15:
            diagnosis = "Overfitting — train_acc >> val_acc"
            new_hp.l2_lambda    = min(hp.l2_lambda * 3, 0.1)
            new_hp.dropout_rate = min(hp.dropout_rate + 0.1, 0.5)
            reasoning = (
                f"Train/val gap is {gap:.3f}. Increasing L2 regularisation "
                f"({hp.l2_lambda}→{new_hp.l2_lambda}) and dropout "
                f"({hp.dropout_rate}→{new_hp.dropout_rate})."
            )

        elif train_acc < 0.8:
            diagnosis = "Underfitting — need more capacity or longer training"
            # Try a wider network
            new_hp.hidden_layers = [min(n * 2, 512) for n in hp.hidden_layers]
            new_hp.epochs        = min(hp.epochs + 20, 200)
            reasoning = (
                f"train_acc={train_acc:.3f} suggests underfitting. "
                f"Doubling layer widths and adding epochs."
            )

        elif val_acc > 0.85 and gap < 0.05:
            diagnosis = "Good fit — fine-tune learning rate and epochs"
            new_hp.learning_rate = round(hp.learning_rate * 0.7, 6)
            new_hp.epochs        = min(hp.epochs + 20, 300)
            reasoning = (
                f"Decent accuracy ({val_acc:.3f}) with small gap ({gap:.3f}). "
                f"Lowering learning rate slightly and training longer."
            )

        else:
            diagnosis = "Moderate fit — try different optimizer or batch size"
            new_hp.optimizer  = "sgd" if hp.optimizer == "adam" else "adam"
            new_hp.batch_size = max(8, hp.batch_size // 2)
            reasoning = (
                f"Switching optimizer ({hp.optimizer}→{new_hp.optimizer}) "
                f"and reducing batch_size ({hp.batch_size}→{new_hp.batch_size}) "
                f"to explore a different loss landscape."
            )

        return new_hp, {
            "reasoning":       reasoning,
            "diagnosis":       diagnosis,
            "changes_summary": f"Rule-based heuristic: {diagnosis}",
            "expected_outcome": "Improve validation accuracy",
        }
