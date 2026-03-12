"""
main.py — Entry point for the Agentic Neural Network Optimiser.

Usage
-----
  python main.py                      # interactive wizard
  python main.py --config cfg.json    # load saved session config
  python main.py --demo               # quick demo (iris, heuristic agent)
  python main.py --help               # show all flags

The wizard guides users through:
  1. LLM API configuration
  2. Dataset selection (sklearn or local file)
  3. Initial hyperparameter choices
  4. Agent loop settings
  5. Running the optimisation
  6. Generating and viewing visualisations
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Optional

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
from config import (
    AgentConfig, DataConfig, HyperParameters, LLMConfig, SessionConfig,
    SKLEARN_DATASETS, VALID_ACTIVATIONS, VALID_OPTIMIZERS,
)
from data_loader import load_dataset, list_sklearn_datasets
from neural_network import build_network, compute_metrics
from agent import HyperparameterAgent
from logger_setup import setup_logging, log_event, get_jsonl_path
from visualizer import generate_all_plots

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ANSI helpers (degrade gracefully on Windows without colorama)
# ---------------------------------------------------------------------------

def _c(text: str, code: str) -> str:
    if os.name == "nt" and "ANSICON" not in os.environ:
        return text
    return f"\033[{code}m{text}\033[0m"

BOLD   = lambda t: _c(t, "1")
CYAN   = lambda t: _c(t, "36")
GREEN  = lambda t: _c(t, "32")
YELLOW = lambda t: _c(t, "33")
RED    = lambda t: _c(t, "31")
DIM    = lambda t: _c(t, "2")


# ===========================================================================
# Interactive wizard helpers
# ===========================================================================

def _ask(prompt: str, default: str = "", validator=None) -> str:
    """Prompt the user for a string value with an optional default."""
    if default:
        full_prompt = f"  {prompt} [{DIM(str(default))}]: "
    else:
        full_prompt = f"  {prompt}: "

    while True:
        try:
            raw = input(full_prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)

        value = raw if raw else str(default)

        if validator:
            try:
                value = validator(value)
                return value
            except (ValueError, AssertionError) as e:
                print(f"  {RED('Invalid')}: {e}  — please try again.")
        else:
            return value


def _ask_int(prompt: str, default: int, lo: int = 1, hi: int = 10_000) -> int:
    def v(s):
        n = int(s)
        assert lo <= n <= hi, f"must be between {lo} and {hi}"
        return n
    return _ask(prompt, str(default), validator=v)


def _ask_float(prompt: str, default: float, lo: float = 0.0,
               hi: float = 1e6) -> float:
    def v(s):
        f = float(s)
        assert lo <= f <= hi, f"must be between {lo} and {hi}"
        return f
    return _ask(prompt, str(default), validator=v)


def _ask_choice(prompt: str, choices: tuple, default: str) -> str:
    choices_str = " | ".join(
        BOLD(c) if c == default else c for c in choices
    )
    def v(s):
        assert s in choices, f"choose one of {choices}"
        return s
    return _ask(f"{prompt}  ({choices_str})", default, validator=v)


def _header(title: str) -> None:
    width = 60
    print()
    print(CYAN("─" * width))
    print(CYAN(f"  {title}"))
    print(CYAN("─" * width))


def _section(title: str) -> None:
    print()
    print(BOLD(f"▶  {title}"))


# ===========================================================================
# Wizard sections
# ===========================================================================

def wizard_llm(cfg: LLMConfig) -> LLMConfig:
    _section("LLM API Configuration")
    print(DIM("  Configure your OpenAI-compatible LLM.  Leave API key blank"))
    print(DIM("  to use the built-in heuristic agent (no API key needed)."))
    print()

    api_key  = _ask("API key (or press Enter to skip)",  default="")
    base_url = _ask("API base URL", default=cfg.base_url)
    model    = _ask("Model name",   default=cfg.model)

    return LLMConfig(
        api_key  = api_key,
        base_url = base_url,
        model    = model,
    )


def wizard_data(cfg: DataConfig) -> DataConfig:
    _section("Dataset Configuration")

    source = _ask_choice("Data source", ("sklearn", "local"), default=cfg.source)

    if source == "sklearn":
        print(f"\n  Available sklearn datasets: {', '.join(list_sklearn_datasets())}")
        dataset = _ask_choice(
            "Dataset name", tuple(SKLEARN_DATASETS.keys()), default=cfg.sklearn_dataset
        )
        return DataConfig(source="sklearn", sklearn_dataset=dataset,
                          test_size=cfg.test_size, val_size=cfg.val_size,
                          normalize=cfg.normalize, random_state=cfg.random_state)
    else:
        local_path = _ask("Path to CSV or .npz file", default=cfg.local_path)
        label_col  = _ask("Label column name (blank = last column)", default="")
        return DataConfig(source="local", local_path=local_path,
                          label_column=label_col,
                          test_size=cfg.test_size, val_size=cfg.val_size,
                          normalize=cfg.normalize, random_state=cfg.random_state)


def wizard_hyperparams(hp: HyperParameters) -> HyperParameters:
    _section("Initial Hyperparameters")
    print(DIM("  Press Enter to accept the default (shown in brackets)."))
    print()

    # Hidden layers
    default_layers = ",".join(str(n) for n in hp.hidden_layers)
    def parse_layers(s: str):
        parts = [int(x.strip()) for x in s.split(",")]
        assert all(p > 0 for p in parts), "all layer sizes must be positive"
        return parts

    hidden_layers_str = _ask(
        "Hidden layer sizes (comma-separated, e.g. 64,32)",
        default=default_layers,
        validator=lambda s: ",".join(str(p) for p in parse_layers(s)),
    )
    hidden_layers = parse_layers(hidden_layers_str)

    lr         = _ask_float("Learning rate",  hp.learning_rate, lo=1e-7, hi=10.0)
    activation = _ask_choice("Activation",    VALID_ACTIVATIONS, default=hp.activation)
    optimizer  = _ask_choice("Optimizer",     VALID_OPTIMIZERS,  default=hp.optimizer)
    batch_size = _ask_int("Batch size",       hp.batch_size,     lo=1, hi=4096)
    epochs     = _ask_int("Epochs",           hp.epochs,         lo=1, hi=5000)
    l2_lambda  = _ask_float("L2 lambda",      hp.l2_lambda,      lo=0.0, hi=1.0)
    dropout    = _ask_float("Dropout rate",   hp.dropout_rate,   lo=0.0, hi=0.9)

    return HyperParameters(
        learning_rate = lr,
        hidden_layers = hidden_layers,
        activation    = activation,
        optimizer     = optimizer,
        batch_size    = batch_size,
        epochs        = epochs,
        l2_lambda     = l2_lambda,
        dropout_rate  = dropout,
    )


def wizard_agent(cfg: AgentConfig) -> AgentConfig:
    _section("Agent Loop Configuration")
    print()

    max_iters    = _ask_int("Max iterations",     cfg.max_iterations,  lo=1, hi=100)
    patience     = _ask_int("Patience (no-improve stop)", cfg.patience, lo=1, hi=50)
    target_acc   = _ask_float("Target val accuracy",   cfg.target_accuracy, lo=0.01, hi=1.0)

    return AgentConfig(
        max_iterations  = max_iters,
        patience        = patience,
        target_accuracy = target_acc,
        min_improvement = cfg.min_improvement,
        log_dir         = cfg.log_dir,
        results_dir     = cfg.results_dir,
    )


# ===========================================================================
# Main flow
# ===========================================================================

def run_session(session: SessionConfig) -> dict:
    """
    Execute a full optimisation session:
      load data → agent loop → visualise → save results.
    """
    # ── Logging ──────────────────────────────────────────────────────────
    session_name = setup_logging(
        session.agent.log_dir,
        console_level=logging.INFO,
        file_level=logging.DEBUG,
    )

    logger.info("Session %s started.", session_name)
    session.save(os.path.join(session.agent.log_dir, f"{session_name}_config.json"))

    # ── Load data ─────────────────────────────────────────────────────────
    logger.info("Loading dataset  source=%s …", session.data.source)
    (X_train, y_train,
     X_val,   y_val,
     X_test,  y_test,
     class_names, ds_info) = load_dataset(session.data)

    logger.info(
        "Data ready — %d train / %d val / %d test   features=%d   classes=%d",
        X_train.shape[0], X_val.shape[0], X_test.shape[0],
        ds_info["n_features"], ds_info["n_classes"],
    )
    log_event("data_loaded", ds_info)

    # ── Agent loop ────────────────────────────────────────────────────────
    agent = HyperparameterAgent(
        llm_cfg      = session.llm,
        agent_cfg    = session.agent,
        dataset_info = ds_info,
    )

    summary = agent.run(
        initial_hp   = session.hyperparams,
        X_train      = X_train,
        y_train      = y_train,
        X_val        = X_val,
        y_val        = y_val,
        X_test       = X_test,
        y_test       = y_test,
        class_names  = class_names,
    )

    summary["class_names"]   = class_names
    summary["session_name"]  = session_name
    summary["dataset_info"]  = ds_info

    # ── Save JSON summary ─────────────────────────────────────────────────
    os.makedirs(session.agent.results_dir, exist_ok=True)
    summary_path = os.path.join(
        session.agent.results_dir, f"{session_name}_summary.json"
    )
    _save_summary(summary, summary_path)
    logger.info("Summary saved to %s", summary_path)

    # ── Visualisations ────────────────────────────────────────────────────
    vis_dir   = os.path.join(session.agent.results_dir, session_name)
    vis_paths = generate_all_plots(
        summary     = summary,
        class_names = class_names,
        save_dir    = vis_dir,
        show        = False,
    )

    logger.info("Visualisations written to: %s", vis_dir)
    for p in vis_paths:
        logger.info("  %s", os.path.basename(p))

    return summary


def _save_summary(summary: dict, path: str) -> None:
    """Save the session summary to JSON, skipping non-serialisable objects."""
    def default(obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return str(obj)

    # Remove raw model weights (large, not needed in JSON)
    clean = json.loads(json.dumps(summary, default=default))
    with open(path, "w") as fh:
        json.dump(clean, fh, indent=2)


# ===========================================================================
# CLI
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog        = "main.py",
        description = "Agentic Neural Network Hyperparameter Optimiser",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = textwrap.dedent("""\
            Examples:
              python main.py                       # interactive wizard
              python main.py --demo                # quick iris demo
              python main.py --config cfg.json     # load saved config
              python main.py --dataset digits \\
                             --api-key sk-xxx \\
                             --model gpt-4o-mini \\
                             --max-iter 8
        """),
    )
    p.add_argument("--config",     metavar="FILE",
                   help="JSON config file produced by a previous session.")
    p.add_argument("--demo",       action="store_true",
                   help="Run a quick demo (iris dataset, heuristic agent, 3 iterations).")
    p.add_argument("--dataset",    metavar="NAME",
                   choices=list(SKLEARN_DATASETS.keys()),
                   help="sklearn dataset name (overrides config).")
    p.add_argument("--local-data", metavar="PATH",
                   help="Path to a local CSV or .npz dataset.")
    p.add_argument("--api-key",    metavar="KEY",
                   help="OpenAI-compatible API key.")
    p.add_argument("--base-url",   metavar="URL",
                   help="API base URL (default: https://api.openai.com/v1).")
    p.add_argument("--model",      metavar="NAME",
                   help="LLM model name (default: gpt-4o-mini).")
    p.add_argument("--max-iter",   type=int, metavar="N",
                   help="Maximum agent iterations (default: 10).")
    p.add_argument("--target-acc", type=float, metavar="F",
                   help="Target validation accuracy 0–1 (default: 0.97).")
    p.add_argument("--no-wizard",  action="store_true",
                   help="Skip interactive wizard, use defaults + CLI flags only.")
    p.add_argument("--log-dir",    metavar="DIR", default="logs",
                   help="Log directory (default: logs/).")
    p.add_argument("--results-dir", metavar="DIR", default="results",
                   help="Results directory (default: results/).")
    return p


import textwrap


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    # ── Banner ────────────────────────────────────────────────────────────
    print()
    print(CYAN("╔══════════════════════════════════════════════════════════╗"))
    print(CYAN("║       Agentic Neural Network Hyperparameter Optimiser    ║"))
    print(CYAN("║         LLM-guided automatic model improvement           ║"))
    print(CYAN("╚══════════════════════════════════════════════════════════╝"))
    print()

    # ── Demo mode ─────────────────────────────────────────────────────────
    if args.demo:
        print(GREEN("Demo mode: iris dataset, heuristic agent, 3 iterations."))
        session = SessionConfig(
            hyperparams = HyperParameters(
                learning_rate = 0.01,
                hidden_layers = [32],
                epochs        = 30,
                batch_size    = 16,
            ),
            llm   = LLMConfig(),   # no key → heuristic
            agent = AgentConfig(
                max_iterations  = 3,
                patience        = 5,
                target_accuracy = 0.99,
                log_dir         = args.log_dir,
                results_dir     = args.results_dir,
            ),
            data  = DataConfig(source="sklearn", sklearn_dataset="iris"),
        )
        summary = run_session(session)
        _print_final_summary(summary)
        return

    # ── Load config from file (optional) ──────────────────────────────────
    if args.config:
        print(f"Loading config from {args.config} …")
        session = SessionConfig.load(args.config)
    else:
        session = SessionConfig()

    # ── Apply CLI overrides ───────────────────────────────────────────────
    if args.dataset:
        session.data = DataConfig(source="sklearn", sklearn_dataset=args.dataset)
    if args.local_data:
        session.data = DataConfig(source="local", local_path=args.local_data)
    if args.api_key:
        session.llm.api_key = args.api_key
    if args.base_url:
        session.llm.base_url = args.base_url
    if args.model:
        session.llm.model = args.model
    if args.max_iter:
        session.agent.max_iterations = args.max_iter
    if args.target_acc:
        session.agent.target_accuracy = args.target_acc
    if args.log_dir:
        session.agent.log_dir    = args.log_dir
    if args.results_dir:
        session.agent.results_dir = args.results_dir

    # ── Interactive wizard (unless --no-wizard) ───────────────────────────
    if not args.no_wizard:
        _header("Welcome!  Let's configure your optimisation session.")
        print(DIM("  Answer each question or press Enter to accept the default."))
        print(DIM("  Ctrl+C at any time to abort."))

        session.llm         = wizard_llm(session.llm)
        session.data        = wizard_data(session.data)
        session.hyperparams = wizard_hyperparams(session.hyperparams)
        session.agent       = wizard_agent(session.agent)
        session.agent.log_dir     = args.log_dir
        session.agent.results_dir = args.results_dir

        _section("Configuration Summary")
        print(json.dumps(session.to_dict(), indent=2))
        print()
        confirm = _ask("Proceed?  (yes/no)", default="yes")
        if confirm.lower() not in ("yes", "y"):
            print("Aborted.")
            return

    # ── Run ───────────────────────────────────────────────────────────────
    summary = run_session(session)
    _print_final_summary(summary)


def _print_final_summary(summary: dict) -> None:
    print()
    print(CYAN("═" * 60))
    print(BOLD(GREEN("  Optimisation complete!")))
    print(CYAN("═" * 60))

    best_metrics = summary.get("best_metrics", {})
    val  = best_metrics.get("val",  {})
    test = best_metrics.get("test", {})

    val_acc_str  = f"{val.get('accuracy', 0):.4f}"
    test_acc_str = f"{test.get('accuracy', 0):.4f}"
    print(f"  Total iterations : {summary.get('total_iterations', '?')}")
    print(f"  Best val  acc    : {GREEN(val_acc_str)}")
    print(f"  Test accuracy    : {GREEN(test_acc_str)}")
    print(f"  Test macro-F1    : {test.get('macro_f1', 0):.4f}")
    print()

    best_hp = summary.get("best_hp", {})
    print(BOLD("  Best hyperparameters:"))
    for k, v in best_hp.items():
        print(f"    {k:<18} = {v}")

    print()
    res_dir = os.path.join(
        "results",
        summary.get("session_name", "session"),
    )
    print(f"  Visualisations saved to: {CYAN(res_dir)}")
    print(f"  Logs saved to          : {CYAN('logs/')}")
    print()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
