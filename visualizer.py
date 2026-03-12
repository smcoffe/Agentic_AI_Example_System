"""
visualizer.py — Matplotlib visualisation suite.

Three types of plots are produced:

  1. per_iteration_curves   — Training & validation loss/accuracy for every run.
  2. hyperparameter_evolution — How each hyperparameter changed over iterations.
  3. agent_reasoning_timeline — Visual log of the agent's diagnoses & decisions.
  4. final_dashboard          — A combined multi-panel report.
  5. confusion_matrix         — Colour-coded confusion matrix for the best model.

All functions accept the session summary dict returned by agent.HyperparameterAgent.run()
and optionally a `save_dir` to write PNG files.
"""

from __future__ import annotations

import json
import logging
import os
import textwrap
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy matplotlib import — keeps the module importable on headless systems
# ---------------------------------------------------------------------------

def _get_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")        # non-interactive backend (safe everywhere)
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        return plt, gridspec
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualisation. "
            "Install with: pip install matplotlib"
        )


# ===========================================================================
# 1. Per-iteration training curves
# ===========================================================================

def plot_training_curves(
    summary:  dict,
    save_dir: Optional[str] = None,
    show:     bool          = False,
) -> str:
    """
    Plot loss and accuracy curves for every training iteration.

    Each iteration is drawn as a separate subplot row (loss on the left,
    accuracy on the right).  A colour gradient darkens with each iteration
    to make progress visible.
    """
    plt, gridspec = _get_plt()

    history_list = summary.get("history", [])
    n_iters      = len(history_list)
    if n_iters == 0:
        logger.warning("No training history to plot.")
        return ""

    fig, axes = plt.subplots(n_iters, 2, figsize=(14, 3.5 * n_iters))
    if n_iters == 1:
        axes = np.array([axes])   # ensure 2-D for consistent indexing

    fig.suptitle("Training Curves — All Iterations", fontsize=14, fontweight="bold", y=1.01)

    cmap   = plt.cm.Blues
    shades = [cmap(0.4 + 0.6 * i / max(n_iters - 1, 1)) for i in range(n_iters)]

    for row, record in enumerate(history_list):
        h       = record["history"]
        epochs  = list(range(1, len(h["train_loss"]) + 1))
        colour  = shades[row]
        hp      = record["hyperparams"]
        val_acc = record["val_metrics"]["accuracy"]

        title_suffix = (
            f"  lr={hp['learning_rate']}  layers={hp['hidden_layers']}  "
            f"act={hp['activation']}  val_acc={val_acc:.4f}"
        )

        # ── Loss ──────────────────────────────────────────────────────────
        ax_l = axes[row, 0]
        ax_l.plot(epochs, h["train_loss"], color=colour,     label="train_loss", linewidth=1.8)
        ax_l.plot(epochs, h["val_loss"],   color=colour, ls="--", label="val_loss",  linewidth=1.8, alpha=0.8)
        ax_l.set_title(f"Iter {record['iteration']} — Loss" + title_suffix, fontsize=9)
        ax_l.set_xlabel("Epoch")
        ax_l.set_ylabel("Loss")
        ax_l.legend(fontsize=8)
        ax_l.grid(True, alpha=0.3)

        # ── Accuracy ──────────────────────────────────────────────────────
        ax_a = axes[row, 1]
        ax_a.plot(epochs, h["train_acc"], color=colour,     label="train_acc", linewidth=1.8)
        ax_a.plot(epochs, h["val_acc"],   color=colour, ls="--", label="val_acc",  linewidth=1.8, alpha=0.8)
        ax_a.set_title(f"Iter {record['iteration']} — Accuracy", fontsize=9)
        ax_a.set_xlabel("Epoch")
        ax_a.set_ylabel("Accuracy")
        ax_a.set_ylim([0, 1.05])
        ax_a.legend(fontsize=8)
        ax_a.grid(True, alpha=0.3)

    plt.tight_layout()
    path = _save(fig, "training_curves.png", save_dir, show)
    plt.close(fig)
    return path


# ===========================================================================
# 2. Hyperparameter evolution
# ===========================================================================

def plot_hyperparameter_evolution(
    summary:  dict,
    save_dir: Optional[str] = None,
    show:     bool          = False,
) -> str:
    """
    Show how each numerical hyperparameter changed across iterations.

    Categorical hyperparameters (activation, optimizer) are shown as
    text annotations above the x-axis.
    """
    plt, gridspec = _get_plt()

    history_list = summary.get("history", [])
    if not history_list:
        return ""

    iters   = [r["iteration"]   for r in history_list]
    val_acc = [r["val_metrics"]["accuracy"] for r in history_list]

    # Numerical params to plot
    num_params = ["learning_rate", "l2_lambda", "dropout_rate", "batch_size", "epochs"]
    # Categorical params to annotate
    cat_params = ["activation", "optimizer"]

    n_num = len(num_params)
    fig, axes = plt.subplots(n_num + 2, 1, figsize=(12, 2.8 * (n_num + 2)))
    fig.suptitle("Hyperparameter Evolution", fontsize=14, fontweight="bold")

    colors = plt.cm.tab10.colors

    # ── Numerical params ──────────────────────────────────────────────────
    for idx, param in enumerate(num_params):
        ax     = axes[idx]
        values = [r["hyperparams"].get(param, None) for r in history_list]
        ax.plot(iters, values, marker="o", color=colors[idx], linewidth=2, markersize=6)
        ax.set_ylabel(param, fontsize=9)
        ax.set_xticks(iters)
        ax.grid(True, alpha=0.3)

        # Annotate each point
        for xi, yi in zip(iters, values):
            if yi is not None:
                ax.annotate(
                    f"{yi:.5g}", (xi, yi),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7,
                )

    # ── Hidden layer sizes (special: list per iteration) ─────────────────
    ax_layers = axes[n_num]
    for rec in history_list:
        hl  = rec["hyperparams"].get("hidden_layers", [])
        txt = "→".join(str(x) for x in hl)
        ax_layers.text(
            rec["iteration"], 0.5, txt,
            ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
        )
    ax_layers.set_xlim([min(iters) - 0.5, max(iters) + 0.5])
    ax_layers.set_ylim([0, 1])
    ax_layers.set_yticks([])
    ax_layers.set_xticks(iters)
    ax_layers.set_ylabel("hidden_layers")
    ax_layers.grid(True, alpha=0.2)

    # ── Validation accuracy + categorical annotations ─────────────────────
    ax_va = axes[n_num + 1]
    ax_va.plot(iters, val_acc, marker="D", color="darkorange",
               linewidth=2.5, markersize=8, label="val_acc")
    ax_va.set_ylim([0, 1.05])
    ax_va.set_ylabel("val_accuracy")
    ax_va.set_xlabel("Iteration")
    ax_va.set_xticks(iters)
    ax_va.grid(True, alpha=0.3)
    ax_va.legend(fontsize=8)

    # Annotate categorical params as small text above the x-axis
    for rec in history_list:
        txt = f"act={rec['hyperparams']['activation']}  opt={rec['hyperparams']['optimizer']}"
        ax_va.text(
            rec["iteration"], -0.12, txt,
            ha="center", va="top", fontsize=7, transform=ax_va.get_xaxis_transform(),
        )

    # Mark best iteration
    best_iter_idx = int(np.argmax(val_acc))
    ax_va.scatter([iters[best_iter_idx]], [val_acc[best_iter_idx]],
                  s=180, zorder=5, color="gold", edgecolors="darkorange",
                  linewidths=2, label="best", marker="*")
    ax_va.legend(fontsize=8)

    plt.tight_layout()
    path = _save(fig, "hyperparameter_evolution.png", save_dir, show)
    plt.close(fig)
    return path


# ===========================================================================
# 3. Agent reasoning timeline
# ===========================================================================

def plot_agent_reasoning(
    summary:  dict,
    save_dir: Optional[str] = None,
    show:     bool          = False,
) -> str:
    """
    Visualise the agent's reasoning chain as a vertical timeline.

    Each iteration node shows:
      • val_accuracy achieved
      • The agent's diagnosis
      • The key hyperparameter changes suggested
    """
    plt, _ = _get_plt()

    history_list = summary.get("history", [])
    if not history_list:
        return ""

    n    = len(history_list)
    fig_h = max(6, n * 2.2)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.set_xlim([-0.2, 1.2])
    ax.set_ylim([-0.5, n])
    ax.axis("off")
    ax.set_title("Agent Reasoning Timeline", fontsize=14, fontweight="bold", pad=16)

    # Best iteration
    val_accs   = [r["val_metrics"]["accuracy"] for r in history_list]
    best_idx   = int(np.argmax(val_accs))

    cmap_node  = plt.cm.RdYlGn
    LINE_X     = 0.08

    for i, rec in enumerate(history_list):
        y    = n - 1 - i   # draw top → bottom
        acc  = rec["val_metrics"]["accuracy"]
        diag = rec.get("diagnosis") or "—"
        chgs = rec.get("changes") or {}
        hp   = rec["hyperparams"]

        # ── Timeline line ────────────────────────────────────────────
        if i < n - 1:
            ax.plot([LINE_X, LINE_X], [y - 0.9, y],
                    color="#cccccc", lw=2, zorder=1)

        # ── Node circle ───────────────────────────────────────────────
        node_colour = cmap_node(acc)
        star = (i == best_idx)
        marker = "*" if star else "o"
        ms     = 22  if star else 14
        ax.plot(LINE_X, y, marker=marker, markersize=ms,
                color=node_colour, markeredgecolor="black", zorder=3)
        ax.text(LINE_X, y, f"{acc:.3f}",
                ha="center", va="center", fontsize=7, zorder=4,
                color="black", fontweight="bold" if star else "normal")

        # ── Iteration label ───────────────────────────────────────────
        ax.text(LINE_X - 0.06, y,
                f"Iter {rec['iteration']}", ha="right", va="center", fontsize=9,
                color="#333333")

        # ── Diagnosis box ─────────────────────────────────────────────
        diag_text = textwrap.fill(diag, 38)
        ax.text(LINE_X + 0.06, y + 0.18, diag_text,
                ha="left", va="top", fontsize=8,
                color="#1a237e",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="#e3f2fd", alpha=0.8, edgecolor="#90caf9"))

        # ── Hyperparameter snapshot ───────────────────────────────────
        hp_txt = (
            f"lr={hp['learning_rate']}  layers={hp['hidden_layers']}  "
            f"act={hp['activation']}  opt={hp['optimizer']}  "
            f"bs={hp['batch_size']}  drop={hp['dropout_rate']}"
        )
        ax.text(LINE_X + 0.06, y - 0.22, hp_txt,
                ha="left", va="top", fontsize=7.5, color="#555555", style="italic")

        # ── Changes arrow + text (from previous) ─────────────────────
        if chgs:
            chg_lines = [f"↳ {k}: {v['from']} → {v['to']}" for k, v in chgs.items()]
            chg_text  = "\n".join(chg_lines[:6])  # cap at 6 lines
            ax.text(0.75, y, chg_text,
                    ha="left", va="center", fontsize=7.5, color="#bf360c",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="#fff3e0", alpha=0.85, edgecolor="#ffcc80"))

    # Legend
    ax.text(0.75, n - 0.3, "Parameter Changes →", fontsize=9,
            color="#bf360c", ha="left", fontstyle="italic")
    if best_idx < n:
        ax.text(LINE_X, n - 0.1, "★ = best", ha="center", fontsize=8, color="goldenrod")

    plt.tight_layout()
    path = _save(fig, "agent_reasoning.png", save_dir, show)
    plt.close(fig)
    return path


# ===========================================================================
# 4. Confusion matrix
# ===========================================================================

def plot_confusion_matrix(
    confusion_matrix: List[List[int]],
    class_names:      List[str],
    title:            str           = "Confusion Matrix (Best Model — Test Set)",
    save_dir:         Optional[str] = None,
    show:             bool          = False,
) -> str:
    plt, _ = _get_plt()

    cm  = np.array(confusion_matrix)
    n   = cm.shape[0]
    fig, ax = plt.subplots(figsize=(max(5, n * 0.8 + 2), max(4, n * 0.8 + 1.5)))

    # Normalised for colour, raw for text
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues,
                   vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Normalised count")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title, fontsize=11, fontweight="bold")

    thresh = cm_norm.max() / 2.0
    for r in range(n):
        for c in range(n):
            colour = "white" if cm_norm[r, c] > thresh else "black"
            ax.text(c, r, f"{cm[r, c]}\n({cm_norm[r, c]:.0%})",
                    ha="center", va="center", fontsize=8, color=colour)

    plt.tight_layout()
    path = _save(fig, "confusion_matrix.png", save_dir, show)
    plt.close(fig)
    return path


# ===========================================================================
# 5. Final dashboard (combined)
# ===========================================================================

def plot_final_dashboard(
    summary:     dict,
    class_names: List[str],
    save_dir:    Optional[str] = None,
    show:        bool          = False,
) -> str:
    """
    Produce a single-page overview dashboard with:
      • Validation accuracy across iterations
      • Best-iteration training curves
      • Hyperparameter change table
      • Key final metrics
    """
    plt, gridspec = _get_plt()

    history_list = summary.get("history", [])
    if not history_list:
        return ""

    iters    = [r["iteration"] for r in history_list]
    val_accs = [r["val_metrics"]["accuracy"] for r in history_list]
    best_i   = int(np.argmax(val_accs))
    best_rec = history_list[best_i]

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Agentic Neural Network Optimisation — Final Dashboard",
                 fontsize=15, fontweight="bold", y=1.01)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # ── (0,0) Val accuracy progress ──────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    colours = ["gold" if i == best_i else "#4a90d9" for i in range(len(iters))]
    bars = ax0.bar(iters, val_accs, color=colours, edgecolor="white", linewidth=0.8)
    ax0.set_xlabel("Iteration")
    ax0.set_ylabel("Val Accuracy")
    ax0.set_title("Validation Accuracy per Iteration")
    ax0.set_ylim([0, 1.05])
    ax0.set_xticks(iters)
    ax0.grid(True, alpha=0.3, axis="y")
    for bar, acc in zip(bars, val_accs):
        ax0.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{acc:.3f}", ha="center", va="bottom", fontsize=8)

    # ── (0,1) Best-iteration loss curve ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    bh  = best_rec["history"]
    ep  = list(range(1, len(bh["train_loss"]) + 1))
    ax1.plot(ep, bh["train_loss"], label="train_loss", color="#e74c3c", lw=2)
    ax1.plot(ep, bh["val_loss"],   label="val_loss",   color="#e74c3c", lw=2, ls="--")
    ax1.set_title(f"Best Iter {best_rec['iteration']} — Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    # ── (0,2) Best-iteration accuracy curve ──────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(ep, bh["train_acc"], label="train_acc", color="#27ae60", lw=2)
    ax2.plot(ep, bh["val_acc"],   label="val_acc",   color="#27ae60", lw=2, ls="--")
    ax2.set_title(f"Best Iter {best_rec['iteration']} — Accuracy")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_ylim([0, 1.05])
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    # ── (1,0) Confusion matrix ────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    best_metrics = summary.get("best_metrics", {})
    test_metrics = best_metrics.get("test", {})
    if "confusion_matrix" in test_metrics:
        cm     = np.array(test_metrics["confusion_matrix"])
        cm_n   = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
        im     = ax3.imshow(cm_n, cmap=plt.cm.Blues, vmin=0, vmax=1)
        n_cls  = cm.shape[0]
        thresh = cm_n.max() / 2
        for r in range(n_cls):
            for c in range(n_cls):
                col = "white" if cm_n[r, c] > thresh else "black"
                ax3.text(c, r, str(cm[r, c]), ha="center", va="center",
                         fontsize=7, color=col)
        ax3.set_xticks(range(n_cls))
        ax3.set_yticks(range(n_cls))
        lbl = [cn[:8] for cn in class_names]
        ax3.set_xticklabels(lbl, rotation=45, ha="right", fontsize=7)
        ax3.set_yticklabels(lbl, fontsize=7)
        ax3.set_title("Confusion Matrix (Test Set)")
        ax3.set_xlabel("Predicted"); ax3.set_ylabel("True")
    else:
        ax3.text(0.5, 0.5, "No confusion matrix\navailable",
                 ha="center", va="center", transform=ax3.transAxes)
        ax3.set_title("Confusion Matrix")

    # ── (1,1) Hyperparameter table ────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    best_hp = summary.get("best_hp", {})
    rows    = [[k, str(v)] for k, v in best_hp.items()]
    tbl     = ax4.table(
        cellText   = rows,
        colLabels  = ["Hyperparameter", "Best Value"],
        cellLoc    = "left",
        loc        = "center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    ax4.set_title("Best Hyperparameters", pad=14, fontweight="bold")

    # ── (1,2) Summary metrics text ────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")

    final_val  = best_metrics.get("val",  {})
    final_test = best_metrics.get("test", {})
    lines = [
        "Final Performance (Best Model)",
        "",
        f"  Val  accuracy  : {final_val.get('accuracy', '?'):.4f}",
        f"  Test accuracy  : {final_test.get('accuracy', '?'):.4f}",
        f"  Test macro-F1  : {final_test.get('macro_f1', '?'):.4f}",
        f"  Test precision : {final_test.get('macro_precision', '?'):.4f}",
        f"  Test recall    : {final_test.get('macro_recall', '?'):.4f}",
        "",
        f"  Total iterations : {summary.get('total_iterations', '?')}",
        f"  Classes          : {len(class_names)}",
    ]
    ax5.text(0.05, 0.95, "\n".join(lines),
             ha="left", va="top", transform=ax5.transAxes,
             fontsize=9, family="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f4f8", alpha=0.9))

    plt.tight_layout()
    path = _save(fig, "final_dashboard.png", save_dir, show)
    plt.close(fig)
    return path


# ===========================================================================
# Convenience: generate all plots
# ===========================================================================

def generate_all_plots(
    summary:     dict,
    class_names: List[str],
    save_dir:    str,
    show:        bool = False,
) -> List[str]:
    """Generate all visualisations and return a list of saved file paths."""
    os.makedirs(save_dir, exist_ok=True)
    paths: List[str] = []

    logger.info("Generating visualisations in '%s' …", save_dir)

    for fn, kwargs in [
        (plot_training_curves,         {"summary": summary, "save_dir": save_dir, "show": show}),
        (plot_hyperparameter_evolution, {"summary": summary, "save_dir": save_dir, "show": show}),
        (plot_agent_reasoning,          {"summary": summary, "save_dir": save_dir, "show": show}),
        (plot_final_dashboard,          {"summary": summary, "class_names": class_names,
                                         "save_dir": save_dir, "show": show}),
    ]:
        try:
            p = fn(**kwargs)
            if p:
                paths.append(p)
                logger.info("  ✓  %s", os.path.basename(p))
        except Exception as exc:
            logger.warning("  ✗  %s failed: %s", fn.__name__, exc)

    # Confusion matrix from best model
    best_metrics = summary.get("best_metrics", {})
    test_metrics = best_metrics.get("test", {})
    if "confusion_matrix" in test_metrics:
        try:
            p = plot_confusion_matrix(
                test_metrics["confusion_matrix"],
                class_names,
                save_dir=save_dir, show=show,
            )
            if p:
                paths.append(p)
                logger.info("  ✓  %s", os.path.basename(p))
        except Exception as exc:
            logger.warning("  ✗  confusion_matrix failed: %s", exc)

    logger.info("Saved %d visualisation(s).", len(paths))
    return paths


# ===========================================================================
# Helper
# ===========================================================================

def _save(
    fig,
    filename: str,
    save_dir: Optional[str],
    show:     bool,
) -> str:
    if show:
        try:
            import matplotlib.pyplot as plt
            plt.show()
        except Exception:
            pass

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)
        fig.savefig(path, dpi=130, bbox_inches="tight")
        return path
    return ""
