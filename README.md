# Agentic Neural Network Hyperparameter Optimiser

An **Agentic AI** demonstration system for a course on Agentic AI.

An LLM acts as an autonomous agent that iteratively trains a neural network, analyses its performance, and modifies hyperparameters to maximise validation accuracy — all logged and visualised automatically.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         main.py                                 │
│              Interactive wizard + session runner                │
└──────────────┬──────────────────────────────┬───────────────────┘
               │                              │
    ┌──────────▼──────────┐       ┌───────────▼──────────┐
    │    data_loader.py   │       │      agent.py         │
    │  sklearn / local    │       │  LLM or heuristic     │
    │  CSV / NPZ / Numpy  │       │  hyperparameter loop  │
    └─────────────────────┘       └───────────┬───────────┘
                                              │
                             ┌────────────────▼────────────────┐
                             │         neural_network.py        │
                             │   Pure-NumPy feedforward NN      │
                             │  He/Xavier init · Adam/SGD opt   │
                             │  L2 reg · Inverted dropout        │
                             └────────────────┬────────────────┘
                                              │
               ┌──────────────────────────────▼──────────────────┐
               │  logger_setup.py              visualizer.py      │
               │  Coloured console             Training curves     │
               │  Rotating text log            HP evolution        │
               │  JSON-lines events            Agent timeline      │
               │                               Confusion matrix    │
               └─────────────────────────────────────────────────┘
```

### How the Agent Loop Works

```
  ┌─────────────┐
  │  User sets  │  Initial hyperparameters + LLM API info
  │  initial HP │
  └──────┬──────┘
         │
         ▼
  ┌─────────────────┐
  │  Build & Train  │  NeuralNetwork from scratch (NumPy)
  │  Neural Network │  Mini-batch SGD · Adam/SGD
  └──────┬──────────┘
         │
         ▼
  ┌─────────────────┐
  │  Evaluate on    │  Accuracy · Loss · F1 · Confusion Matrix
  │  Val / Test     │
  └──────┬──────────┘
         │
         ▼
  ┌─────────────────────────────────┐
  │  Agent (LLM or heuristic)       │  Tool-calling API:
  │  Analyses:                      │  suggest_hyperparameters()
  │    • training curves            │
  │    • train/val gap              │  Returns structured JSON with:
  │    • history of all iterations  │    • reasoning (chain-of-thought)
  │    • dataset characteristics    │    • diagnosis
  │                                 │    • new hyperparameters
  └──────┬──────────────────────────┘
         │
         ▼
  ┌─────────────────┐
  │  Log & Visualise│  JSON-lines events · Matplotlib plots
  └──────┬──────────┘
         │
    [Repeat until target accuracy / patience / max iterations]
         │
         ▼
  ┌─────────────────┐
  │  Final Report   │  Dashboard · Confusion matrix · Summary JSON
  └─────────────────┘
```

---

## Files

| File | Purpose |
|---|---|
| `main.py` | Entry point — interactive wizard and CLI |
| `config.py` | Configuration dataclasses (HyperParameters, LLMConfig, AgentConfig, DataConfig) |
| `data_loader.py` | Load sklearn or local (CSV / .npz) datasets |
| `neural_network.py` | Pure-NumPy feedforward NN with Adam/SGD, dropout, L2 reg |
| `agent.py` | LLM-powered optimiser + rule-based heuristic fallback |
| `logger_setup.py` | Coloured console, rotating file log, JSON-lines event stream |
| `visualizer.py` | Matplotlib: training curves, HP evolution, agent timeline, dashboard |
| `requirements.txt` | Python dependencies |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the interactive wizard

```bash
python main.py
```

The wizard prompts you for:
- LLM API key and endpoint (or press Enter to use the built-in heuristic agent)
- Dataset (sklearn built-in, or path to a local CSV/.npz)
- Starting hyperparameters
- Agent loop settings (max iterations, target accuracy, patience)

### 3. Demo mode (no API key needed)

```bash
python main.py --demo
```

Uses the `iris` dataset and the heuristic agent for 3 iterations.

### 4. CLI-only (skip wizard)

```bash
python main.py \
  --no-wizard \
  --dataset digits \
  --api-key sk-your-key-here \
  --model gpt-4o-mini \
  --max-iter 10 \
  --target-acc 0.98
```

---

## Datasets

### Built-in (scikit-learn)

| Name | Classes | Features | Samples |
|---|---|---|---|
| `iris` | 3 | 4 | 150 |
| `digits` | 10 | 64 | 1 797 |
| `breast_cancer` | 2 | 30 | 569 |
| `wine` | 3 | 13 | 178 |

### Local datasets

**CSV:** The last column is used as the label by default. Provide `--local-data path/to/file.csv` and optionally configure the label column in the wizard.

**NumPy .npz:** Must contain arrays named `X` (features) and `y` (integer labels).

---

## Hyperparameters Tuned

| Parameter | Description | Typical Range |
|---|---|---|
| `learning_rate` | Gradient step size | 1e-4 – 0.1 |
| `hidden_layers` | Neuron count per hidden layer | e.g. `[128, 64]` |
| `activation` | Hidden layer activation | `relu`, `sigmoid`, `tanh` |
| `optimizer` | Gradient update rule | `adam`, `sgd` |
| `batch_size` | Mini-batch size | 8 – 256 |
| `epochs` | Training epochs | 20 – 300 |
| `l2_lambda` | L2 regularisation strength | 0 – 0.1 |
| `dropout_rate` | Dropout probability on hidden layers | 0 – 0.5 |

---

## LLM Configuration

Any **OpenAI-compatible** endpoint is supported (OpenAI, Anthropic via compatibility layer, Ollama, LM Studio, etc.).

| Setting | Description | Default |
|---|---|---|
| `api_key` | Your API key | _(required for LLM)_ |
| `base_url` | API base URL | `https://api.openai.com/v1` |
| `model` | Model name | `gpt-4o-mini` |

**No API key?** Leave it blank — the system falls back to a built-in rule-based heuristic agent that still demonstrates the full agentic loop.

---

## Outputs

After a session, everything is saved to:

```
logs/
  session_YYYYMMDD_HHMMSS.log        # human-readable rotating log
  session_YYYYMMDD_HHMMSS.jsonl      # JSON-lines event stream
  session_YYYYMMDD_HHMMSS_config.json

results/
  session_YYYYMMDD_HHMMSS/
    training_curves.png              # loss + accuracy curves per iteration
    hyperparameter_evolution.png     # how each HP changed over iterations
    agent_reasoning.png              # agent decision timeline
    final_dashboard.png              # single-page combined overview
    confusion_matrix.png             # best model test-set confusion matrix
  session_YYYYMMDD_HHMMSS_summary.json
```

---

## Neural Network Details

The network is implemented **from scratch in NumPy** — no PyTorch or TensorFlow required.

- **Architecture:** Input → `[Hidden × N]` → Softmax output
- **Weight init:** He (ReLU) or Xavier (sigmoid/tanh)
- **Optimisers:** Adam (with bias correction) and SGD
- **Regularisation:** L2 weight decay + inverted dropout
- **Loss:** Cross-entropy (numerically stable softmax)
- **Best-model tracking:** Weights are restored to the epoch with highest val accuracy

---

## Agentic Reasoning

When an LLM is configured, the agent uses the **OpenAI tool-calling API** (function calling) with a single structured tool `suggest_hyperparameters`. This forces the model to return:

- `reasoning` — step-by-step analysis of the training curves
- `diagnosis` — one-sentence summary of the issue
- `changes_summary` — what changed and why
- `expected_outcome` — predicted improvement

All reasoning is logged to the JSON-lines file and displayed in the agent reasoning timeline visualisation.

---

## Example Session (heuristic agent, iris, 5 iterations)

```
Iteration 1: lr=0.001  layers=[64,32]  val_acc=0.9200
  Diagnosis: Underfitting — model too small
  Change:    layers [64,32] → [128,64], epochs 50 → 70

Iteration 2: lr=0.001  layers=[128,64]  val_acc=0.9600
  Diagnosis: Good fit — fine-tune learning rate
  Change:    lr 0.001 → 0.0007, epochs 70 → 90

Iteration 3: lr=0.0007  layers=[128,64]  val_acc=0.9733  ★ best
  Diagnosis: Excellent — minor overfitting detected
  Change:    dropout 0.0 → 0.1, l2 1e-4 → 3e-4
  ...
```

---

## Requirements

```
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.3.0   # for built-in datasets only
openai>=1.10.0        # for LLM agent only
```

The core neural network and heuristic agent have **no external dependencies** beyond NumPy.
