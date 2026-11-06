# QuantSteps (Investment Trying Lab)

A Python research toolkit for rule-first quantitative trading: chart pattern detection, signal scoring, backtesting, ML and regime experiments, risk models, and paper-trading signals.

The public source includes research components and experiment records. It does not establish production readiness or guarantee returns.

---

## Reported Experiments (Q2 2026)

[BESTS.md](BESTS.md) reports a mean instrument Sharpe of **1.135** for a selected 17-instrument basket in a May 2026 re-run, with IS-to-OOS correlation of **-0.388**. Its summary and individual rows report 17/17 positive instruments, while its tier table lists tier B as 5/6 positive. That inconsistency needs reconciliation before reusing the positive-rate summary.

A separate 2022-2026 comparison reports 12/18 (67%) OOS-positive instruments. These figures were not reproduced in this documentation review. They describe selected historical runs, not expected returns, a verified portfolio Sharpe, or a general relationship between IS and OOS performance. The original leaderboard remains unchanged.

---

## Architecture

| Source area | Purpose |
|-------------|---------|
| [`src/patterns/`](src/patterns/) | Chart patterns, similarity search, recognition wrappers, factor DSL |
| [`src/strategies/`](src/strategies/) | Rules-first, combined, ML, regime, SMC and indicator strategies |
| [`src/signals/`](src/signals/) | Signal aggregation, quality registries, confluence and factor diagnostics |
| [`src/indicators/`](src/indicators/) | Technical indicators, shared caches and SMC/ICT helpers |
| [`src/ml/`](src/ml/) | Features, labeling, model training, regime models and evaluation helpers |
| [`src/backtest/`](src/backtest/) | Custom event-driven backtesting |
| [`src/risk/`](src/risk/) and [`src/portfolio/`](src/portfolio/) | Sizing, risk estimates, limits and allocation models |
| [`src/optimization/`](src/optimization/) | Parameter search and portfolio optimization |
| [`src/data/`](src/data/), [`src/features/`](src/features/) and [`src/nlp/`](src/nlp/) | Market-data helpers, derived features and sentiment experiments |
| [`src/analysis/`](src/analysis/) | Attribution, ablation and statistical diagnostics |
| [`scripts/`](scripts/) and [`tests/`](tests/) | CLI workflows and tests |

**Rules-first baseline:** [`RulesFirstStrategy`](src/strategies/rules_first_strategy.py) combines pattern signals with reliability weights, entry/exit thresholds and optional gates. Compare ML additions against this baseline for the same data, costs and evaluation period.

---

## Research Practices

- Compare exit rules, confluence thresholds and signal additions through controlled experiments.
- Keep model selection, preprocessing and final evaluation data separate.
- Include transaction-cost assumptions and report failures alongside favorable runs.
- Treat instrument-selection rules and parameter sweeps as hypotheses to retest.

[BESTS_INSIGHTS.md](docs/BESTS_INSIGHTS.md) records conclusions from past experiments. Those conclusions depend on their data and configurations.

---

## System Capabilities

### Chart Pattern Detection

The catalog includes basic, breakout, candlestick, classic, complex, continuation, harmonic, FMZ, technical and range-persistence patterns, plus additional pattern modules, similarity search and a Patternity wrapper. Each strategy selects its own active subset; the catalog size is not the number of detectors enabled in every backtest.

### ML Infrastructure

- **Training and evaluation:** Feature extraction, triple-barrier labeling, purged and nested CV helpers, calibration, meta-labeling and walk-forward workflows.
- **Models:** CatBoost, LightGBM, EBM and AutoGluon components, classifiers and ensembles.
- **Explainability:** SHAP utilities, feature importance and stability selection.
- **Regime experiments:** HMM, GMM, SVM, CNN, change-point, ensemble, macro and path-signature modules.
- **Generative experiments:** OHLCV augmentation, anomaly detection and forecasting/fine-tuning modules. Model weights, hardware and data requirements vary.

### Risk & Position Sizing

Kelly allocator, signal-strength dynamic sizing, VIX regime sizing, circuit breakers, daily loss limits, binomial VaR/CVaR, copula risk, diversity constraints, strategy-aware sizing, market impact estimation.

### Optimization

NSGA2 multi-objective, Two-Phase GA (per-rule then ensemble), Dynamic GA (per-regime memory), skfolio (MV/CVaR/HRP/RiskBudget/InvVol).

### Data & Signals

- **Alternative data:** Congressional signals, futures inventory, options and fund-flow helpers.
- **Sentiment:** SVM/TF-IDF, BiLSTM and ensemble experiments, plus an Adanos API integration.
- **Fundamentals:** Screening and financial-data helpers; provider availability and credentials affect usable data.
- **Cross-asset:** Sector maps, VAR/Granger analysis, Kalman hedge ratios and wavelet/FFT transforms.
- **Historical universes:** Point-in-time membership helpers; delisting coverage depends on the supplied data.

### Stock Selection

The [stock selection criteria](docs/stock_selection_criteria.md) document market-cap, liquidity, price, history, exchange, ETF and sector filters for research baskets.

## Evaluation Limits

The presence of CV, label-shuffling or LockBox utilities does not establish that every experiment uses them correctly. LockBox tracks access through its object API; it is not an independent audit of an entire research workflow.

**Known holdout contamination:** [`src/ml/blind_analysis.py`](src/ml/blind_analysis.py) fits the final model on `X_holdout.iloc[:-1]` and `y_holdout.iloc[:-1]` when the holdout has more than one row, then scores the whole holdout. Its `true_score` therefore includes training observations. Its `is_honest` flag and reported score must not be treated as blind or independent OOS validation. This documentation change does not repair that implementation.

---

## Quick Start

Run from the repository root after installing [uv](https://github.com/astral-sh/uv). The backtest examples require local daily CSVs at `data/raw/SPY_daily.csv` and `data/raw/QQQ_daily.csv`, with a date index and `Open`, `High`, `Low`, `Close`, `Volume` columns covering the evaluation windows. This public snapshot does not include those data files; `uv sync` installs dependencies, not market history.

```bash
# Install dependencies
uv sync

# Backtest one rules-first configuration
uv run scripts/backtest_rules_first.py SPY --start 2025-01-01 --end 2025-12-31 --entry-threshold 0.55 --min-reliability 0.70

# Compare instruments using the comprehensive runner's configured date windows
uv run scripts/backtest_all_comprehensive.py --tickers SPY,QQQ --use-best

# Generate and log hypothetical daily signals
uv run scripts/paper_trade_daily.py --basket SPY,QQQ
```

The daily paper-signal workflow can fetch market data and create local logs. Examples were checked against source declarations, not executed in this review.

The comprehensive runner declares `--period`, but its current `main()` calls `run_ticker()` without applying that option; do not use the flag to assert an OOS-only run.

Full command reference: [COMMAND_CHEATSHEET.md](docs/COMMAND_CHEATSHEET.md). Check the target script for current options and prerequisites.

---

## Documentation

| Document | Purpose |
|----------|---------|
| [BESTS.md](BESTS.md) | Backtest leaderboard — all best results by configuration |
| [BESTS_INSIGHTS.md](docs/BESTS_INSIGHTS.md) | Historical experiment interpretations and hypotheses |
| [PROJECT_SHOWCASE.md](PROJECT_SHOWCASE.md) | Project overview and historical development snapshot |
| [TECHNICAL_INTRODUCTION.md](docs/TECHNICAL_INTRODUCTION.md) | Architecture notes with source-review qualifications |
| [COMMAND_CHEATSHEET.md](docs/COMMAND_CHEATSHEET.md) | CLI reference; verify options against the target script |
| [stock_selection_criteria.md](docs/stock_selection_criteria.md) | 11 hard filters + 9 desirability dimensions for ticker selection |
| [ML_TRAINING_GUIDE.md](docs/ML_TRAINING_GUIDE.md) | ML components, decision tree, data flow, file location reference |
| [guide-ml-pipeline.md](docs/guide-ml-pipeline.md) | Standard 9-stage ML training pipeline |
| [blind_analysis.py](src/ml/blind_analysis.py) | Experimental helper with the holdout-contamination limit described above |
| [GPU_TASK_QUEUE.md](docs/GPU_TASK_QUEUE.md) | GPU-blocked tasks with self-contained tutorials |
| [reference_phase28_external_tools.md](docs/reference_phase28_external_tools.md) | Indicators comparison, OpenStock architecture, WFGY evaluation |

---

## Research & Knowledge Management

- **Insight registry:** Research notes in [insight_registry.md](docs/research_logic_map/insight_registry.md)
- **Web source registry:** Non-paper web sources with W-IDs in [web_source_registry.md](docs/research_logic_map/web_source_registry.md)
- **Attribution:** Source references appear in module docstrings and research registries. A reference or wrapper is not evidence that the cited result has been reproduced here.

---

## Progress & Planning

The [master plan](progress_docs/plans/full.md) and [session log](progress_docs/current.md) preserve development history. Their completion labels and development-machine inventories are not release or production-readiness certifications.

---

## Requirements

- Project metadata specifies Python `>=3.12,<3.14`; dependency compatibility can narrow usable environments.
- Package management via [uv](https://github.com/astral-sh/uv)
- Storage, GPU requirements and API access depend on the selected workflow and downloaded artifacts.

## Development

```bash
uv sync --group dev     # Install declared dependencies and development tools
uv run pytest           # Run test suite
uv run ruff check .     # Lint
uv run ruff format .    # Format
```

Tests cover individual components and workflows; no test suite, training job or benchmark was run for this documentation review.
