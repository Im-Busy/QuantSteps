# Investment Trying Lab — Project Showcase

> QuantSteps is a quantitative trading research toolkit with chart patterns, rule-first strategies, machine learning experiments, risk models and backtesting workflows.

**Public-source review (2026-11-07):** use the [README](README.md) for current public capabilities, setup and evaluation limits. The dated inventories, architecture sketch and experiment tables below preserve the development snapshot reported on 2026-05-18. Counts of papers, model artifacts, agents and indexed symbols are historical reports, not inventories verified in the public mirror. Performance figures describe selected experiments and have not been reproduced in this review.

---

## Historical Development Inventory (2026-05-18)

| Metric | Value |
|--------|-------|
| **Python source files** | 364 across 20+ modules |
| **Pattern detectors** | 54 across 10 categories (basic, candlestick, classic, complex, harmonic, breakout, continuation, fmz, technical, range-persistence) |
| **Trading strategies** | 55 (indicator-based, pattern-based, ML-based, pairs, portfolio, multi-timeframe) |
| **ML model artifacts** | 265 (186 .pkl + 79 .json configs across CatBoost, LightGBM, ensembles, meta-labelers) |
| **Research papers integrated** | 97 (45 PDFs + 52 MDs; sentiment, overfitting, RL, event-driven, forecasting, risk) |
| **Research insights** | 77 from 21 sources in insight registry |
| **Test coverage** | 66 test files |
| **CLI scripts** | 119 |
| **Lines of code** | ~187,000 (src + scripts + tests) |
| **Kilo AI agents** | 6 agents, 7 slash commands, 40 skills |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    SIGNAL GENERATION                     │
├──────────────┬──────────────┬──────────────┬────────────┤
│ 54 Pattern   │ 55 Indicator │ ML Classifier│ Foundation  │
│ Detectors    │ Strategies   │ (CatBoost v3)│ Models      │
│ (+reliability│ (SMA,MACD,   │ + Meta-Label │ (Chronos,   │
│  weights)    │  RSI,ICH,etc)│ + Ensemble   │  FinCast)   │
└──────┬───────┴──────┬───────┴──────┬───────┴──────┬─────┘
       │              │              │              │
       └──────────────┴──────────────┴──────────────┘
                          │
              ┌───────────▼───────────┐
              │   SIGNAL AGGREGATION   │
              │  PatternBoostFilter    │
              │  Confluence Scoring    │
              │  Event Weighting       │
              └───────────┬───────────┘
                          │
       ┌──────────────────┼──────────────────┐
       │                  │                  │
┌──────▼──────┐   ┌───────▼───────┐   ┌──────▼──────┐
│   REGIME    │   │     RISK      │   │  PORTFOLIO  │
│  Detection  │   │  Management   │   │  Allocation │
│ (HMM, GMM,  │   │ (Kelly, ATR,  │   │ (Black-Lit- │
│  PCA-KMeans,│   │  Circuit Brk, │   │  terman,    │
│  CNN, R²,   │   │  VaR/CVaR,   │   │  PyPortOpt, │
│  Changepoint│   │  Diversity)   │   │  Multi-Str) │
│  8 methods) │   │               │   │             │
└──────┬──────┘   └───────┬───────┘   └──────┬──────┘
       │                  │                  │
       └──────────────────┼──────────────────┘
                          │
              ┌───────────▼───────────┐
              │   BACKTEST ENGINE      │
              │  Custom Event-Driven   │
              │  + backtesting.py      │
              │  + VectorBT adapter    │
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │   ANALYSIS & REPORTS   │
              │  DSR/PSR/FDR, SHAP,   │
              │  Ablation, Contrib.,  │
              │  Regime Analysis,     │
              │  Calibration Audit    │
              └───────────────────────┘
```

---

## Research Components and Reported Experiments

### ML Pipeline (9-stage)
- **Triple-barrier labeling** with dynamic take-profit/stop-loss from ATR
- **Purged K-Fold CV** + **Combinatorial Purged CV** (CPCV) helpers for controlling overlap around evaluation folds; correct label horizons and data handling remain necessary
- **Stability Selection** (N=100 bootstrap) replacing single-run feature importance
- **Meta-labeling** — secondary CatBoost filter; a historical experiment reported Sharpe +36% (0.69→0.94)
- **Dynamic ensemble** — 5-model weighted ensemble with regime-awareness
- **Isotonic calibration** — a historical experiment reported ECE from 0.197 to 0.070 and IS Sharpe from 0.59 to 0.85 after a calibration change
- **Optuna Bayesian hyperparameter optimization** with multi-objective Pareto front
- **Walk-forward workflows** with per-ticker IC tracking; historical notes describe runs across 125+ tickers
- **Feature health monitoring** — KS drift tests, KL divergence, correlation flips, retrain recommendation

### Baseline Pattern Catalog

This table records the historical catalog grouping and weights. The current public tree contains additional pattern modules, and each strategy chooses its active subset. The weights are scoring inputs, not calibrated probabilities of profitable trades.

| Category | Patterns | Reliability Weights |
|----------|----------|-------------------|
| **Classic** | Double Top/Bottom, Triple Top/Bottom, Asc./Desc. Triangle, Rectangle, Wedge, Dead Cat Bounce, Trader Vic 2B | 0.75–0.87 (from NCFE/Duddella research) |
| **Candlestick** | Engulfing, Hammer, Doji, Harami, Dark Cloud Cover, Piercing Line | 0.65–0.78 |
| **Harmonic** | Gartley, ABC Correction, Symmetric Triangle, Butterfly, Bat, Crab, Cypher, Shark, Bollinger Bands | 0.80–0.85 |
| **Complex** | Head & Shoulders, Cup & Handle, Parabolic Arc, Spike & Ledge, Three Hills, Pipe | 0.80–0.87 |
| **Breakout** | Donchian Channel, Gap Detection | 0.70 |
| **Continuation** | Flag, Pennant | 0.72–0.78 |
| **Basic** | Floor Pivot, MSL, NR7ID, N-Bar Decline, 2-Bar Reversal, Matching Lows | 0.60–0.72 |
| **FMZ** (PineScript→Python) | AlphaBeast, MultiFactorTrend, MomentumZigZag, EMAMACDHF, AdaptiveBollinger, AIVolatilityBreakout | 0.65–0.78 |
| **Technical** | Ichimoku Cloud, Keltner Channel, Williams %R, CCI | 0.68–0.75 |
| **Range Persistence** | PersistentRange, ContractingRange, ExpandingRange | 0.70–0.80 |

### Feature Engineering
- **Price features**: returns, log returns, volatility (ATR-normalized after regime shift fix)
- **Technical indicators**: MACD, RSI, Bollinger Bands, ADX, EMAs (multi-timeframe)
- **Cross-asset**: relative returns, beta, correlation to SPY/QQQ/TLT/GLD/XLE
- **Volume**: volume profile, OBV, volume regime
- **Regime**: volatility regime, trend strength, market regime classification
- **Risk**: drawdown, VaR proxy, position concentration
- **Fractional differentiation** for stationarity without information loss

### Backtesting & Validation
- **Custom event-driven engine** with configurable slippage, commission and market-hours assumptions
- **backtesting.py integration** for rapid strategy prototyping
- **DSR (Deflated Sharpe Ratio)** — corrects for multiple testing bias
- **PSR (Probabilistic Sharpe Ratio)** — probability that Sharpe exceeds threshold
- **FDR (False Discovery Rate)** — Benjamini-Hochberg correction
- **Ablation studies** — isolate per-pattern contribution to P&L
- **Correlation decomposition** — identify signal redundancy
- **Synergy analysis** — detect pattern co-occurrence effects
- **Walk-forward paper trading** — expanding-window OOS with feature recomputation

### Historical Backtest Results (SPY 2016–2024)
| Configuration | Return | Sharpe | Trades | Win Rate | PF | Max DD |
|--------------|--------|--------|--------|----------|-----|--------|
| ML + Trail Stop (et=0.45) | 81.0% | 0.73 | 66 | 43.9% | 1.65 | -17.5% |
| ML + Trail + Meta-Label | 88.3% | 0.94 | 47 | 45.2% | 2.05 | -12.8% |
| ML Retrained (B9-B14 fixes) | 60.8% | 1.03 | 18 | 61.1% | 2.82 | -8.8% |

### Historical OOS Challenge
The recorded ML experiment reported Sharpe 0.85 in-sample and -1.24 out-of-sample (2025-2026). Its diagnostics reported 3/38 features with flipped correlations and `vol_regime` KS=0.81. These observations motivated further regime and feature research; they do not prove a unique cause of the performance change.

**Current evaluation limit:** `src/ml/blind_analysis.py` fits its final model on almost all holdout rows and then scores that same holdout. Its `true_score` and `is_honest` result do not establish blind OOS performance. LockBox and CV utilities elsewhere do not remove this contamination. See the [source-linked explanation](README.md#evaluation-limits). The implementation remains unchanged.

---

## Research Integration

### Historical Paper Coverage (Reported 2026-05-18)
| Topic | Papers | Implementation Status |
|-------|--------|----------------------|
| Time Series Forecasting | 35 | Partial (CatBoost, GARCH, Kalman, wavelet, VAR, ARIMA, ensemble, foundation model wrappers) |
| Machine Learning Methods | 30 | Strong (full pipeline, SHAP, calibration, Optuna, CPTV, structural break, model validation) |
| Overfitting & Data Leakage | 28 | Strong (CPCV, PurgedKFold, stability selection, meta-labeling, DSR) |
| Sentiment Analysis & NLP | 22 | Implemented (LM dictionary, FinBERT, SEC filing analyzer, multi-source fusion) |
| Backtesting & Validation | 21 | Strong (custom engine, DSR/PSR/FDR, WFO, ablation, 125+ ticker cross-validation) |
| Risk Management | 19 | Strong (VaR, circuit breakers, diversity, copula models, Almgren-Chriss impact) |
| Event-Driven Trading | 12 | Partial (gap pattern, order book features, microstructure signals) |
| Reinforcement Learning | 9 | Implemented (PPO/SAC trade execution, CQL offline RL) |
| Factor Models & Alpha | 8 | Partial (AutoAlpha, Huatai 4-phase pipeline, QRAFTI diagnostics, 11 factor engine) |
| Portfolio Optimization | 7 | Strong (Black-Litterman, PyPortfolioOpt, HRP, CVaR, Kelly criterion) |

### Knowledge Graph — Top Implementation Gaps
1. **HIGH**: Training-history-based overfitting detection (5520 paper)
2. **HIGH**: Synthetic OOS comparison framework (Backtest Overfitting paper)
3. **MEDIUM**: Event calendar + spike detection as pattern category
4. **MEDIUM**: Circuit-based overfitting detection
5. **MEDIUM**: Network-derived factor model (research phase)
6. **MEDIUM**: Tensor-based signal fusion (research phase)
7. **LOW**: Strategy decay monitoring & automatic retirement

### Chart Pattern Knowledge Base
727-line consolidated reference from 6 sources: Suri Duddella (65 patterns), Harmonic Trading Guide (6 Fibonacci patterns), NCFE Technical Analysis (reliability stats 75-88%), Fidelity Chart Patterns, Warrior Trading, and 151 Trading Strategies (Python implementations of 151 option strategies).

---

## Historical Infrastructure & Tooling

| Category | Tools |
|----------|-------|
| **Package management** | uv (Astral) |
| **ML frameworks** | CatBoost, LightGBM, scikit-learn, SHAP, InterpretML (EBM), AutoGluon |
| **Optimization** | Optuna (Bayesian), GWO, GA, WOA, ARO (metaheuristics) |
| **Portfolio** | PyPortfolioOpt, Black-Litterman |
| **Backtesting** | Custom event-driven engine, backtesting.py, VectorBT |
| **Quality** | ruff, mypy, Bandit, pre-commit, pytest (66 test files) |
| **Experiment tracking** | MLflow |
| **Data** | DuckDB, Yahoo Finance, Polygon.io, FMP, FRED, CCXT (Binance) |
| **GPU acceleration** | CuPy, Numba |
| **AI agents** | Kilo (6 specialized agents, 7 slash commands, 40 skills) |
| **Paper processing** | markitdown, paper2md, Docling, Marker, knowledge graph analysis |
| **Code intelligence** | GitNexus (28K+ symbols), graphify (7K+ nodes), qmd (hybrid docs search) |
| **Fixed income & options** | Nelson-Siegel, Vasicek/CIR, Black-Scholes, Heston, SABR, CDS pricing |
| **RL & advanced ML** | stable-baselines3 (PPO/SAC), CQL offline RL, survival analysis, structural break |
| **Wavelet & signal** | FFT, wavelet denoising, cycle detection, signal decomposition |

---

## Practical Research Uses

- Compare geometric chart patterns, indicator rules and optional ML filters through the strategy and backtest modules.
- Inspect feature drift, calibration, ablations and statistical diagnostics alongside favorable and unfavorable experiment results.
- Explore cross-asset features and portfolio/risk models with explicit data and cost assumptions.
- Use the component tests as evidence of the behaviors they assert. Test files, tooling and stored artifacts alone do not establish production readiness or generalization.

---

## Paper-Trading Research

The public source includes daily signal logging in `scripts/paper_trade_daily.py` and walk-forward paper-trading workflows. These generate hypothetical observations for further evaluation.

The May 2026 notes reported a Quality Registry experiment at Sharpe 2.00 OOS and proposed a 14-day live paper-trading protocol. This documentation review does not establish that the protocol completed or that its criteria certify a deployable strategy. Preserve those results as historical experiment evidence and evaluate future periods separately.

---

*Last updated: 2026-05-18*
*Project: investment_trying*
