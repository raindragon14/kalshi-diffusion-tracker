# Kalshi Information Diffusion Tracker

System that measures how quickly new information is absorbed into Kalshi prediction market prices — built on Jev, TypeSafe AI's structured decision model live on OpenRouter, ready to use today.

## Overview

This project tracks the **information diffusion lag** — the time delay between when news breaks and when Kalshi market prices fully adjust. It uses:

- **Kalshi API** — Real-time market prices (public, no auth needed)
- **Firecrawl** — News article search and extraction
- **Jev (via OpenRouter)** — Structured scoring of news relevance, direction, and confidence
- **Exponential decay modeling** — Quantifies how fast price movements decay after news events
- **Calibration evaluation** — Brier scores and reliability diagrams to assess Jev's confidence calibration

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Kalshi API │────▶│  Ingestion  │────▶│  Time-Series│────▶│  Dashboard  │
│  (prices)   │     │  Service    │     │  Storage    │     │  (HTML/JS)  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                           │
                           ▼
                   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
                   │  Firecrawl  │────▶│    Jev      │────▶│  Diffusion  │
                   │  (news)     │     │  Scoring    │     │  Model      │
                   └─────────────┘     └─────────────┘     └─────────────┘
                                                          │
                                                          ▼
                                                ┌─────────────────┐
                                                │  Calibration    │
                                                │  (Brier Score,  │
                                                │   Reliability)  │
                                                └─────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Firecrawl API key (free tier at [firecrawl.dev](https://firecrawl.dev))
- OpenRouter API key (for Jev at [openrouter.ai](https://openrouter.ai))

### Installation

```bash
git clone https://github.com/yourusername/kalshi-diffusion-tracker
cd kalshi-diffusion-tracker
pip install -e ".[dev]"
cp .env.example .env
# Edit .env with your API keys
```

### Configuration

Edit `.env`:

```bash
# Required
FIRECRAWL_API_KEY=your_key
OPENROUTER_API_KEY=your_key

# Market to track (find tickers at Kalshi API)
KALSHI_MARKET_TICKER=FEDFUNDS

# Optional: Failover for Jev
VERCEL_AI_GATEWAY_URL=https://gateway.vercel.ai/v1/chat/completions
VERCEL_AI_GATEWAY_TOKEN=your_token
```

### Running the Pipeline

Each step is a separate script — run in order:

```bash
# 1. Start ingestion (runs continuously, polls prices & news)
python -m scripts.run_ingestion

# 2. Score accumulated articles (run periodically)
python -m scripts.run_scoring

# 3. Fit diffusion curves & evaluate calibration
python -m scripts.run_diffusion

# 4. Generate dashboard
python -m scripts.run_dashboard
```

Open `dashboard/index.html` in your browser to view results.

## Project Structure

```
kalshi-diffusion-tracker/
├── .env.example           # Configuration template
├── pyproject.toml         # Dependencies & tool config
├── src/
│   ├── config.py          # Settings management
│   ├── models/            # Pydantic data models
│   ├── ingestion/         # Kalshi & Firecrawl clients
│   ├── scoring/           # Jev client & prompts
│   ├── diffusion/         # Curve fitting & calibration
│   ├── storage/           # SQLite operations
│   └── dashboard/         # Static site generator
├── scripts/               # Runnable pipeline steps
└── dashboard/             # Generated static site (gitignored)
```

## Key Concepts

### Jev Scoring

Each article gets three scores from Jev:
- **Relevance (0–1)**: How directly the news relates to the market event
- **Direction (up/down/neutral)**: Expected impact on "Yes" probability
- **Confidence (0–1)**: Model's certainty in its assessment

### Diffusion Modeling

Price movement after news is modeled as exponential decay:
```
price_change(t) = A * exp(-t / τ) + C
```
- **A (amplitude)**: Initial impact magnitude
- **τ (timescale)**: Decay time constant
- **C (baseline)**: Long-term offset
- **Half-life**: τ × ln(2) — time for impact to halve

### Calibration Evaluation

- **Brier Score**: Mean squared error of probabilistic predictions
- **Reliability Diagram**: Binned confidence vs. realized accuracy
- **ECE/MCE**: Expected/Maximum Calibration Error

## Costs

| Component | Est. Cost |
|-----------|-----------|
| OpenRouter (Jev) | ~$0.01–0.05 per 1000 articles |
| Firecrawl | Free tier (500 pages/month) |
| Kalshi API | Free (public) |
| **Total** | **<< $5** for development |

## Extending

- Add new market tickers in `.env`
- Modify `FIRECRAWL_SEARCH_QUERIES` for different topics
- Experiment with diffusion models in `src/diffusion/fitter.py`
- Customize dashboard in `src/dashboard/generate.py`

## License

MIT