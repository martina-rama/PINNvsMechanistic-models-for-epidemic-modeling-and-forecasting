# PINN vs Mechanistic models for epidemic forecasting

Code and data for the paper comparing a Physics-Informed Neural Network (PINN) with a classical Mechanistic model (calibrated via ABC) for epidemic forecasting, under a deliberately minimal shared setting.

> Paper: _title, authors, journal and DOI to be added_

## The two models

- **PINN**: implemented in this repository (`src/pinnvsmech/`), with MCMC inference and rolling-window forecasting.
- **Mechanistic**: implemented in a separate repository, [_link to be added_](#). This repository contains only its results, converted to the same format as the PINN results (`precomputed/`), so that the two models can be compared.

## What is in this repository

| Folder | Content |
|---|---|
| `data/` | Input data: synthetic scenarios, PINN training set, real epidemiological data (see `data/README.md`) |
| `src/pinnvsmech/` | Python package, one subpackage per step of the pipeline (see `src/pinnvsmech/README.md`) |
| `notebooks/` | The pipeline, one notebook per step, numbered in order (see `notebooks/README.md`) |
| `precomputed/` | Ready-to-use files: pretrained PINN, baselines, posteriors and forecasts of both models (see `precomputed/README.md`) |
| `outputs/` | Where new runs write their results (ignored by git) |
| `figures/`, `tables/` | Figures and tables written by the notebooks, by type (`parameters`, `forecasts`) and case |

## Reproducing the results

**1. Without rerunning anything.** `precomputed/` contains the pretrained network, the baselines, and the posteriors and forecasts of the PINN (best seed) and of the Mechanistic model used in the paper. Notebooks 07, 08 and 09 load them directly and regenerate the figures and the tables (`SOURCE = "precomputed"`, the default).

**2. From scratch (PINN).** Run the notebooks in order: 01 synthetic data, 02 PINN training, 03 parameter estimation (one or 100 seeds), 04 forecasts, 06 baseline, 10 best seed; then 07–09 with `SOURCE = "outputs"`. New results go to `outputs/`, so the files in `precomputed/` are never overwritten. The Mechanistic results can only be regenerated with the Mechanistic repository; notebook 05 converts them to the format used here.

## Installation

```bash
git clone <repo-url>
cd PINNvsMech_code
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Citation

_To be added._

## License

_To be added._
