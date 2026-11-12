# pinnvsmech

One subpackage per step of the pipeline.

| Subpackage | Content |
|---|---|
| `ode/` | Compartmental models (age-structured SEIRDDobs, reduced system without deaths) |
| `data/` | Synthetic data generation; PINN training-set generation and loading |
| `pinn/` | Architecture, losses, training |
| `inference/` | PINN + MCMC parameter estimation: `cases.py` (observations and settings of the five cases: four synthetic scenarios and London), `mcmc.py` (rolling-window MCMC, posterior samples, Rt) |
| `forecasting/` | PINN rolling-window forecasts: `forecast.py` (posterior samples propagated through the PINN), `summaries.py` (quantiles and intervals of the forecasts), `baseline.py` (random-walk baseline) and `windows.py` (rolling windows scored: T_LAST and trimming of the synthetic scenarios, evaluation period of the real case) |
| `mechanistic/` | `convert.py`: conversion of the raw Mechanistic results (parameter posteriors and forecasts) to the PINN format. The Mechanistic model itself lives in its own repository |
| `evaluation/` | `forecast_validation.py` (scores of one forecast: interval score, WIS, coverage, wMAPE), `metrics.py` (tables per horizon and pooled, normalisation by the baseline) and `best_seed.py` (best PINN run by the score of the fit) |
| `plotting/` | Plots of the comparison PINN vs Mechanistic. `params.py`: comparison of the parameter estimates of the two models; `forecasts.py`: comparison of the forecasts and of the fit (grid, fixed horizons, retrospective fit, fit by age class) |

## Modules at the top level

- `paths.py` – locations of the data, `precomputed/` and `outputs/` folders.
- `constants.py` – population by age class, IFR, epsilon, xi, parameter ranges.
- `utils_age_structured.py` – PINN loss (`compute_loss_age`, `F_age`, `df`) and legacy helpers for estimation, forecasting and metrics. Kept as it is, the other modules import from it.

## Training

- `data/pinn_training_data.py` – parameter grid, collocation set, file names.
- `pinn/training.py` – `train_pinn`: training loop (Adamax, best-train-loss checkpoint, CSV log of the losses, runtime in JSON).
