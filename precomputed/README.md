# precomputed/

Files produced by the runs reported in the paper, so that the results can be reproduced without rerunning the code. Notebooks 07, 08 and 09 read them by default (`SOURCE = "precomputed"`). New runs write to `outputs/` with the same structure and never overwrite these files.

## Cases

`<case>` is the name of the folder of a case:

| Case | Observations | Look-back | Likelihood weights |
|---|---|---|---|
| `synthetic_R0_1.5_noise_light`, `synthetic_R0_1.5_noise_mild`, `synthetic_R0_2.0_noise_light`, `synthetic_R0_2.0_noise_mild` | weekly infections (noisy) | all the history | yes |
| `real_London_lookback25` | weekly deaths, London | all the history (25 weeks) | yes (notebooks 07 and 08) |
| `real_London_lookback5` | weekly deaths, London | 5 weeks | no (notebook 09) |

The baseline does not depend on the look-back: its folder for the real case is `real_London`.

## Contents

- `pretrained_pinn/` – weights of the pretrained PINN (see its README).
- `posteriors/pinn/<case>/` – PINN results of the best run of the case, in the folder `best_WIS/` (best seed by the WIS of the fit, notebook 10), read by notebooks 07, 08 and 09 with `PINN_SEED = "best"`. A run of a single seed is in `seed_<seed>/` and is read with `PINN_SEED = <seed>`. `best_WIS/` and `seed_<seed>/` have the same files (if you run notebook 10 on new runs it writes `best_seed.json`, which points "best" to the best `seed_<seed>/` when there is no `best_WIS/`). In the list `<run>` is `best_WIS` or `seed_<seed>`:
  - `<run>/mcmc_distributions.pkl` – posterior samples of all the weeks, with Rt (notebook 03; used in notebook 07).
  - `<run>/forecast_h4/forecast_results.pickle` – forecasts 4 weeks ahead (notebook 04; used in notebooks 08 and 09): for every week the samples, the 50% and 90% intervals and the quantiles. `forecast_h10/` is the same for 10 weeks and is optional.
  - Synthetic scenarios only: `<run>/forecast_age<a>/forecast_results.pickle`, a = 0..15, the forecast of every age class (fit by age in notebook 08). The window of the last week T_LAST (t0 = 33 for R0 = 1.5, 23 for R0 = 2.0) is included in the forecasts.
  - Optional: `<run>/obs=<week>.pickle` (chain of every week) and `run_config.json` (settings of the run).
- `posteriors/mechanistic/<case>/` – results of the Mechanistic model converted to the PINN format (notebook 05; the model itself is run in its own repository):
  - `abc_distributions.pkl` – parameter posteriors (notebook 07). For the synthetic scenarios it also has `last_week`: the extra fit at the last week T_LAST (R0 samples, simulated trajectories and incidence by age class), used in notebooks 07 and 08.
  - `forecast_h4/forecast_results.pickle` – forecasts in the same format as those of the PINN (notebooks 08 and 09); `forecast_h10/` is optional.
- `baselines/<case>/forecast_results_baseline.pkl` – baseline forecasts (notebook 06 writes them to `outputs/baselines/`), in the same format as the forecasts of the models; used in notebooks 08 and 09. `<case>` is `real_London` or one of the four synthetic cases.
