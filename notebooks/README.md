# notebooks/

Numbered by theme, in the order of the pipeline. Every notebook (except 01, 02, 05 and 06, which generate files) is for the four synthetic scenarios and the real case, chosen with `CASE`, `R0_TRUE` and `NOISE` in the first cell.

1. `01_synthetic_data.ipynb` – synthetic scenarios
2. `02_pinn_training.ipynb` – generates (or loads) the training and collocation sets and trains the PINN. No plots
3. `03_parameter_estimation.ipynb` – PINN + MCMC rolling-window estimation of R0, Delta and the onset. One notebook for the four synthetic scenarios and the real case (London); 1 or 100 independent runs (seeds). No plots
4. `04_forecast.ipynb` – PINN probabilistic forecasts (4 or 10 weeks ahead) from the posteriors of notebook 03. Same cases and seeds as notebook 03; optional forecast of every age class for the synthetic cases. No plots
5. `05_mechanistic_results_format.ipynb` – raw results of the Mechanistic model (calibrated in its own repository) converted to the PINN format: parameter posteriors and forecasts. One notebook for the four synthetic scenarios and the real case. No plots
6. `06_baseline_forecast.ipynb` – baseline forecasts (symmetrised random walk on the last observation, 10,000 trajectories) for the four synthetic scenarios and the real case, with the same rolling windows as the models. The synthetic baseline uses the noisy series. Writes `outputs/baselines/<case>/forecast_results_baseline.pkl`. No plots
7. `07_compare_parameters.ipynb` – split violins of the parameter estimates, PINN vs Mechanistic, week by week, and (synthetic scenarios) density of R0 at the last week. One notebook for the four synthetic scenarios (R0 or Rt) and the real case (Rt only)
8. `08_compare_forecasts.ipynb` – forecasts and fit, PINN vs Mechanistic (optionally the baseline): grid of the rolling windows, forecasts at fixed horizon and, for the synthetic scenarios, retrospective fit and fit by age class at the last week T_LAST (33 for R0 = 1.5, 23 for R0 = 2.0). One notebook for the four synthetic scenarios and the real case
9. `09_forecast_metrics.ipynb` – quantitative comparison of the forecasts (MAE, RMSE, WIS, IS_80, wMAPE, coverage of the 50% and 90% intervals) per horizon and pooled, absolute and relative to the baseline. Needs the baseline of notebook 06. Tables in `tables/forecasts/<case>/`. One notebook for the four synthetic scenarios and the real case
10. `10_best_seed.ipynb` – best PINN run among the seeds: score (WIS or wMAPE) of the fit of every seed, mean over the rolling windows, weighted as the likelihood of the case. Writes `best_seed.json` in `posteriors/pinn/<case>/`. Notebooks 07, 08, 09 read the best run with `PINN_SEED = "best"`: the folder `posteriors/pinn/<case>/best_WIS/` if it exists (shipped in `precomputed/`), otherwise the seed of `best_seed.json`; with `PINN_SEED = 10` they read `seed_010/`. Run after notebook 04, before 07–09 (the shipped `best_seed.json` makes it unnecessary to rerun it)

Each notebook reads from `precomputed/` by default, with a switch to use `outputs/` after a new run.
