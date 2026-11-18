"""Step 4. VAR and Granger causality as a multivariate sanity check (class VAR notebook).

The VAR only sees the past: yesterday's pollution and yesterday's weather. Comparing it with
SARIMAX (which is given the weather of the target day) shows whether the gain comes from
past weather or from knowing tomorrow's weather.
"""
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

from config import PROCESSED, RESULTS, TEST_WINTERS, FIRST_TEST_DAY, FOURIER_K, WHO_DAILY_LIMIT
from ts_utils import optimize_VAR, rolling_forecast_VAR, fourier_terms, all_metrics, diebold_mariano

daily = pd.read_csv(PROCESSED / "daily.csv", index_col="date", parse_dates=True)
tune = daily.index < FIRST_TEST_DAY

# remove the annual cycle from every variable with Fourier terms fitted on the tuning sample
F = sm.add_constant(fourier_terms(daily.index, FOURIER_K))
endog = pd.DataFrame(index=daily.index)
season = {}
for name, col in {"log_pm25": "log_pm25", "wind": "obs_wind", "temp": "obs_temp"}.items():
    ok = tune & daily[col].notna()
    fit = sm.OLS(daily.loc[ok, col], F[ok]).fit()
    season[name] = fit.predict(F)
    resid = daily[col] - season[name]
    endog[name] = resid / resid[tune].std()
scale_pm = (daily["log_pm25"] - season["log_pm25"])[tune].std()

out = {"adf": {}}
for name in endog:
    stat, pval = adfuller(endog.loc[tune, name].interpolate().dropna())[:2]
    out["adf"][name] = {"stat": float(stat), "p": float(pval)}
    print(f"{name}: ADF Statistic: {stat:.3f}, p-value: {pval:.4f}")

# Granger causality on the tuning sample (the test needs a complete series)
g = endog[tune].interpolate().dropna()
out["granger"] = {}
for cause, effect in [("wind", "log_pm25"), ("temp", "log_pm25"), ("log_pm25", "wind")]:
    res = grangercausalitytests(g[[effect, cause]], [1, 2, 3], verbose=False)
    out["granger"][f"{cause} -> {effect}"] = {str(lag): float(r[0]["ssr_ftest"][1]) for lag, r in res.items()}
    print(f"{cause} Granger causes {effect}? p values by lag:", out["granger"][f"{cause} -> {effect}"])

result_df = optimize_VAR(endog[tune], max_p=5)
print(result_df)
result_df.to_csv(RESULTS / "var_aic.csv", index=False)
P_BEST = int(result_df.iloc[0]["p"])
best_model_fit = VARMAX(endog[tune], order=(P_BEST, 0)).fit(disp=False, maxiter=200)
lb = acorr_ljungbox(best_model_fit.resid["log_pm25"].dropna(), lags=10, return_df=True)
out["var_order"] = P_BEST
out["ljung_box_min_p_pm"] = float(lb["lb_pvalue"].min())

TRAIN_LEN = int(tune.sum())
HORIZON = len(daily) - TRAIN_LEN
pred_std = rolling_forecast_VAR(endog, TRAIN_LEN, HORIZON, P_BEST, "log_pm25", refit_every=30)
idx = daily.index[TRAIN_LEN:]
pred_var = np.exp(np.array(pred_std) * scale_pm + season["log_pm25"].loc[idx].values)

pred = pd.read_csv(RESULTS / "daily_predictions.csv", index_col="date", parse_dates=True)
pred["VAR"] = pred_var
pred.to_csv(RESULTS / "daily_predictions.csv")
rows = []
for winter, (a, b) in TEST_WINTERS.items():
    sub = pred.loc[a:b].dropna(subset=["actual", "VAR", "Persistence"])
    m = all_metrics(sub["actual"].values, sub["VAR"].values, WHO_DAILY_LIMIT)
    m["skill_vs_persistence_%"] = 100 * (1 - m["RMSE"] / all_metrics(sub["actual"].values, sub["Persistence"].values)["RMSE"])
    m["DM_stat"], m["DM_p"] = diebold_mariano(sub["actual"].values, sub["VAR"].values, sub["Persistence"].values)
    rows.append({"winter": winter, "model": "VAR", **m})
metrics = pd.read_csv(RESULTS / "daily_metrics.csv")
metrics = pd.concat([metrics[metrics["model"] != "VAR"], pd.DataFrame(rows)], ignore_index=True)
metrics.to_csv(RESULTS / "daily_metrics.csv", index=False)
print(pd.DataFrame(rows).round(3).to_string())
with open(RESULTS / "var_granger.json", "w") as f:
    json.dump(out, f, indent=2)
