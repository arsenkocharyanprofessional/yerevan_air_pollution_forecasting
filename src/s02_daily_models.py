"""Step 3. Daily one day ahead forecasts: baselines, ETS, SARIMA and SARIMAX.

Rolling origin, expanding window, from 1 Nov 2024 to the last day. Model orders are chosen
by AIC on the data before 1 Nov 2024 only. Metrics are reported for the winter test windows.
"""
from itertools import product
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.sarimax import SARIMAX

from config import (PROCESSED, RESULTS, TEST_WINTERS, FIRST_TEST_DAY, WEATHER_VARS, FOURIER_K, REFIT_EVERY,
                    WHO_DAILY_LIMIT)
from plot_style import apply_style, save, INK, RUST, BLUE, GREY, LIGHT, MODEL_COLORS
from ts_utils import (optimize_SARIMAX, recursive_forecast, fourier_terms, all_metrics, diebold_mariano, rmse)

apply_style()
daily = pd.read_csv(PROCESSED / "daily.csv", index_col="date", parse_dates=True)
y = daily["log_pm25"]
tune = daily.index < FIRST_TEST_DAY
fourier = fourier_terms(daily.index, FOURIER_K)


def weather_block(source: str) -> pd.DataFrame:
    """Standardised daily weather regressors from one source (obs, ecmwf1, gfs1, ecmwf2 ...).
    Means and standard deviations come from the tuning sample of the observed weather."""
    cols = [f"{source}_{v}" for v in WEATHER_VARS]
    ref = daily.loc[tune, [f"obs_{v}" for v in WEATHER_VARS]]
    block = (daily[cols] - ref.mean().values) / ref.std().values
    block.columns = WEATHER_VARS
    return block


def design(source=None) -> pd.DataFrame:
    return fourier if source is None else pd.concat([fourier, weather_block(source)], axis=1)


# ---------------------------------------------------------------- 1. order selection
# (a) class method: AIC grid over (p, q, P, Q) with weekly seasonal terms, on the tuning sample
p = q = range(0, 4)
P = Q = range(0, 2)
order_list = list(product(p, q, P, Q))
grid = optimize_SARIMAX(y[tune], design("obs")[tune], order_list, d=0, D=0, s=7)
grid.to_csv(RESULTS / "sarimax_aic_grid.csv", index=False)
print("AIC grid, best orders:\n", grid.head(5))
aic_best = grid.iloc[0, 0]
aic_fit = SARIMAX(y[tune], design("obs")[tune], order=(aic_best[0], 0, aic_best[1]), trend="c").fit(disp=False)
aic_info = {"aic_best_order": [int(v) for v in aic_best],
            "aic_best_min_ar_root": float(np.abs(aic_fit.arroots).min()) if aic_best[0] else None,
            "aic_best_min_ma_root": float(np.abs(aic_fit.maroots).min()) if aic_best[1] else None}

# (b) decision rule: AIC rewards fitting the whole year, which is mostly calm summer days. What matters
# here is one day ahead accuracy in winter, so the order is chosen by rolling origin validation on
# the only winter inside the tuning sample (Dec 2023 to Feb 2024), with and without a constant.
v0, v1 = daily.index.get_loc(pd.Timestamp("2023-12-01")), daily.index.get_loc(pd.Timestamp("2024-03-01"))
val_actual = daily["pm25"].iloc[v0:v1]
ok_val = val_actual.notna().values
val_rows = []
for trend in ["n", "c"]:
    for vp, vq in product(range(0, 3), range(0, 3)):
        if vp == 0 and vq == 0:
            continue
        try:
            vpred = np.exp(np.array(recursive_forecast(y, design("obs"), v0, v1 - v0, 1, "SARIMAX", order=(vp, 0, vq),
                                                       refit_every=14, trend=trend), dtype=float))
            val_rows.append({"trend": trend, "p": vp, "q": vq, "val_RMSE": rmse(val_actual[ok_val], vpred[ok_val])})
        except Exception:
            continue
val = pd.DataFrame(val_rows).sort_values("val_RMSE").reset_index(drop=True)
val_persistence = rmse(val_actual[ok_val], np.exp(y.ffill().shift(1)).iloc[v0:v1][ok_val])
val.to_csv(RESULTS / "order_validation_winter2023.csv", index=False)
print(val.head(6), "\npersistence on validation winter:", round(val_persistence, 2))
TREND = val.loc[0, "trend"]
ORDER, SEASONAL = (int(val.loc[0, "p"]), 0, int(val.loc[0, "q"])), (0, 0, 0, 0)
print("chosen order", ORDER, "trend", TREND)

best = SARIMAX(y[tune], design("obs")[tune], order=ORDER, seasonal_order=SEASONAL, trend=TREND).fit(disp=False)
with open(RESULTS / "sarimax_summary.txt", "w") as f:
    f.write(str(best.summary()))
fig = best.plot_diagnostics(figsize=(10, 7))
save(fig, "fig07_sarimax_diagnostics.png")
with open(RESULTS / "sarimax_summary.txt", "w") as f:
    f.write(str(best.summary()))
lb = acorr_ljungbox(best.resid.dropna(), np.arange(1, 11, 1))
lb.to_csv(RESULTS / "ljung_box.csv")
print(lb)

# ETS form chosen by AIC on the tuning sample (interpolated, ETS needs a complete series)
y_filled = y.interpolate(limit_direction="both")
ets_forms = {"ETS(A,N,N)": dict(trend=None, seasonal=None),
             "ETS(A,Ad,N)": dict(trend="add", damped_trend=True, seasonal=None),
             "ETS(A,N,A) weekly": dict(trend=None, seasonal="add", seasonal_periods=7)}
ets_aic = {name: ETSModel(y_filled[tune], error="add", **kw).fit(disp=False).aic for name, kw in ets_forms.items()}
ETS_NAME = min(ets_aic, key=ets_aic.get)
print("ETS AIC", ets_aic, "->", ETS_NAME)

# ---------------------------------------------------------------- 2. rolling forecasts
TRAIN_LEN = int(tune.sum())
HORIZON = len(daily) - TRAIN_LEN
WINDOW = 1
test_index = daily.index[TRAIN_LEN:]
pred = pd.DataFrame({"actual": daily["pm25"].iloc[TRAIN_LEN:]})

pred["Persistence"] = np.exp(recursive_forecast(y, None, TRAIN_LEN, HORIZON, WINDOW, "last"))
pred["Seasonal naive"] = np.exp(recursive_forecast(y, None, TRAIN_LEN, HORIZON, WINDOW, "seasonal_naive"))

ets_pred = []
for i in range(TRAIN_LEN, TRAIN_LEN + HORIZON):
    fit = ETSModel(y_filled[:i], error="add", **ets_forms[ETS_NAME]).fit(disp=False)
    ets_pred.append(fit.forecast(1).iloc[0])
pred["ETS"] = np.exp(ets_pred)


def run_sarimax(source, start=None, **kw):
    """Rolling SARIMAX with regressors from `source`; `start` trims the sample (forecast archives
    only begin in early 2024)."""
    X = design(source)
    first = X.dropna().index.min() if start is None else pd.Timestamp(start)
    X = X.loc[first:].interpolate(limit=3)
    yy = y.loc[first:]
    out = recursive_forecast(yy, X, TRAIN_LEN - daily.index.get_loc(first), HORIZON, WINDOW, "SARIMAX",
                             order=ORDER, seasonal_order=SEASONAL, refit_every=REFIT_EVERY, trend=TREND, **kw)
    return np.exp(np.array(out, dtype=float))


pred["SARIMA + Fourier"] = run_sarimax(None)
pred["SARIMAX, actual weather"] = run_sarimax("obs")
pred["SARIMAX, ECMWF forecast"] = run_sarimax("ecmwf1")
pred["SARIMAX, GFS forecast"] = run_sarimax("gfs1")

# plug in variant: trained on observed weather, fed with the forecast for the target day
X_obs, X_fc = design("obs"), design("ecmwf1").interpolate(limit=3)
plug, res, params = [], None, None
for n, i in enumerate(range(TRAIN_LEN, TRAIN_LEN + HORIZON)):
    if n % REFIT_EVERY == 0:
        res = SARIMAX(y[:i], X_obs[:i], order=ORDER, seasonal_order=SEASONAL, trend=TREND).fit(disp=False, start_params=params)
        params = res.params
    else:
        res = res.append(y[i - 1:i], exog=X_obs[i - 1:i])
    plug.append(res.get_forecast(1, exog=X_fc[i:i + 1]).predicted_mean.iloc[0])
pred["SARIMAX trained on actual, fed ECMWF"] = np.exp(plug)

# two day lead: forecast day i+1 from the origin of day i
first = design("ecmwf1").dropna().index.min()
X1 = design("ecmwf1").loc[first:].interpolate(limit=3)
X2 = design("ecmwf2").loc[first:].interpolate(limit=3)
lead2 = recursive_forecast(y.loc[first:], X1, TRAIN_LEN - daily.index.get_loc(first), HORIZON, WINDOW, "SARIMAX",
                           order=ORDER, seasonal_order=SEASONAL, refit_every=REFIT_EVERY, steps_ahead=2, exog_last=X2, trend=TREND)
pred["SARIMAX, ECMWF forecast, 2 day lead"] = pd.Series(np.exp(np.array(lead2, dtype=float)), index=test_index).shift(1)
pred["Persistence, 2 day lead"] = pred["Persistence"].shift(1)
pred.to_csv(RESULTS / "daily_predictions.csv")

# ---------------------------------------------------------------- 3. metrics
ONE_DAY = ["Persistence", "Seasonal naive", "ETS", "SARIMA + Fourier", "SARIMAX, GFS forecast",
           "SARIMAX, ECMWF forecast", "SARIMAX, actual weather", "SARIMAX trained on actual, fed ECMWF"]
rows = []
for winter, (a, b) in TEST_WINTERS.items():
    block = pred.loc[a:b].dropna(subset=["actual"])
    for model in pred.columns.drop("actual"):
        base = "Persistence, 2 day lead" if "2 day" in model else "Persistence"
        sub = block.dropna(subset=[model, base])
        m = all_metrics(sub["actual"].values, sub[model].values, WHO_DAILY_LIMIT)
        m["skill_vs_persistence_%"] = 100 * (1 - m["RMSE"] / all_metrics(sub["actual"].values, sub[base].values)["RMSE"])
        if model != base:
            h = 2 if "2 day" in model else 1   # 2 step forecasts have overlapping errors
            m["DM_stat"], m["DM_p"] = diebold_mariano(sub["actual"].values, sub[model].values, sub[base].values, h=h)
        rows.append({"winter": winter, "model": model, **m})
metrics = pd.DataFrame(rows)
metrics.to_csv(RESULTS / "daily_metrics.csv", index=False)
print(metrics.round(3).to_string())
with open(RESULTS / "daily_setup.json", "w") as f:
    json.dump({"order": [int(v) for v in ORDER], "seasonal_order": [int(v) for v in SEASONAL], "trend": TREND,
               "validation_persistence_rmse": val_persistence, "validation_best_rmse": float(val.loc[0, "val_RMSE"]),
               **aic_info, "ets": ETS_NAME,
               "ets_aic": ets_aic, "train_len_first_origin": TRAIN_LEN, "horizon_days": HORIZON,
               "params_actual_weather": {k: float(v) for k, v in best.params.items()},
               "pvalues_actual_weather": {k: float(v) for k, v in best.pvalues.items()}}, f, indent=2)

# ---------------------------------------------------------------- 4. figures
show = ["Persistence", "Seasonal naive", "ETS", "SARIMA + Fourier", "SARIMAX, GFS forecast",
        "SARIMAX, ECMWF forecast", "SARIMAX, actual weather"]
main_winters = ["2024/25", "2025/26"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
for ax, winter in zip(axes, main_winters):
    m = metrics[(metrics["winter"] == winter) & metrics["model"].isin(show)].set_index("model").loc[show]
    bars = ax.barh(range(len(show)), m["RMSE"], color=[MODEL_COLORS[s] for s in show], height=0.68)
    ax.axvline(m.loc["Persistence", "RMSE"], color=INK, ls="--", lw=1)
    for k, v in enumerate(m["RMSE"]):
        ax.text(v + 0.3, k, f"{v:.1f}", va="center", fontsize=9)
    ax.set_yticks(range(len(show)), show)
    ax.invert_yaxis()
    ax.set_xlabel("RMSE (µg/m³), one day ahead")
    ax.set_title(f"Winter {winter}")
    ax.grid(axis="y", visible=False)
fig.tight_layout()
save(fig, "fig08_rmse_by_model.png")

for winter in main_winters:
    a, b = TEST_WINTERS[winter]
    block = pred.loc[a:b]
    fig, ax = plt.subplots(figsize=(11, 3.9))
    ax.plot(block.index, block["actual"], color=INK, lw=2, label="Actual")
    ax.plot(block.index, block["Persistence"], color=GREY, lw=1, ls=":", label="Persistence")
    ax.plot(block.index, block["SARIMAX, ECMWF forecast"], color=BLUE, lw=1.5, label="SARIMAX, ECMWF forecast")
    ax.plot(block.index, block["SARIMAX, actual weather"], color=RUST, lw=1.2, ls="--", label="SARIMAX, actual weather")
    ax.axhline(WHO_DAILY_LIMIT, color=INK, lw=0.8, ls="--")
    ax.set_ylabel("Daily mean PM2.5 (µg/m³)")
    ax.set_title(f"Actual vs one day ahead forecasts, winter {winter}")
    ax.legend(ncol=4, fontsize=8, loc="upper right")
    save(fig, f"fig09_actual_vs_pred_{winter.replace('/', '_')}.png")

# where the predictability comes from: persistence -> own memory -> forecast weather -> actual weather
fig, ax = plt.subplots(figsize=(7.5, 3.8))
steps = ["Persistence", "SARIMA + Fourier", "SARIMAX, ECMWF forecast", "SARIMAX, actual weather"]
labels = ["Yesterday's\nvalue", "Own memory\n+ season", "+ weather\nforecast", "+ actual weather\n(upper bound)"]
width = 0.38
for k, winter in enumerate(main_winters):
    m = metrics[metrics["winter"] == winter].set_index("model").loc[steps, "RMSE"]
    xs = np.arange(len(steps)) + (k - 0.5) * width
    ax.bar(xs, m.values, width=width, color=[MODEL_COLORS[s] for s in steps], alpha=1 if k else 0.6,
           edgecolor="white")
    for x, v in zip(xs, m.values):
        ax.text(x, v + 0.2, f"{v:.1f}", ha="center", fontsize=8)
ax.set_xticks(range(len(steps)), labels)
ax.set_ylabel("RMSE (µg/m³)")
ax.set_title("Where predictability comes from (pale: 2024/25, solid: 2025/26)")
ax.grid(axis="x", visible=False)
save(fig, "fig10_predictability_sources.png")
