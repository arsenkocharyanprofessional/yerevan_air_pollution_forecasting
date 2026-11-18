"""Step 7. Short side questions: episode persistence, anomalies, weather adjusted trend, lead time."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import (PROCESSED, RESULTS, PM_FILE, TEST_WINTERS, WINTER_MONTHS, WHO_DAILY_LIMIT, WEATHER_VARS,
                    MIN_READINGS_PER_HOUR, MIN_HOURS_PER_DAY, UTC_OFFSET_HOURS, FOURIER_K)
from plot_style import apply_style, save, INK, RUST, BLUE, GREY, GOLD, TEAL
from ts_utils import fourier_terms

apply_style()
daily = pd.read_csv(PROCESSED / "daily.csv", index_col="date", parse_dates=True)
pred = pd.read_csv(RESULTS / "daily_predictions.csv", index_col="date", parse_dates=True)
metrics = pd.read_csv(RESULTS / "daily_metrics.csv")
winter = daily.index.month.isin(WINTER_MONTHS)
out = {}

# ------------------------------------------------------------ 1. episode persistence and clearing
above = (daily["pm25"] > WHO_DAILY_LIMIT).where(daily["pm25"].notna())
pair = pd.DataFrame({"today": above, "tomorrow": above.shift(-1)})[winter].dropna()
out["p_stay_above"] = float(pair.loc[pair["today"] == 1, "tomorrow"].mean())
out["p_stay_below"] = float(1 - pair.loc[pair["today"] == 0, "tomorrow"].mean())
runs, n = [], 0
for flag, is_winter in zip(above.values, winter):
    if flag == 1 and is_winter:
        n += 1
    else:
        if n:
            runs.append(n)
        n = 0
if n:
    runs.append(n)
runs = np.array(runs)
out["episodes"] = int(len(runs))
out["episode_median_days"], out["episode_mean_days"], out["episode_max_days"] = float(np.median(runs)), float(runs.mean()), int(runs.max())
out["share_days_in_long_episodes"] = float(runs[runs >= 7].sum() / runs.sum())

F = sm.add_constant(fourier_terms(daily.index, FOURIER_K))
ok = daily["log_pm25"].notna()
des = pd.Series(np.nan, index=daily.index)
des[ok] = sm.OLS(daily.loc[ok, "log_pm25"], F[ok]).fit().resid
w = pd.DataFrame({"x": des.shift(1), "y": des})[winter].dropna()
phi = float(np.polyfit(w["x"], w["y"], 1)[0])
out["winter_ar1"], out["half_life_days"] = phi, float(np.log(0.5) / np.log(phi))

change = daily["log_pm25"].diff()
wd = pd.DataFrame({"change": change, "wind": daily["obs_wind"], "level_yday": daily["pm25"].shift(1)})[winter].dropna()
dirty = wd[wd["level_yday"] > WHO_DAILY_LIMIT]
clearing = dirty[dirty["change"] <= np.log(0.6)]         # fell by 40 percent or more in one day
out["clearing_days"] = int(len(clearing))
out["clearing_share"] = float(len(clearing) / len(dirty))
out["wind_on_clearing_days"], out["wind_on_other_dirty_days"] = float(clearing["wind"].mean()), float(dirty.drop(clearing.index)["wind"].mean())
out["corr_change_wind"] = float(wd["change"].corr(wd["wind"]))

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
axes[0].hist(runs, bins=np.arange(0.5, runs.max() + 1.5, 1), color=RUST, edgecolor="white")
axes[0].set_xlabel("Length of an episode above 15 µg/m³ (days)")
axes[0].set_ylabel("Number of episodes")
axes[0].set_title(f"Pollution episodes: median {np.median(runs):.0f} days, longest {runs.max()}")
axes[1].scatter(wd["wind"], np.exp(wd["change"]) * 100 - 100, s=12, color=BLUE, alpha=0.55, edgecolor="none")
axes[1].axhline(0, color=INK, lw=0.8)
axes[1].set_xlabel("Daily mean wind speed (m/s)")
axes[1].set_ylabel("Change from the day before (%)")
axes[1].set_ylim(-90, 250)
axes[1].set_title("Windy days clear the air")
fig.tight_layout()
save(fig, "fig13_episodes.png")

# ------------------------------------------------------------ 2. anomalies: what weather and history cannot explain
model = "SARIMAX, actual weather"
resid = (np.log(pred["actual"]) - np.log(pred[model])).dropna()
sigma = 1.4826 * np.median(np.abs(resid - resid.median()))
z = (resid - resid.median()) / sigma
flag = z[np.abs(z) > 3]
out["anomaly_sigma_log"] = float(sigma)
out["anomalies"] = [{"date": str(d.date()), "z": round(float(v), 2), "actual": round(float(pred.loc[d, "actual"]), 1),
                     "expected": round(float(pred.loc[d, model]), 1)} for d, v in flag.items()]
out["anomalies_high"], out["anomalies_low"] = int((flag > 0).sum()), int((flag < 0).sum())
for year in (2024, 2025):
    d0, d1 = pd.Timestamp(f"{year}-12-31"), pd.Timestamp(f"{year + 1}-01-01")
    out[f"new_year_{year}_z"] = [round(float(z.get(d0, np.nan)), 2), round(float(z.get(d1, np.nan)), 2)]
fig, ax = plt.subplots(figsize=(11, 3.4))
ax.plot(z.index, z.values, color=GREY, lw=0.8)
ax.scatter(flag.index, flag.values, color=RUST, s=28, zorder=3, label=f"|z| > 3 ({len(flag)} days)")
for lim in (-3, 3):
    ax.axhline(lim, color=INK, ls="--", lw=0.8)
ax.set_ylabel("Standardised forecast error (log scale)")
ax.set_title("Days that weather and history do not explain")
ax.legend(fontsize=8)
save(fig, "fig14_anomalies.png")

# ------------------------------------------------------------ 3. weather adjusted winter trend
def season_label(idx):
    return np.where(idx.month >= 11, idx.year, idx.year - 1)

def winter_effects(series):
    d = pd.DataFrame({"y": np.log(series)}).join(daily[[f"obs_{v}" for v in WEATHER_VARS]])
    d = d[d.index.month.isin(WINTER_MONTHS)].dropna()
    d["season"] = season_label(d.index)
    d = d[d["season"].isin([2023, 2024, 2025])]
    raw = d.groupby("season")["y"].mean()
    dummies = pd.get_dummies(d["season"], prefix="s", drop_first=True).astype(float)
    Xw = (d[[f"obs_{v}" for v in WEATHER_VARS]] - d[[f"obs_{v}" for v in WEATHER_VARS]].mean())
    Xm = sm.add_constant(pd.concat([dummies, Xw, fourier_terms(d.index, 1)], axis=1))
    fit = sm.OLS(d["y"], Xm).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    res = {"raw_change_%": {f"{s}/{str(s + 1)[2:]}": float(100 * (np.exp(raw[s] - raw[2023]) - 1)) for s in (2024, 2025)},
           "adjusted_change_%": {f"{s}/{str(s + 1)[2:]}": float(100 * (np.exp(fit.params[f"s_{s}"]) - 1)) for s in (2024, 2025)},
           "adjusted_p": {f"{s}/{str(s + 1)[2:]}": float(fit.pvalues[f"s_{s}"]) for s in (2024, 2025)},
           "n": int(len(d)), "raw_mean_ug": {str(s): float(np.exp(raw[s])) for s in raw.index}}
    return res

out["trend_city"] = winter_effects(daily["pm25"])
# fixed panel check: only the sensors that already reported in the first winter
pm = pd.read_csv(PM_FILE)
pm["time"] = pd.to_datetime(pm["hour_utc"], utc=True).dt.tz_localize(None) + pd.Timedelta(hours=UTC_OFFSET_HOURS)
pm = pm[pm["n_readings_pm25"] >= MIN_READINGS_PER_HOUR]
first_seen = pm.groupby("sensor_id")["time"].min()
panel = [s for s in first_seen.index if first_seen[s] < pd.Timestamp("2023-12-25") and s not in (83835, 99632)]
ph = pm[pm["sensor_id"].isin(panel)].groupby("time")["pm25"].median()
pday = ph.groupby(ph.index.floor("D")).agg(["mean", "count"])
panel_daily = pday["mean"].where(pday["count"] >= MIN_HOURS_PER_DAY).reindex(daily.index)
out["trend_fixed_panel"] = winter_effects(panel_daily)
out["fixed_panel_sensors"] = [int(s) for s in panel]

# ------------------------------------------------------------ 4. one day vs two day lead
lead = metrics[metrics["model"].isin(["Persistence", "Persistence, 2 day lead", "SARIMAX, ECMWF forecast",
                                      "SARIMAX, ECMWF forecast, 2 day lead"])]
out["lead_rmse"] = {f"{r.winter} | {r.model}": round(float(r.RMSE), 2) for r in lead.itertuples()}
fig, ax = plt.subplots(figsize=(7, 3.6))
for k, winter in enumerate(["2024/25", "2025/26"]):
    m = lead[lead["winter"] == winter].set_index("model")["RMSE"]
    ax.plot([1, 2], [m["Persistence"], m["Persistence, 2 day lead"]], color=GREY, marker="o", ls="--" if k == 0 else "-",
            label=f"Persistence {winter}")
    ax.plot([1, 2], [m["SARIMAX, ECMWF forecast"], m["SARIMAX, ECMWF forecast, 2 day lead"]], color=BLUE, marker="o",
            ls="--" if k == 0 else "-", label=f"SARIMAX, ECMWF forecast {winter}")
ax.set_xticks([1, 2], ["1 day ahead", "2 days ahead"])
ax.set_xlim(0.7, 2.3)
ax.set_ylabel("RMSE (µg/m³)")
ax.set_title("The model's advantage grows with the lead time")
ax.legend(fontsize=8)
save(fig, "fig15_lead_time.png")

with open(RESULTS / "extras.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
