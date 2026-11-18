"""Step 2. Exploratory time series analysis: plots, seasonality, stationarity, ACF and PACF."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
import statsmodels.api as sm

from config import PROCESSED, RESULTS, TEST_WINTERS, WHO_DAILY_LIMIT, WINTER_MONTHS, PM_FILE, EXCLUDED_SENSORS
from plot_style import apply_style, save, INK, RUST, BLUE, GREY, GOLD, LIGHT, TEAL
from ts_utils import fourier_terms

apply_style()
daily = pd.read_csv(PROCESSED / "daily.csv", index_col="date", parse_dates=True)
hourly = pd.read_csv(PROCESSED / "city_hourly.csv", index_col="time", parse_dates=True)
winter = daily.index.month.isin(WINTER_MONTHS)
stats = {}

# ------------------------------------------------------------ basic facts
valid = daily["pm25"].notna()
stats["hours_total"] = int(len(hourly))
stats["hours_with_data"] = int(hourly["pm25"].notna().sum())
stats["days_total"] = int(len(daily))
stats["days_valid"] = int(valid.sum())
stats["winter_days_valid"] = int((valid & winter).sum())
stats["winter_share_above_who"] = float((daily.loc[valid & winter, "pm25"] > WHO_DAILY_LIMIT).mean())
stats["first_day"], stats["last_day"] = str(daily.index.min().date()), str(daily.index.max().date())
stats["median_winter"] = float(daily.loc[winter, "pm25"].median())
stats["median_non_winter"] = float(daily.loc[~winter, "pm25"].median())
gaps = daily["pm25"].isna().astype(int).groupby(daily["pm25"].notna().cumsum()).sum()
stats["longest_gap_days"] = int(gaps.max())
monthly = daily["pm25"].groupby([daily.index.year, daily.index.month]).median().unstack(0)
monthly.round(1).to_csv(RESULTS / "monthly_medians.csv")
stats["median_december"] = float(daily.loc[daily.index.month == 12, "pm25"].median())
stats["median_may"] = float(daily.loc[daily.index.month == 5, "pm25"].median())

# ------------------------------------------------------------ fig 1: the series
fig, ax = plt.subplots(figsize=(11, 3.8))
for name, (a, b) in TEST_WINTERS.items():
    ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=LIGHT, zorder=0)
    label = "live check" if "live" in name else f"test {name}"
    ax.text(pd.Timestamp(a), daily["pm25"].max() * 0.97, f" {label}", fontsize=8, color=GREY, va="top",
            ha="right" if "live" in name else "left")
ax.plot(daily.index, daily["pm25"], color=RUST, lw=1.1)
ax.axhline(WHO_DAILY_LIMIT, color=INK, ls="--", lw=1)
ax.text(pd.Timestamp("2025-05-01"), WHO_DAILY_LIMIT + 2, "WHO 24 hour guideline, 15", fontsize=8, color=INK)
ax.set_ylabel("Daily mean PM2.5 (µg/m³)")
ax.set_title("Daily PM2.5 in Yerevan: quiet summers, heavy and noisy winters")
save(fig, "fig01_daily_series.png")

# ------------------------------------------------------------ fig 2: seasonality
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1.3, 1, 1]})
data = [daily.loc[daily.index.month == m, "pm25"].dropna() for m in range(1, 13)]
bp = axes[0].boxplot(data, patch_artist=True, showfliers=False, widths=0.65,
                     medianprops=dict(color=INK), whiskerprops=dict(color=GREY), capprops=dict(color=GREY))
for m, box in enumerate(bp["boxes"], start=1):
    box.set(facecolor=RUST if m in WINTER_MONTHS else "#f0c9bf", edgecolor="white")
axes[0].axhline(WHO_DAILY_LIMIT, color=INK, ls="--", lw=1)
axes[0].set_xticks(range(1, 13), list("JFMAMJJASOND"))
axes[0].set_ylabel("Daily mean PM2.5 (µg/m³)")
axes[0].set_title("Annual cycle")
h = hourly.copy()
h["hour"] = h.index.hour
hw = h.index.month.isin(WINTER_MONTHS)
axes[1].plot(h[hw].groupby("hour")["pm25"].median(), color=RUST, label="winter (Nov to Feb)")
axes[1].plot(h[~hw].groupby("hour")["pm25"].median(), color=GREY, label="rest of the year")
axes[1].set_xticks([0, 6, 12, 18, 23])
axes[1].set_xlabel("Hour of day (local)")
axes[1].set_title("Daily cycle")
axes[1].legend(fontsize=8)
dw = daily.loc[winter, "pm25"].groupby(daily.index[winter].dayofweek).median()
axes[2].bar(range(7), dw.values, color=RUST, width=0.65)
axes[2].set_xticks(range(7), ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"])
axes[2].set_title("Weekly cycle (winter): not significant")
save(fig, "fig02_seasonality.png")
stats["winter_peak_hour"] = int(h[hw].groupby("hour")["pm25"].median().idxmax())
stats["weekday_medians_winter"] = [float(v) for v in dw.values]

# is the weekly pattern real? day of week dummies, controlling for weather, annual cycle and winter
wd = daily[winter].dropna(subset=["log_pm25"] + [f"obs_{v}" for v in ["wind", "temp", "rh", "precip", "pressure"]])
dow = pd.get_dummies(wd.index.dayofweek, prefix="d", drop_first=True).astype(float).set_axis(wd.index)
season_id = pd.Series(np.where(wd.index.month >= 11, wd.index.year, wd.index.year - 1), index=wd.index)
ctrl = pd.concat([wd[[f"obs_{v}" for v in ["wind", "temp", "rh", "precip", "pressure"]]], fourier_terms(wd.index, 2),
                  pd.get_dummies(season_id, prefix="s", drop_first=True).astype(float)], axis=1)
fit = sm.OLS(wd["log_pm25"], sm.add_constant(pd.concat([dow, ctrl], axis=1))).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
stats["weekday_joint_p"] = float(fit.f_test(", ".join(f"{c} = 0" for c in dow.columns)).pvalue)
weekend = pd.Series((wd.index.dayofweek >= 5).astype(float), index=wd.index, name="weekend")
fit = sm.OLS(wd["log_pm25"], sm.add_constant(pd.concat([weekend, ctrl], axis=1))).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
stats["weekend_effect_pct"], stats["weekend_effect_p"] = float(100 * (np.exp(fit.params["weekend"]) - 1)), float(fit.pvalues["weekend"])

# ------------------------------------------------------------ fig 3: decomposition
# STL with a 365 day period needs many years; with about three cycles its seasonal part absorbs
# the noise. A short series is better split as: trend = centred 365 day moving average,
# seasonal = annual Fourier terms (K = 3) fitted to the detrended series, remainder = the rest.
logy = daily["log_pm25"].interpolate(limit_direction="both")
trend = logy.rolling(365, center=True, min_periods=365).mean()
detr = logy - trend
F3 = sm.add_constant(fourier_terms(daily.index, 3))
ok3 = detr.notna()
seasonal = pd.Series(sm.OLS(detr[ok3], F3[ok3]).fit().predict(F3), index=daily.index)
remainder = logy - trend - seasonal
fig, axes = plt.subplots(4, 1, figsize=(11, 6.5), sharex=True)
for ax, series, label, color in zip(axes, [logy, trend, seasonal, remainder],
                                    ["log PM2.5", "Trend (365 d MA)", "Annual season", "Remainder"],
                                    [INK, BLUE, RUST, GREY]):
    ax.plot(series, color=color, lw=1)
    ax.set_ylabel(label, fontsize=9)
axes[0].set_title("Decomposition of the log daily series: almost all structure is the annual cycle")
save(fig, "fig03_decomposition.png")
r = remainder[trend.notna()]
stats["seasonal_strength"] = float(max(0, 1 - r.var() / (r + seasonal[trend.notna()]).var()))
stats["trend_strength"] = float(max(0, 1 - r.var() / (r + trend[trend.notna()]).var()))
stats["trend_range_log"] = [float(trend.min()), float(trend.max())]

# ------------------------------------------------------------ stationarity
F = sm.add_constant(fourier_terms(daily.index, 2))
ok = daily["log_pm25"].notna()
deseason = pd.Series(np.nan, index=daily.index)
deseason[ok] = sm.OLS(daily.loc[ok, "log_pm25"], F[ok]).fit().resid
tests = {}
for name, series in {"log PM2.5": daily["log_pm25"], "log PM2.5 minus Fourier seasonality": deseason,
                     "first difference of log PM2.5": daily["log_pm25"].diff()}.items():
    s = series.interpolate().dropna()
    tests[name] = {"ADF_stat": float(adfuller(s)[0]), "ADF_p": float(adfuller(s)[1]),
                   "KPSS_stat": float(kpss(s, nlags="auto")[0]), "KPSS_p": float(kpss(s, nlags="auto")[1])}
stats["stationarity"] = tests
acf_vals = sm.tsa.acf(logy, nlags=8)
stats["acf_log_lag1"], stats["acf_log_lag7"] = float(acf_vals[1]), float(acf_vals[7])

# ------------------------------------------------------------ fig 4: ACF and PACF
fig, axes = plt.subplots(2, 2, figsize=(11, 5.6))
plot_acf(logy, lags=40, ax=axes[0, 0], color=RUST, vlines_kwargs={"colors": RUST}, title="ACF, log PM2.5")
plot_pacf(logy, lags=40, ax=axes[0, 1], color=RUST, vlines_kwargs={"colors": RUST}, title="PACF, log PM2.5", method="ywm")
d = deseason.interpolate().dropna()
plot_acf(d, lags=40, ax=axes[1, 0], color=BLUE, vlines_kwargs={"colors": BLUE}, title="ACF after removing annual Fourier seasonality")
plot_pacf(d, lags=40, ax=axes[1, 1], color=BLUE, vlines_kwargs={"colors": BLUE}, title="PACF after removing annual Fourier seasonality", method="ywm")
for ax in axes.ravel():
    ax.set_xlabel("Lag (days)")
fig.tight_layout()
save(fig, "fig04_acf_pacf.png")

# ------------------------------------------------------------ fig 5: weather link
w = daily[winter & valid]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
for ax, col, label in zip(axes[:2], ["obs_wind", "obs_blh"], ["Daily mean wind speed (m/s)", "Boundary layer height (m)"]):
    sub = w[[col, "pm25"]].dropna()
    ax.scatter(sub[col], sub["pm25"], s=12, color=RUST, alpha=0.55, edgecolor="none")
    ax.set_yscale("log")
    ax.set_xlabel(label)
    r = np.corrcoef(sub[col], np.log(sub["pm25"]))[0, 1]
    stats[f"winter_corr_logpm_{col}"] = float(r)
    ax.set_title(f"Winter days, r = {r:.2f} with log PM2.5")
axes[0].set_ylabel("Daily mean PM2.5 (µg/m³, log scale)")
sub = w[["obs_wind", "ecmwf1_wind", "gfs1_wind"]].dropna()
axes[2].scatter(sub["obs_wind"], sub["ecmwf1_wind"], s=12, color=BLUE, alpha=0.6, edgecolor="none", label="ECMWF day ahead")
lim = [0, max(sub.max()) * 1.05]
axes[2].plot(lim, lim, color=GREY, ls="--", lw=1)
axes[2].set_xlabel("Wind in the weather archive (m/s)")
axes[2].set_ylabel("Wind forecast one day earlier (m/s)")
stats["winter_corr_wind_obs_ecmwf1"] = float(sub["obs_wind"].corr(sub["ecmwf1_wind"]))
stats["winter_corr_wind_obs_gfs1"] = float(sub["obs_wind"].corr(sub["gfs1_wind"]))
axes[2].set_title(f"Forecast wind vs archive wind, r = {stats['winter_corr_wind_obs_ecmwf1']:.2f}")
fig.tight_layout()
save(fig, "fig05_weather_link.png")
for col in ["obs_temp", "obs_rh", "obs_precip", "obs_pressure"]:
    sub = w[[col, "log_pm25"]].dropna()
    stats[f"winter_corr_logpm_{col}"] = float(sub[col].corr(sub["log_pm25"]))

# ------------------------------------------------------------ fig 6: sensor network
pm = pd.read_csv(PM_FILE)
pm = pm[~pm["sensor_id"].isin(EXCLUDED_SENSORS)]
pm["month"] = pd.to_datetime(pm["hour_utc"], utc=True).dt.tz_localize(None).dt.to_period("M")
n_sensors = pm.groupby("month")["sensor_id"].nunique()
fig, ax = plt.subplots(figsize=(11, 2.6))
ax.bar(n_sensors.index.to_timestamp(), n_sensors.values, width=25, color=TEAL)
ax.set_ylabel("Active sensors")
ax.set_title("The citizen sensor network grew from 1 to 8 sensors")
save(fig, "fig06_sensor_network.png")
stats["sensors_first_month"], stats["sensors_last_month"] = int(n_sensors.iloc[0]), int(n_sensors.iloc[-1])
stats["raw_rows"] = int(len(pd.read_csv(PM_FILE)))

with open(RESULTS / "eda_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print(json.dumps(stats, indent=2))
