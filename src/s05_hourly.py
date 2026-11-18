"""Step 6. Secondary task: hourly PM2.5 for the next day (24 hours ahead), winter only.

Forecast origin is local midnight. Every hour of the coming day is predicted with information
available before that midnight plus the ECMWF weather forecast issued one day earlier.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from config import PROCESSED, RESULTS, TEST_WINTERS, SEED
from plot_style import apply_style, save, INK, RUST, BLUE, GREY, TEAL
from ts_utils import rmse, mae, mape

apply_style()
h = pd.read_csv(PROCESSED / "city_hourly.csv", index_col="time", parse_dates=True)
daily_pred = pd.read_csv(RESULTS / "daily_predictions.csv", index_col="date", parse_dates=True)

day = h.index.floor("D")
df = pd.DataFrame(index=h.index)
df["actual"] = h["pm25"]
df["Same hour yesterday"] = h["pm25"].shift(24)
df["Same hour last week"] = h["pm25"].shift(168)

# daily forecast times the typical shape of the day (median hour / day ratio over the previous 28 days)
day_mean = h["pm25"].groupby(day).transform("mean")
ratio = (h["pm25"] / day_mean).to_frame("r")
ratio["hour"], ratio["day"] = h.index.hour, day
shape = ratio.pivot_table(index="day", columns="hour", values="r").rolling(28, min_periods=10).median().shift(1)
shape_long = shape.stack().rename("shape")
shape_long.index = shape_long.index.get_level_values(0) + pd.to_timedelta(shape_long.index.get_level_values(1), unit="h")
df["Daily SARIMAX x daily shape"] = daily_pred["SARIMAX, ECMWF forecast"].reindex(day).values * shape_long.reindex(h.index).values

# gradient boosting on features known at midnight
logy = np.log(h["pm25"])
X = pd.DataFrame(index=h.index)
X["hour_sin"], X["hour_cos"] = np.sin(2 * np.pi * h.index.hour / 24), np.cos(2 * np.pi * h.index.hour / 24)
X["doy_sin"], X["doy_cos"] = np.sin(2 * np.pi * h.index.dayofyear / 365.25), np.cos(2 * np.pi * h.index.dayofyear / 365.25)
X["weekend"] = (h.index.dayofweek >= 5).astype(int)
X["same_hour_yday"] = logy.shift(24)
X["same_hour_2days"] = logy.shift(48)
yday_mean = logy.groupby(day).mean().shift(1)
X["yday_mean"] = yday_mean.reindex(day).values
late = logy[h.index.hour >= 21].groupby(day[h.index.hour >= 21]).mean().shift(1)
X["yday_evening"] = late.reindex(day).values
for v in ["wind_speed_10m", "temperature_2m", "relative_humidity_2m", "surface_pressure"]:
    X[f"fc_{v}"] = h[f"fc_ecmwf_{v}_d1"]
    X[f"fc_day_{v}"] = h[f"fc_ecmwf_{v}_d1"].groupby(day).transform("mean")

df["Gradient boosting"] = np.nan
have_fc = X["fc_wind_speed_10m"].notna() & logy.notna()
for winter, (a, b) in TEST_WINTERS.items():
    train = have_fc & (h.index < a)
    test = (h.index >= a) & (h.index < pd.Timestamp(b) + pd.Timedelta(days=1))
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=4, random_state=SEED)
    model.fit(X[train], logy[train])
    df.loc[test, "Gradient boosting"] = np.exp(model.predict(X[test]))

models = ["Same hour yesterday", "Same hour last week", "Daily SARIMAX x daily shape", "Gradient boosting"]
rows = []
for winter, (a, b) in TEST_WINTERS.items():
    block = df.loc[a: pd.Timestamp(b) + pd.Timedelta(hours=23)].dropna(subset=["actual"] + models)
    for m in models:
        rows.append({"winter": winter, "model": m, "n": len(block), "RMSE": rmse(block["actual"], block[m]),
                     "MAE": mae(block["actual"], block[m]), "MAPE": float(mape(block["actual"], block[m]))})
metrics = pd.DataFrame(rows)
metrics.to_csv(RESULTS / "hourly_metrics.csv", index=False)
print(metrics.round(2).to_string())

colors = {"Same hour yesterday": GREY, "Same hour last week": "#b8c0cc", "Daily SARIMAX x daily shape": BLUE,
          "Gradient boosting": TEAL}
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1, 1.5]})
piv = metrics[metrics["winter"].isin(["2024/25", "2025/26"])].pivot(index="model", columns="winter", values="RMSE").loc[models]
xs = np.arange(len(models))
for k, winter in enumerate(piv.columns):
    axes[0].bar(xs + (k - 0.5) * 0.38, piv[winter], width=0.38, color=[colors[m] for m in models],
                alpha=0.6 if k == 0 else 1, edgecolor="white")
    for x, v in zip(xs + (k - 0.5) * 0.38, piv[winter]):
        axes[0].text(x, v + 0.3, f"{v:.1f}", ha="center", fontsize=8)
axes[0].set_xticks(xs, ["Same hour\nyesterday", "Same hour\nlast week", "Daily model\nx day shape", "Gradient\nboosting"], fontsize=8)
axes[0].set_ylabel("RMSE (µg/m³)")
axes[0].set_title("Hourly, 24 h ahead (pale 2024/25, solid 2025/26)")
axes[0].grid(axis="x", visible=False)
week = df.loc["2026-01-12":"2026-01-18"]
axes[1].plot(week.index, week["actual"], color=INK, lw=2, label="Actual")
axes[1].plot(week.index, week["Same hour yesterday"], color=GREY, ls=":", lw=1, label="Same hour yesterday")
axes[1].plot(week.index, week["Daily SARIMAX x daily shape"], color=BLUE, lw=1.3, label="Daily model x day shape")
axes[1].plot(week.index, week["Gradient boosting"], color=TEAL, lw=1.3, label="Gradient boosting")
axes[1].set_title("One week of hourly forecasts, 12 to 18 January 2026")
axes[1].legend(fontsize=8, ncol=2)
axes[1].tick_params(axis="x", labelsize=8)
fig.tight_layout()
save(fig, "fig12_hourly.png")
