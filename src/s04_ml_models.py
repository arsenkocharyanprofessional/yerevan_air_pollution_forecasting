"""Step 5. Machine learning models: random forest, gradient boosting and SVR (scikit-learn).

Same task and same information as SARIMAX with the ECMWF forecast: lags of pollution, the
calendar, yesterday's weather and the weather forecast for the target day.
Hyperparameters are tuned with a time series split inside the data before 1 Nov 2024.
Models are re-trained at the start of every month on all data available by then.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from config import PROCESSED, RESULTS, TEST_WINTERS, FIRST_TEST_DAY, WEATHER_VARS, FOURIER_K, WHO_DAILY_LIMIT, SEED
from plot_style import apply_style, save, MODEL_COLORS, INK
from ts_utils import fourier_terms, all_metrics, diebold_mariano

apply_style()
daily = pd.read_csv(PROCESSED / "daily.csv", index_col="date", parse_dates=True)
y = daily["log_pm25"]
y_known = y.ffill()                       # last available value when a day is missing

X = fourier_terms(daily.index, FOURIER_K)
for lag in (1, 2, 3, 7):
    X[f"lag{lag}"] = y_known.shift(lag)
X["roll7"] = y_known.shift(1).rolling(7).mean()
X["weekend"] = (daily.index.dayofweek >= 5).astype(int)
for v in WEATHER_VARS:
    X[f"fc_{v}"] = daily[f"ecmwf1_{v}"].interpolate(limit=3)      # forecast for the target day
    X[f"yday_{v}"] = daily[f"obs_{v}"].shift(1)                    # observed yesterday
data = X.join(y.rename("target")).dropna()
tune = data.index < FIRST_TEST_DAY
features = list(X.columns)

candidates = {
    "Random forest": (RandomForestRegressor(random_state=SEED, n_jobs=-1),
                      {"n_estimators": [300], "max_depth": [4, 8, None], "min_samples_leaf": [3, 10]}),
    "Gradient boosting": (GradientBoostingRegressor(random_state=SEED),
                          {"n_estimators": [100, 300], "learning_rate": [0.03, 0.1], "max_depth": [2, 3],
                           "subsample": [0.8]}),
    "SVR": (make_pipeline(StandardScaler(), SVR()),
            {"svr__C": [0.3, 1, 3, 10], "svr__epsilon": [0.05, 0.1, 0.2], "svr__gamma": ["scale", 0.01]}),
}
cv = TimeSeriesSplit(n_splits=4)
best_params, models = {}, {}
for name, (est, grid) in candidates.items():
    search = GridSearchCV(est, grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
    search.fit(data.loc[tune, features], data.loc[tune, "target"])
    best_params[name] = search.best_params_
    models[name] = search.best_estimator_
    print(name, search.best_params_, f"CV RMSE (log scale) {-search.best_score_:.3f}")

test = data[~tune]
ml_pred = pd.DataFrame(index=test.index, columns=list(models), dtype=float)
for month_start in pd.date_range(FIRST_TEST_DAY, test.index.max(), freq="MS"):
    month = test.loc[month_start: month_start + pd.offsets.MonthEnd(0)]
    if month.empty:
        continue
    train = data[data.index < month_start]
    for name, est in models.items():
        est.fit(train[features], train["target"])
        ml_pred.loc[month.index, name] = np.exp(est.predict(month[features]))

pred = pd.read_csv(RESULTS / "daily_predictions.csv", index_col="date", parse_dates=True)
for name in models:
    pred[name] = ml_pred[name].reindex(pred.index)
pred.to_csv(RESULTS / "daily_predictions.csv")

rows = []
for winter, (a, b) in TEST_WINTERS.items():
    for name in models:
        sub = pred.loc[a:b].dropna(subset=["actual", name, "Persistence"])
        m = all_metrics(sub["actual"].values, sub[name].values, WHO_DAILY_LIMIT)
        m["skill_vs_persistence_%"] = 100 * (1 - m["RMSE"] / all_metrics(sub["actual"].values, sub["Persistence"].values)["RMSE"])
        m["DM_stat"], m["DM_p"] = diebold_mariano(sub["actual"].values, sub[name].values, sub["Persistence"].values)
        rows.append({"winter": winter, "model": name, **m})
metrics = pd.read_csv(RESULTS / "daily_metrics.csv")
metrics = pd.concat([metrics[~metrics["model"].isin(models)], pd.DataFrame(rows)], ignore_index=True)
metrics.to_csv(RESULTS / "daily_metrics.csv", index=False)
print(pd.DataFrame(rows).round(3).to_string())

# feature importance of the gradient boosting model trained on everything before the last winter
gb = models["Gradient boosting"].fit(data.loc[data.index < "2025-11-01", features],
                                     data.loc[data.index < "2025-11-01", "target"])
imp = pd.Series(gb.feature_importances_, index=features).sort_values()
with open(RESULTS / "ml_setup.json", "w") as f:
    json.dump({"best_params": best_params, "features": features, "n_tuning_rows": int(tune.sum()),
               "gb_importance": imp.round(4).to_dict()}, f, indent=2, default=str)

# final comparison figure across all model families
show = ["Persistence", "ETS", "VAR", "SARIMA + Fourier", "Random forest", "Gradient boosting", "SVR",
        "SARIMAX, ECMWF forecast", "SARIMAX, actual weather"]
show = [s for s in show if s in set(metrics["model"])]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
for ax, winter in zip(axes, ["2024/25", "2025/26"]):
    m = metrics[metrics["winter"] == winter].set_index("model").loc[show]
    ax.barh(range(len(show)), m["skill_vs_persistence_%"], color=[MODEL_COLORS[s] for s in show], height=0.68)
    ax.axvline(0, color=INK, lw=1)
    for k, v in enumerate(m["skill_vs_persistence_%"]):
        ax.text(v + (0.6 if v >= 0 else -0.6), k, f"{v:+.0f}%", va="center", ha="left" if v >= 0 else "right", fontsize=9)
    ax.set_yticks(range(len(show)), show)
    ax.invert_yaxis()
    ax.set_xlabel("RMSE improvement over persistence (%)")
    ax.set_title(f"Winter {winter}")
    ax.grid(axis="y", visible=False)
fig.tight_layout()
save(fig, "fig11_skill_all_models.png")
