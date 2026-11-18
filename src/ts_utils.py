"""Helper functions adapted from the DS207 class notebooks (SARIMAX and VAR lectures).

Class functions kept with the same names and logic:
  optimize_SARIMAX   AIC grid search over (p, q, P, Q)
  recursive_forecast one step ahead rolling forecast, 'last' baseline and 'SARIMAX'
  optimize_VAR       AIC search over the VAR order
  mape               mean absolute percentage error

Changes made for this project:
  * recursive_forecast in class passes the actual exogenous values of the day being
    predicted. Here the caller chooses what goes in that row: observed weather (an
    upper bound) or the weather forecast that was available one day earlier (realistic).
  * parameters are re-estimated every `refit_every` days instead of every day, and the
    state is updated daily in between, to keep the run time reasonable.
  * missing days are allowed in the target (the state space model skips them).
"""
from itertools import product
from typing import Union
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.varmax import VARMAX

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------- metrics
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def exceedance_accuracy(y_true, y_pred, limit=15.0):
    """Share of days where the forecast is on the correct side of the WHO limit."""
    return float(np.mean((np.asarray(y_true) > limit) == (np.asarray(y_pred) > limit)))


def all_metrics(y_true, y_pred, limit=15.0) -> dict:
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = np.asarray(y_true)[ok], np.asarray(y_pred)[ok]
    return {"n": int(ok.sum()), "RMSE": rmse(yt, yp), "MAE": mae(yt, yp), "MAPE": float(mape(yt, yp)),
            "WHO_acc": exceedance_accuracy(yt, yp, limit)}


def diebold_mariano(y_true, pred_model, pred_base, h: int = 1):
    """Diebold Mariano test on squared errors with the Harvey small sample correction.
    Negative statistic means the model has the smaller loss. Returns (statistic, p value)."""
    y_true, pred_model, pred_base = map(np.asarray, (y_true, pred_model, pred_base))
    ok = np.isfinite(y_true) & np.isfinite(pred_model) & np.isfinite(pred_base)
    d = (y_true[ok] - pred_model[ok]) ** 2 - (y_true[ok] - pred_base[ok]) ** 2
    n = len(d)
    dbar = d.mean()
    gamma = [np.sum((d[k:] - dbar) * (d[:n - k] - dbar)) / n for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    stat = dbar / np.sqrt(var)
    stat *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    return float(stat), float(p)


# ---------------------------------------------------------------- features
def fourier_terms(index: pd.DatetimeIndex, K: int = 2, period: float = 365.25) -> pd.DataFrame:
    """Annual sine and cosine pairs, used instead of a 365 day seasonal ARIMA term."""
    t = index.dayofyear.values
    cols = {}
    for k in range(1, K + 1):
        cols[f"sin{k}"] = np.sin(2 * np.pi * k * t / period)
        cols[f"cos{k}"] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(cols, index=index)


# ---------------------------------------------------------------- SARIMAX (class notebook)
def optimize_SARIMAX(endog: Union[pd.Series, list], exog: Union[pd.Series, list], order_list: list,
                     d: int, D: int, s: int) -> pd.DataFrame:
    results = []
    for order in order_list:
        try:
            model = SARIMAX(endog, exog, order=(order[0], d, order[1]),
                            seasonal_order=(order[2], D, order[3], s),
                            simple_differencing=False).fit(disp=False)
        except Exception:
            continue
        results.append([order, model.aic])
    result_df = pd.DataFrame(results, columns=["(p,q,P,Q)", "AIC"])
    # Sort in ascending order, lower AIC is better
    return result_df.sort_values(by="AIC", ascending=True).reset_index(drop=True)


def recursive_forecast(endog: pd.Series, exog: Union[pd.DataFrame, None], train_len: int, horizon: int,
                       window: int, method: str, order=(1, 0, 1), seasonal_order=(0, 0, 0, 0),
                       refit_every: int = 7, steps_ahead: int = 1, exog_last: pd.DataFrame = None,
                       trend: str = "n") -> list:
    """Rolling origin forecast with an expanding training window.

    endog[:i] is everything known at the forecast origin; row i of `exog` holds the
    regressors for the day being predicted (already lagged or forecast, so no leakage).
    steps_ahead=2 predicts day i+1 from the same origin (two day lead); `exog_last` then
    supplies the regressors of that last day (the two day ahead weather forecast).
    """
    total_len = train_len + horizon

    if method == "last":
        pred_last_value = []
        for i in range(train_len, total_len, window):
            last_value = endog[:i].dropna().iloc[-1]
            pred_last_value.extend(last_value for _ in range(window))
        return pred_last_value

    if method == "seasonal_naive":      # same weekday one week earlier
        return [endog.iloc[i - 7] if np.isfinite(endog.iloc[i - 7]) else endog[:i].dropna().iloc[-1]
                for i in range(train_len, total_len)]

    if method == "SARIMAX":
        pred_SARIMAX = []
        res, params = None, None
        for n, i in enumerate(range(train_len, total_len, window)):
            ex_train = None if exog is None else exog[:i]
            if n % refit_every == 0:
                model = SARIMAX(endog[:i], ex_train, order=order, seasonal_order=seasonal_order,
                                trend=trend, simple_differencing=False)
                res = model.fit(disp=False, start_params=params, maxiter=200)
                params = res.params
            else:
                res = res.append(endog[i - window:i], exog=None if exog is None else exog[i - window:i])
            ex_future = None if exog is None else exog[i:i + steps_ahead]
            if ex_future is not None and len(ex_future) < steps_ahead:
                pred_SARIMAX.append(np.nan)
                continue
            if exog_last is not None:
                ex_future = pd.concat([ex_future.iloc[:-1], exog_last.iloc[[i + steps_ahead - 1]]])
            if ex_future is not None and ex_future.isna().any().any():
                pred_SARIMAX.append(np.nan)
                continue
            oos_pred = res.get_forecast(steps=steps_ahead, exog=ex_future).predicted_mean
            pred_SARIMAX.append(oos_pred.iloc[-1])
        return pred_SARIMAX

    raise ValueError(method)


# ---------------------------------------------------------------- VAR (class notebook)
def optimize_VAR(endog: Union[pd.DataFrame, list], max_p: int = 8) -> pd.DataFrame:
    results = []
    for i in range(1, max_p + 1):
        try:
            model = VARMAX(endog, order=(i, 0)).fit(disp=False, maxiter=100)
        except Exception:
            continue
        results.append([i, model.aic])
    result_df = pd.DataFrame(results, columns=["p", "AIC"])
    return result_df.sort_values(by="AIC", ascending=True).reset_index(drop=True)


def rolling_forecast_VAR(df: pd.DataFrame, train_len: int, horizon: int, p: int, target: str,
                         refit_every: int = 30) -> list:
    """One step ahead VAR forecast of `target` (class rolling_forecast with window 1)."""
    preds, res, params = [], None, None
    for n, i in enumerate(range(train_len, train_len + horizon)):
        if n % refit_every == 0:
            res = VARMAX(df[:i], order=(p, 0)).fit(disp=False, maxiter=100, start_params=params)
            params = res.params
        else:
            res = res.append(df[i - 1:i])
        preds.append(res.get_forecast(steps=1).predicted_mean[target].iloc[0])
    return preds
