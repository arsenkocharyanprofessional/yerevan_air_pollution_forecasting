# Forecasting winter PM2.5 air pollution in Yerevan

DS207 Time Series, American University of Armenia. Individual project by Arsen Kocharyan.

**Question.** How well can tomorrow's daily PM2.5 in Yerevan be forecast in winter, which methods beat the
simple rule "tomorrow equals today", and where does the predictability come from: the series' own memory,
the weather that actually happens, or the weather forecast that is available in advance?

## Data

| Source | What | Span |
|---|---|---|
| [Sensor.Community archive](https://archive.sensor.community/) | PM2.5 and PM10 from 10 citizen sensors (8 SDS011 sensors in Yerevan are used) | 1 Jul 2023 to 16 Nov 2026, hourly |
| [Open-Meteo historical weather API](https://open-meteo.com/en/docs/historical-weather-api) | Weather archive (ECMWF IFS reanalysis based) for 40.18 N, 44.50 E | same, hourly |
| [Open-Meteo Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api) | Archived 1 and 2 day ahead forecasts from ECMWF IFS 0.25 and GFS | from Jan/Feb 2024 |

`src/download_data_browser.js` is the script that produced the three files in `data/raw/` (it runs in a
browser console because both archives are reached through a browser).

Series definition: sensors 83835 (Gyumri) and 99632 (PMS5003, different sensor model) are excluded, sensor
hours with fewer than 5 raw readings are dropped, the city value is the hourly median across sensors, times
are local (UTC+4), and a day is valid with 18 or more hours. Result: 1,234 days, 1,190 valid, 362 valid
winter days (Nov to Feb), 67% of them above the WHO 24 hour guideline of 15 µg/m³.

## Method in short

* Target: log of daily mean PM2.5, forecast one day ahead (also two days ahead, and hourly for the next day).
* Rolling origin with an expanding window from 1 Nov 2024; everything before that date is the tuning sample.
  Test windows: winters 2024/25 and 2025/26 (Nov to Feb), plus 1 to 16 Nov 2026 as a live check.
* Models: persistence, seasonal naive (7 days), ETS, SARIMA with annual Fourier terms, SARIMAX with weather
  (actual weather as an upper bound, archived ECMWF and GFS forecasts as the realistic case), VAR with Granger
  tests, random forest, gradient boosting and SVR.
* SARIMAX order: the class AIC grid picks ARMA(1,3), but its AR root sits at 1.009 and its forecasts were
  numerically unstable, so the order was chosen by rolling validation on the one winter inside the tuning
  sample (Dec 2023 to Feb 2024): AR(1) with a constant.
* Metrics: RMSE, MAE, MAPE (winter only), accuracy of the WHO exceedance call, Diebold Mariano test against
  persistence.

## Main results (daily mean PM2.5, one day ahead, RMSE in µg/m³)

| Model | Winter 2024/25 | Winter 2025/26 |
|---|---|---|
| Persistence (yesterday's value) | 11.18 | 12.59 |
| Seasonal naive (7 days ago) | 24.79 | 27.68 |
| ETS(A,N,N) | 11.27 | 12.94 |
| SARIMA + Fourier (no weather) | 12.12 | 13.24 |
| VAR (PM, wind, temperature) | 11.86 | 12.97 |
| SARIMAX + ECMWF forecast | 10.58 | 11.88 |
| SARIMAX + GFS forecast | 9.98 | 12.19 |
| SARIMAX + actual weather (upper bound) | 10.71 | 11.36 |
| Random forest | 17.95 | 12.42 |
| Gradient boosting | 16.71 | 11.99 |
| SVR | 14.16 | 10.32 |

* Weather forecasts give a 3 to 11% gain over persistence one day ahead; no single winter is significant on
  its own (Diebold Mariano p > 0.3). Two days ahead the gain grows to 14 to 17% (p = 0.02 in 2025/26).
* Without weather, no model beats persistence. VAR only sees yesterday's weather and does not help, although
  wind strongly Granger causes PM2.5: the value lies in knowing tomorrow's weather.
* A model trained on actual weather but fed forecasts is 18 to 20% worse than persistence. Train on the
  inputs you will really have.
* ML models failed in 2024/25 (only one partial winter to learn from) and SVR was the best model in 2025/26.
* Hourly, 24 h ahead: the daily SARIMAX forecast times the typical daily shape beats "same hour yesterday"
  (13.1 vs 14.1 and 15.5 vs 16.7).
* Episodes above 15 µg/m³ last 4 days at the median (longest 40); a shock halves in about 2 days.

Full numbers are in `results/`, figures in `figures/`.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_all.py                                       # about 20 to 30 minutes
```

Steps (all in `src/`): `data_prep.py` builds the series, `s01_eda.py` exploratory analysis, `s02_daily_models.py`
baselines, ETS and SARIMAX, `s03_var_granger.py` VAR and Granger tests, `s04_ml_models.py` random forest,
gradient boosting and SVR, `s05_hourly.py` the hourly task, `s06_extras.py` episodes, anomalies, weather
adjusted trend and lead time. `ts_utils.py` holds the functions adapted from the course notebooks
(`optimize_SARIMAX`, `recursive_forecast`, `optimize_VAR`, `mape`).

## Limitations

Only three winters; low cost sensors with a known humidity bias; the sensor network grew from 1 to 8 and three
sensors changed location in 2025 and 2026; weather comes from one grid point; forecast archives start in early 2024.
