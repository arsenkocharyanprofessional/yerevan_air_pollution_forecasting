"""Shared settings for the Yerevan PM2.5 forecasting project (DS207 Time Series)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
for _p in (PROCESSED, FIGURES, RESULTS):
    _p.mkdir(parents=True, exist_ok=True)

PM_FILE = RAW / "yerevan_pm_hourly_sensorcommunity.csv"
SENSOR_FILE = RAW / "yerevan_pm_sensors.csv"
WEATHER_FILE = RAW / "yerevan_weather_obs_and_forecasts.csv"

# ---- series definition (same rules as the data visualization project) ----
EXCLUDED_SENSORS = [83835,   # located in Gyumri, not Yerevan
                    99632]   # different sensor model (PMS5003), reads about 2.5 times higher
MIN_READINGS_PER_HOUR = 5    # drop sensor hours built from fewer raw readings
MIN_HOURS_PER_DAY = 18       # a local day is valid with at least 18 hourly values
UTC_OFFSET_HOURS = 4         # Yerevan local time, no daylight saving
WHO_DAILY_LIMIT = 15.0       # WHO 2021 24 hour guideline for PM2.5, micrograms per m3

# ---- evaluation design ----
WINTER_MONTHS = [11, 12, 1, 2]
TEST_WINTERS = {
    "2024/25": ("2024-11-01", "2025-02-28"),
    "2025/26": ("2025-11-01", "2026-02-28"),
    "2026/27 (live)": ("2026-11-01", "2026-11-16"),
}
FIRST_TEST_DAY = "2024-11-01"    # everything before this date is the tuning sample

# ---- model inputs ----
WEATHER_VARS = ["wind", "temp", "rh", "precip", "pressure"]   # daily features used as regressors
FOURIER_K = 2                    # annual Fourier harmonics
REFIT_EVERY = 7                  # days between parameter re-estimation in the rolling forecast
SEED = 42
