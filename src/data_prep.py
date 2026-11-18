"""Step 1. Build the modelling tables from the raw downloads.

Outputs (data/processed):
  city_hourly.csv   hourly city PM2.5 (median across Yerevan sensors), local time
  daily.csv         daily mean PM2.5 plus daily weather: observed, ECMWF and GFS forecasts
"""
import numpy as np
import pandas as pd

from config import (PM_FILE, WEATHER_FILE, PROCESSED, EXCLUDED_SENSORS, MIN_READINGS_PER_HOUR,
                    MIN_HOURS_PER_DAY, UTC_OFFSET_HOURS)

LOCAL = pd.Timedelta(hours=UTC_OFFSET_HOURS)


def build_city_hourly() -> pd.DataFrame:
    pm = pd.read_csv(PM_FILE)
    pm["time"] = pd.to_datetime(pm["hour_utc"], utc=True).dt.tz_localize(None) + LOCAL
    pm = pm[~pm["sensor_id"].isin(EXCLUDED_SENSORS)]
    pm = pm[pm["n_readings_pm25"] >= MIN_READINGS_PER_HOUR]
    grouped = pm.groupby("time")["pm25"]
    city = pd.DataFrame({"pm25": grouped.median(), "n_sensors": grouped.size()})
    full_index = pd.date_range(city.index.min(), city.index.max(), freq="h")
    city = city.reindex(full_index)
    city["n_sensors"] = city["n_sensors"].fillna(0).astype(int)
    city.index.name = "time"
    return city


def build_hourly_weather() -> pd.DataFrame:
    w = pd.read_csv(WEATHER_FILE)
    w["time"] = pd.to_datetime(w["time_utc"]) + LOCAL
    return w.drop(columns="time_utc").set_index("time")


def daily_weather(w: pd.DataFrame, prefix: str, suffix: str = "") -> pd.DataFrame:
    """Daily features from hourly weather columns named <prefix><variable><suffix>."""
    col = lambda v: f"{prefix}{v}{suffix}"
    day = w.index.floor("D")
    out = pd.DataFrame({
        "wind": w[col("wind_speed_10m")].groupby(day).mean(),
        "temp": w[col("temperature_2m")].groupby(day).mean(),
        "rh": w[col("relative_humidity_2m")].groupby(day).mean(),
        "precip": np.log1p(w[col("precipitation")].groupby(day).sum(min_count=20)),
        "pressure": w[col("surface_pressure")].groupby(day).mean(),
    })
    # keep a day only when at least 20 of its 24 hours are present
    enough = w[col("wind_speed_10m")].groupby(day).count() >= 20
    return out.where(enough)


def build_daily(city: pd.DataFrame, w: pd.DataFrame) -> pd.DataFrame:
    day = city.index.floor("D")
    counts = city["pm25"].groupby(day).count()
    daily = pd.DataFrame({"pm25": city["pm25"].groupby(day).mean().where(counts >= MIN_HOURS_PER_DAY),
                          "hours": counts})
    daily = daily.asfreq("D")
    daily["log_pm25"] = np.log(daily["pm25"])

    blocks = {"obs": daily_weather(w, "obs_")}
    blocks["obs"]["blh"] = w["obs_boundary_layer_height"].groupby(w.index.floor("D")).mean()
    for model in ("ecmwf", "gfs"):
        for lead in (1, 2):
            blocks[f"{model}{lead}"] = daily_weather(w, f"fc_{model}_", f"_d{lead}")
    # archived forecast gaps: fall back to the other model's forecast of the same lead time,
    # the way an operational forecaster would (ECMWF misses 18 to 22 April 2026)
    for lead in (1, 2):
        e, g = blocks[f"ecmwf{lead}"], blocks[f"gfs{lead}"]
        start_e, start_g = e.dropna().index.min(), g.dropna().index.min()
        gap_e = e.isna().any(axis=1) & (e.index > start_e)
        gap_g = g.isna().any(axis=1) & (g.index > start_g)
        blocks[f"ecmwf{lead}"] = e.where(~gap_e, g)
        blocks[f"gfs{lead}"] = g.where(~gap_g, e)
        daily[f"ecmwf{lead}_filled_from_gfs"] = gap_e.reindex(daily.index, fill_value=False)
        daily[f"gfs{lead}_filled_from_ecmwf"] = gap_g.reindex(daily.index, fill_value=False)
    for name, block in blocks.items():
        daily = daily.join(block.add_prefix(f"{name}_"))
    daily.index.name = "date"
    return daily


def main():
    city = build_city_hourly()
    w = build_hourly_weather()
    # the first and the last local day are cut by the UTC download window
    first = city.index.min().ceil("D")
    last = (city.index.max() + pd.Timedelta(hours=1)).floor("D") - pd.Timedelta(hours=1)
    city = city.loc[first:last]
    daily = build_daily(city, w)

    hourly = city.join(w[[c for c in w.columns if c.startswith("obs_")
                          or c.startswith("fc_ecmwf_") and c.endswith("_d1")]], how="left")
    hourly.to_csv(PROCESSED / "city_hourly.csv")
    daily.to_csv(PROCESSED / "daily.csv")

    valid = daily["pm25"].notna()
    winter = daily.index.month.isin([11, 12, 1, 2])
    print(f"Hourly city series: {len(city):,} hours, {city['pm25'].notna().sum():,} with data "
          f"({city.index.min()} to {city.index.max()})")
    print(f"Daily series: {len(daily):,} days, {valid.sum():,} valid, {(valid & winter).sum():,} valid winter days")
    for c in [c for c in daily.columns if "filled_from" in c]:
        print(f"{c}: {int(daily[c].sum())} days")
    print(f"Winter days above WHO 15: {(daily.loc[valid & winter, 'pm25'] > 15).mean():.1%}")
    return city, daily


if __name__ == "__main__":
    main()
