// Data download used for this project (run on 17 November 2026).
// How to use: open https://archive.sensor.community/ in a browser, open the developer console,
// paste this file and wait. Three CSV files are saved to the Downloads folder; move them to data/raw/.
// The script only reads public files: the Sensor.Community archive and the Open-Meteo APIs.

const START = '2023-07-01', END = '2026-11-16';            // last full day in the archive
const SENSORS = {                                          // id: [sensor type, first day to try]
  '80868': ['sds011', '2023-06-29'], '83131': ['sds011', '2023-09-20'], '83835': ['sds011', '2023-10-28'],
  '84847': ['sds011', '2023-12-15'], '85218': ['sds011', '2024-01-03'], '88230': ['sds011', '2024-06-01'],
  '91307': ['sds011', '2025-01-10'], '91358': ['sds011', '2025-01-12'], '91627': ['sds011', '2025-01-25'],
  '99632': ['pms5003', '2026-06-05'] };

function saveCsv(name, text) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], { type: 'text/csv' }));
  a.download = name; document.body.appendChild(a); a.click();
}

// ---------- 1. Sensor.Community: one file per sensor per day, averaged to hourly means ----------
async function sensorText(day, id, type) {
  const year = day.slice(0, 4);
  if (year === String(new Date().getUTCFullYear())) {     // current year: plain csv
    const r = await fetch(`/${day}/${day}_${type}_sensor_${id}.csv`);
    return r.ok ? await r.text() : null;
  }
  const r = await fetch(`/${year}/${day}/${day}_${type}_sensor_${id}.csv.gz`);   // older years: gzip
  if (!r.ok) return null;
  const buf = await r.arrayBuffer(), u8 = new Uint8Array(buf);
  if (u8[0] === 0x1f && u8[1] === 0x8b)
    return await new Response(new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'))).text();
  return new TextDecoder().decode(u8);
}

async function downloadSensors() {
  const tasks = [], agg = new Map(), meta = {};
  for (let d = new Date(START + 'T00:00:00Z') - 864e5; d <= new Date(END + 'T00:00:00Z'); d += 864e5) {
    const day = new Date(d).toISOString().slice(0, 10);
    for (const id in SENSORS) if (day >= SENSORS[id][1]) tasks.push([day, id, SENSORS[id][0]]);
  }
  let next = 0;
  async function worker() {
    while (next < tasks.length) {
      const [day, id, type] = tasks[next++];
      const text = await sensorText(day, id, type);
      if (text === null) continue;                         // sensor was offline that day
      const lines = text.split('\n'), head = lines[0].trim().split(';');
      const it = head.indexOf('timestamp'), i1 = head.indexOf('P1'), i2 = head.indexOf('P2');
      const ila = head.indexOf('lat'), ilo = head.indexOf('lon');
      for (let k = 1; k < lines.length; k++) {
        const c = lines[k].split(';'); if (c.length < head.length) continue;
        if (k === 1) {
          const m = meta[id] || (meta[id] = { type, first: day, last: day });
          if (day < m.first) m.first = day;
          if (day >= m.last) { m.last = day; m.lat = c[ila]; m.lon = c[ilo]; }
        }
        const key = id + '|' + c[it].slice(0, 13);         // sensor and UTC hour
        const a = agg.get(key) || agg.set(key, [0, 0, 0, 0]).get(key);
        const p10 = parseFloat(c[i1]), p25 = parseFloat(c[i2]);   // P1 is PM10, P2 is PM2.5
        if (isFinite(p10)) { a[0] += p10; a[1]++; }
        if (isFinite(p25)) { a[2] += p25; a[3]++; }
      }
    }
  }
  await Promise.all(Array.from({ length: 8 }, worker));
  const rows = ['sensor_id,hour_utc,pm10,pm25,n_readings_pm10,n_readings_pm25'];
  [...agg.keys()].sort().forEach(key => {
    const [id, hour] = key.split('|'), a = agg.get(key);
    if (hour < START) return;
    rows.push([id, hour + ':00:00Z', a[1] ? (a[0] / a[1]).toFixed(3) : '', a[3] ? (a[2] / a[3]).toFixed(3) : '', a[1], a[3]].join(','));
  });
  saveCsv('yerevan_pm_hourly_sensorcommunity.csv', rows.join('\n'));
  const s = ['sensor_id,sensor_type,lat,lon,first_day,last_day'];
  Object.keys(meta).sort().forEach(id => s.push([id, meta[id].type, meta[id].lat, meta[id].lon, meta[id].first, meta[id].last].join(',')));
  saveCsv('yerevan_pm_sensors.csv', s.join('\n'));
}

// ---------- 2. Open-Meteo: weather archive plus archived forecasts (Previous Runs API) ----------
async function downloadWeather() {
  const lat = 40.18, lon = 44.50, end = new Date(new Date(END + 'T00:00:00Z').getTime() + 864e5).toISOString().slice(0, 10);
  const obsVars = ['temperature_2m', 'relative_humidity_2m', 'precipitation', 'surface_pressure', 'wind_speed_10m',
                   'wind_direction_10m', 'boundary_layer_height', 'cloud_cover', 'shortwave_radiation'];
  const fcVars = obsVars.filter(v => v !== 'boundary_layer_height');   // not available as archived forecast
  const cols = {};
  const obs = await (await fetch(`https://archive-api.open-meteo.com/v1/archive?latitude=${lat}&longitude=${lon}` +
    `&start_date=${START}&end_date=${end}&hourly=${obsVars.join(',')}&wind_speed_unit=ms&timezone=GMT`)).json();
  const time = obs.hourly.time;
  for (const v of obsVars) cols['obs_' + v] = obs.hourly[v];
  const edges = ['2023-07-01', '2024-01-01', '2024-07-01', '2025-01-01', '2025-07-01', '2026-01-01', '2026-07-01'];
  for (const [tag, model] of [['ecmwf', 'ecmwf_ifs025'], ['gfs', 'gfs_seamless']]) {
    const hv = fcVars.flatMap(v => [v + '_previous_day1', v + '_previous_day2']), byTime = new Map();
    for (let i = 0; i < edges.length; i++) {               // half year chunks keep each response small
      const stop = i + 1 < edges.length ? new Date(new Date(edges[i + 1] + 'T00:00:00Z') - 864e5).toISOString().slice(0, 10) : end;
      const j = await (await fetch(`https://previous-runs-api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}` +
        `&start_date=${edges[i]}&end_date=${stop}&hourly=${hv.join(',')}&models=${model}&wind_speed_unit=ms&timezone=GMT`)).json();
      j.hourly.time.forEach((t, k) => byTime.set(t, Object.fromEntries(hv.map(h => [h, j.hourly[h] ? j.hourly[h][k] : null]))));
      await new Promise(r => setTimeout(r, 500));
    }
    for (const v of fcVars) for (const d of ['1', '2'])
      cols[`fc_${tag}_${v}_d${d}`] = time.map(t => { const o = byTime.get(t); return o ? o[`${v}_previous_day${d}`] : null; });
  }
  const names = Object.keys(cols), rows = ['time_utc,' + names.join(',')];
  time.forEach((t, k) => rows.push(t + ',' + names.map(n => cols[n][k] == null ? '' : cols[n][k]).join(',')));
  saveCsv('yerevan_weather_obs_and_forecasts.csv', rows.join('\n'));
}

downloadWeather().then(downloadSensors).then(() => console.log('all three files saved'));
