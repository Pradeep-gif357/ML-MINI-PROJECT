"""Creates FAKE earthquake + eruption data so you can test the pipeline
without downloading anything. Results on this data mean nothing scientifically."""
import numpy as np, pandas as pd
rng = np.random.default_rng(0)

# eruption episodes (start, end) between 1985 and 2016
starts = pd.to_datetime(["1986-07-01","1987-06-20","1988-02-10","1989-01-05","1990-03-01",
                         "1991-05-15","1993-04-01","1995-02-20","1997-01-25","1999-07-10",
                         "2002-03-05","2005-06-01","2008-08-15","2011-03-05","2014-06-27"])
durs = rng.integers(10, 120, len(starts))
ends = starts + pd.to_timedelta(durs, "D")
pd.DataFrame({"start": starts, "end": ends}).to_csv("eruptions.csv", index=False)

# earthquakes: background rate + more quakes in the 30 days before and during eruptions
t0, t1 = pd.Timestamp("1985-01-01"), pd.Timestamp("2016-12-31")
days = (t1 - t0).days
rate = np.full(days, 0.15)
for s, e in zip(starts, ends):
    a, b, c = (s - t0).days, (e - t0).days, max((s - t0).days - 30, 0)
    rate[c:a] += 0.5; rate[a:b] += 0.8
counts = rng.poisson(rate)
day_idx = np.repeat(np.arange(days), counts)
times = t0 + pd.to_timedelta(day_idx + rng.random(len(day_idx)), "D")
n = len(times)
df = pd.DataFrame({
    "time": times,
    "latitude": rng.normal(19.35, 0.05, n).clip(19.1, 19.5),
    "longitude": rng.normal(-155.15, 0.12, n).clip(-155.5, -155.0),
    "depth": rng.gamma(4, 2, n),
    "mag": rng.exponential(0.6, n) + 1.0,
}).sort_values("time")
df.to_csv("earthquakes.csv", index=False)
print(f"wrote earthquakes.csv ({n} quakes) and eruptions.csv ({len(starts)} eruptions)")
