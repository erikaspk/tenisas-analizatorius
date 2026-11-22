"""Duomenų atsisiuntimas (Jeff Sackmann / Tennis Abstract, CC BY-NC-SA) ir aspektų skaičiavimas."""
import os, re, time, urllib.request
import numpy as np
import pandas as pd

BASE = {"ATP": "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{y}.csv",
        "WTA": "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_{y}.csv"}
DATA, YEARS = "data", 5

# (raktas, pavadinimas, režimas, numatytas svoris). Režimai: pct = procentai, hi = daugiau geriau, elo = Elo skirtumas
FACTORS = [
    ("winrate", "Winrate (365 d.), %", "pct", 10),
    ("surface", "Surface winrate (365 d.), %", "pct", 10),
    ("serve", "Serve: taškai laimėti, %", "pct", 10),
    ("return", "Return: taškai laimėti, %", "pct", 10),
    ("experience", "Experience (rungtynės per 5 m.)", "hi", 4),
    ("common", "Common opponents, %", "pct", 5),
    ("momentum", "Momentum (forma − winrate)", "pct", 8),
    ("form", "Form (10 rungt., 90 d.), %", "pct", 5),
    ("quality", "Quality form (30 d.)", "hi", 5),
    ("comeback", "Come back strength, %", "pct", 7),
    ("elo", "Elo (mano pasiūlymas)", "elo", 8),
    ("selo", "Elo pagal dangą", "elo", 0),
]


def load(tour, refresh=False):
    os.makedirs(DATA, exist_ok=True)
    now, frames = pd.Timestamp.today(), []
    for y in range(now.year - YEARS + 1, now.year + 1):
        f = f"{DATA}/{tour}_{y}.csv"
        stale = (not os.path.exists(f)) or (y == now.year and (refresh or time.time() - os.path.getmtime(f) > 86400))
        if stale:
            try:
                urllib.request.urlretrieve(BASE[tour].format(y=y), f)
            except Exception:
                if not os.path.exists(f):
                    continue
        frames.append(pd.read_csv(f))
    if not frames:
        raise RuntimeError("Nepavyko gauti duomenų. Patikrink internetą arba įdėk CSV į data/ aplanką.")
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df.tourney_date.astype(int).astype(str), format="%Y%m%d")
    return df.sort_values("date").reset_index(drop=True)


def _first_set(score):
    m = re.match(r"\s*(\d+)-(\d+)", str(score))
    return (int(m[1]), int(m[2])) if m else None


def to_long(df):
    fs = df.score.map(_first_set)
    w = pd.DataFrame({"player": df.winner_name, "opp": df.loser_name, "win": 1, "date": df.date, "surface": df.surface,
                      "opp_rank": df.loser_rank, "svpt": df.w_svpt, "svwon": df.w_1stWon + df.w_2ndWon,
                      "rpt": df.l_svpt, "rwon": df.l_svpt - df.l_1stWon - df.l_2ndWon,
                      "lost1": fs.map(lambda x: x is not None and x[0] < x[1])})
    l = pd.DataFrame({"player": df.loser_name, "opp": df.winner_name, "win": 0, "date": df.date, "surface": df.surface,
                      "opp_rank": df.winner_rank, "svpt": df.l_svpt, "svwon": df.l_1stWon + df.l_2ndWon,
                      "rpt": df.w_svpt, "rwon": df.w_svpt - df.w_1stWon - df.w_2ndWon,
                      "lost1": fs.map(lambda x: x is not None and x[0] > x[1])})
    return pd.concat([w, l]).sort_values("date", kind="stable").reset_index(drop=True)


def elo(df, k=32):
    e, es = {}, {}
    for w, l, s in zip(df.winner_name, df.loser_name, df.surface):
        for d in (e, es.setdefault(s, {})):
            a, b = d.get(w, 1500), d.get(l, 1500)
            p = 1 / (1 + 10 ** ((b - a) / 400))
            d[w], d[l] = a + k * (1 - p), b - k * (1 - p)
    return e, es


def stats(L, name, surface, today, elos):
    p = L[L.player == name]
    if p.empty:
        return None
    d = lambda n: today - pd.Timedelta(days=n)
    y = p[p.date >= d(365)]
    o = {"winrate": y.win.mean() * 100 if len(y) >= 10 else None}
    s = y[y.surface == surface]
    o["surface"] = s.win.mean() * 100 if len(s) >= 5 else None
    sv = y.dropna(subset=["svpt", "svwon"])
    o["serve"] = sv.svwon.sum() / sv.svpt.sum() * 100 if sv.svpt.sum() > 0 else None
    rt = y.dropna(subset=["rpt", "rwon"])
    o["return"] = rt.rwon.sum() / rt.rpt.sum() * 100 if rt.rpt.sum() > 0 else None
    o["experience"] = len(p)
    f = p[p.date >= d(90)].tail(10)
    o["form"] = f.win.mean() * 100 if len(f) >= 5 else None
    o["momentum"] = o["form"] - o["winrate"] if o["form"] is not None and o["winrate"] is not None else None
    q = p[(p.date >= d(30)) & (p.win == 1)]
    o["quality"] = float((10 / np.sqrt(q.opp_rank.fillna(150).clip(lower=1))).sum())
    c = p[(p.date >= d(730)) & p.lost1]
    o["comeback"] = c.win.mean() * 100 if len(c) >= 5 else None
    o["elo"] = elos[0].get(name)
    o["selo"] = elos[1].get(surface, {}).get(name)
    return o


def common(L, a, b, today):
    y = L[L.date >= today - pd.Timedelta(days=365)]
    ya, yb = y[y.player == a], y[y.player == b]
    c = (set(ya.opp) & set(yb.opp)) - {a, b}
    if not c:
        return None, None, 0
    return ya[ya.opp.isin(c)].win.mean() * 100, yb[yb.opp.isin(c)].win.mean() * 100, len(c)


def h2h(L, a, b):
    m = L[(L.player == a) & (L.opp == b)]
    return int(m.win.sum()), int(len(m) - m.win.sum())


def _norm(mode, a, b):
    if mode == "pct":
        n = (a - b) / 30
    elif mode == "elo":
        n = (a - b) / 200
    else:
        m = max(abs(a), abs(b))
        n = (a - b) / m if m else 0
    return max(-1.0, min(1.0, n))


def score(rows):
    """rows: [(pavadinimas, režimas, svoris, a, b)] -> (PP_A, PP_B, lentelė). Taškai dalijami proporcingai skirtumui."""
    A = B = 0.0
    out = []
    for label, mode, w, a, b in rows:
        ok = a is not None and b is not None and not (pd.isna(a) or pd.isna(b)) and w and w > 0
        if not ok:
            out.append({"Aspektas": label, "Svoris": w, "A": a, "B": b, "PP A": None, "PP B": None})
            continue
        sh = 0.5 + _norm(mode, a, b) / 2
        A += w * sh
        B += w * (1 - sh)
        out.append({"Aspektas": label, "Svoris": w, "A": round(a, 1), "B": round(b, 1),
                    "PP A": round(w * sh, 2), "PP B": round(w * (1 - sh), 2)})
    return A, B, out
