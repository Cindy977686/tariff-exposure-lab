"""合成演示数据。

重要说明：这里的数据全部是程序生成的，只用来验证管道和界面能否跑通，
不代表任何真实结论。接上真实海关与财务数据后，把 app 的 DATA_MODE
切到 "real" 即可，其余代码不用改。
"""

import numpy as np
import pandas as pd

from .mapping import hs_to_industry

CHAPTERS = [
    3, 7, 10, 17, 22, 25, 26, 28, 29, 30, 31, 32, 33, 35, 38, 39, 40,
    42, 44, 48, 52, 54, 55, 61, 62, 64, 65, 68, 69, 70, 72, 73, 74,
    76, 82, 83, 84, 85, 86, 87, 89, 90, 94,
]

YEARS = list(range(2014, 2026))
PARTNER = "US"
EVENT_YEARS = (2018, 2020, 2025)


def _build_trade_and_tariff(rng):
    intensity = {ch: float(rng.beta(2.2, 2.8)) for ch in CHAPTERS}

    t_rows = []
    for ch in CHAPTERS:
        base = float(rng.uniform(1.5, 6.0))
        for y in YEARS:
            jump = 0.0
            if y >= 2018:
                jump += 18.0 * intensity[ch]
            if y >= 2025:
                jump += 12.0 * intensity[ch]
            t_rows.append((y, f"{ch:02d}", PARTNER, round(base + jump, 2)))
    tariffs = pd.DataFrame(t_rows, columns=["year", "hs_code", "partner", "tariff_rate"])

    e_rows = []
    for ch in CHAPTERS:
        scale = float(rng.lognormal(mean=21.0, sigma=0.9))
        for y in YEARS:
            trend = 1 + 0.035 * (y - 2014)
            hit = 1.0
            if y >= 2018:
                hit -= 0.28 * intensity[ch] * min(1.0, (y - 2017) / 2)
            val = scale * trend * hit * float(rng.lognormal(0, 0.08))
            e_rows.append((y, f"{ch:02d}", PARTNER, round(val, 2)))
    trade = pd.DataFrame(e_rows, columns=["year", "hs_code", "partner", "export_value"])

    pairs = trade["hs_code"].map(hs_to_industry)
    trade["industry_code"] = [p[0] for p in pairs]
    trade["industry_name"] = [p[1] for p in pairs]
    return trade, tariffs, intensity


def _industry_intensity(trade, intensity):
    base = trade[trade["year"] == 2017].copy()
    base["ch"] = base["hs_code"].astype(int)
    base["s"] = base["ch"].map(intensity)
    base["w"] = base["export_value"] / base.groupby("industry_code")["export_value"].transform("sum")
    base["ws"] = base["w"] * base["s"]
    return base.groupby("industry_code")["ws"].sum().to_dict()


def _build_firms(rng, trade, ind_intensity, n_firms=2400):
    base = trade[trade["year"] == 2017]
    share = base.groupby("industry_code")["export_value"].sum()
    share = share / share.sum()
    industries = list(share.index)
    probs = share.values

    ind_code = rng.choice(industries, size=n_firms, p=probs)
    names = trade.drop_duplicates("industry_code").set_index("industry_code")["industry_name"].to_dict()

    s = np.array([ind_intensity.get(c, 0.4) for c in ind_code])
    s_z = (s - s.mean()) / (s.std() + 1e-9)

    size = rng.normal(22.0 + 0.7 * s_z, 1.1)
    soe = (rng.random(n_firms) < 0.28).astype(int)
    export_share = np.clip(rng.beta(2.0, 5.0) + 0.35 * s_z.clip(-1, 2) * 0.2, 0.0, 0.95)
    sa = -0.9 - 0.12 * size + 0.15 * rng.normal(size=n_firms)
    lev0 = np.clip(0.42 + 0.05 * s_z + 0.10 * rng.normal(size=n_firms), 0.05, 0.92)

    rows = []
    for i in range(n_firms):
        code = f"{600000 + i:06d}"
        for y in YEARS:
            t = y - 2014
            head = 1 + 0.004 * t
            capex = 0.058 + 0.008 * (size[i] - 22.0) / 1.1 + 0.010 * rng.normal()
            cash = 0.185 + 0.012 * rng.normal()
            rd = 0.030 + 0.006 * rng.normal()
            lev = lev0[i] + 0.01 * rng.normal()
            roa = 0.052 + 0.012 * rng.normal()

            if y >= 2018:
                k = min(1.0, (y - 2017) / 2.0)
                capex -= 0.030 * s_z[i] * k
                cash += 0.022 * s_z[i] * k
                rd -= 0.006 * s_z[i] * k
                lev += 0.018 * s_z[i] * k
                roa -= 0.008 * s_z[i] * k
            if y >= 2020:
                k = min(1.0, (y - 2019) / 2.0)
                capex -= 0.016 * s_z[i] * k
                cash += 0.014 * s_z[i] * k
            if y >= 2025:
                k = min(1.0, (y - 2024) / 1.0)
                capex -= 0.012 * s_z[i] * k
                cash += 0.009 * s_z[i] * k

            rows.append((
                code, y, ind_code[i], names.get(ind_code[i], "其他制造"),
                round(max(capex * head, 0.001), 4),
                round(max(cash, 0.005), 4),
                round(max(rd, 0.0), 4),
                round(lev, 4),
                round(roa, 4),
                round(size[i] + 0.02 * t, 3),
                int(soe[i]),
                round(float(export_share[i]), 4),
                round(float(sa[i]), 4),
            ))

    return pd.DataFrame(rows, columns=[
        "stkcd", "year", "industry_code", "industry_name", "capex_ratio",
        "cash_ratio", "rd_ratio", "leverage", "roa", "size", "soe",
        "export_share", "sa_index",
    ])


def generate(seed: int = 42, n_firms: int = 2400):
    rng = np.random.default_rng(seed)
    trade, tariffs, intensity = _build_trade_and_tariff(rng)
    ind_intensity = _industry_intensity(trade, intensity)
    firms = _build_firms(rng, trade, ind_intensity, n_firms=n_firms)
    meta = {
        "source": "synthetic",
        "seed": seed,
        "n_firms": n_firms,
        "note": "合成数据，仅用于验证管道与界面，不代表真实结论。",
    }
    return trade, tariffs, firms, meta


def save_csv(data_dir: str, seed: int = 42):
    import os

    trade, tariffs, firms, _ = generate(seed=seed)
    os.makedirs(data_dir, exist_ok=True)
    trade.to_csv(os.path.join(data_dir, "demo_trade.csv"), index=False, encoding="utf-8-sig")
    tariffs.to_csv(os.path.join(data_dir, "demo_tariff.csv"), index=False, encoding="utf-8-sig")
    firms.to_csv(os.path.join(data_dir, "demo_firms.csv"), index=False, encoding="utf-8-sig")
