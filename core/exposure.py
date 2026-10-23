"""关税暴露度构造与事件研究。

暴露度定义（行业层面，Pierce & Schott 式加权）：

    Exposure_jt = Σ_k  w_kj  ×  ΔTariff_kt
    w_kj        = 基期 t0 行业 j 中产品 k 的出口占该行业总出口的比重
    ΔTariff_kt  = 事件年税率 − 基期税率

w 用基期权重是为了避免企业因冲击改变出口结构而反向影响权重，
这一条在答辩时经常被问，写进变量说明里。
"""

from typing import Optional

import numpy as np
import pandas as pd

GROUP_HIGH = "高暴露"
GROUP_MID = "中暴露"
GROUP_LOW = "低暴露"
GROUP_ORDER = [GROUP_HIGH, GROUP_MID, GROUP_LOW]

from .mapping import hs2_prefix

DEFAULT_OUTCOME = "capex_ratio"


def compute_exposure(
    trade: pd.DataFrame,
    tariffs: pd.DataFrame,
    event_year: int,
    base_year: Optional[int] = None,
    partner: str = "US",
) -> pd.DataFrame:
    """行业层面关税暴露度。

    trade   列：year, hs_code, partner, export_value
    tariffs 列：year, hs_code, partner, tariff_rate
    """
    if base_year is None:
        base_year = event_year - 1

    t = trade[(trade["year"] == base_year) & (trade["partner"] == partner)].copy()
    if t.empty:
        t = trade[trade["year"] == base_year].copy()
    if t.empty:
        return pd.DataFrame(columns=["industry_code", "industry_name", "exposure", "export_value", "n_hs"])

    t["hs_prefix"] = hs2_prefix(t["hs_code"])
    t = t.groupby(["industry_code", "industry_name", "hs_prefix"], as_index=False).agg(
        export_value=("export_value", "sum")
    )
    t["weight"] = t["export_value"] / t.groupby("industry_code")["export_value"].transform("sum")

    tr = tariffs[tariffs["partner"] == partner].copy()
    tr["hs_prefix"] = hs2_prefix(tr["hs_code"])
    tr = tr.groupby(["hs_prefix", "year"], as_index=False).agg(tariff_rate=("tariff_rate", "mean"))

    base_rate = tr[tr["year"] == base_year].set_index("hs_prefix")["tariff_rate"]
    event_rate = tr[tr["year"] == event_year].set_index("hs_prefix")["tariff_rate"]
    delta = (event_rate - base_rate).rename("delta_tariff").reset_index()

    m = t.merge(delta, on="hs_prefix", how="left")
    m["delta_tariff"] = m["delta_tariff"].fillna(0.0)
    m["contrib"] = m["weight"] * m["delta_tariff"]

    out = m.groupby(["industry_code", "industry_name"], as_index=False).agg(
        exposure=("contrib", "sum"),
        export_value=("export_value", "sum"),
        n_hs=("hs_prefix", "nunique"),
    )
    return out.sort_values("exposure", ascending=False).reset_index(drop=True)


def assign_groups(firm: pd.DataFrame, exposure: pd.DataFrame, n_groups: int = 3) -> pd.DataFrame:
    """按行业暴露度给企业分组。

    关键：先在**行业层面**分位，再挂到企业上。若直接对企业行做 qcut，
    同一行业的不同企业会被拆进不同组，分组就失去含义（这是最容易踩的坑）。
    """
    exp = exposure[["industry_code", "exposure"]].drop_duplicates("industry_code").copy()

    if n_groups == 3:
        labels = [GROUP_LOW, GROUP_MID, GROUP_HIGH]
    else:
        labels = [GROUP_LOW, GROUP_HIGH]

    try:
        exp["group"] = pd.qcut(
            exp["exposure"].rank(method="first"), n_groups, labels=labels
        ).astype(str)
    except ValueError:
        exp["group"] = GROUP_MID

    df = firm.drop(columns=[c for c in ("exposure", "group") if c in firm.columns])
    df = df.merge(exp, on="industry_code", how="left")
    df["exposure"] = df["exposure"].fillna(float(exp["exposure"].median()))
    df["group"] = df["group"].fillna(GROUP_MID)
    return df


def add_firm_exposure(df: pd.DataFrame) -> pd.DataFrame:
    """企业层面暴露度：行业暴露度 × 出口依存度权重。

    没有企业出口数据时把 export_share 视为 1，退化为行业暴露度。
    """
    out = df.copy()
    share = out["export_share"] if "export_share" in out.columns else 1.0
    out["firm_exposure"] = out["exposure"] * share
    return out


def event_study(
    df: pd.DataFrame,
    outcome: str,
    event_year: int,
    window: int = 3,
    group_col: str = "group",
    demean: bool = True,
) -> pd.DataFrame:
    """分组事件研究。

    返回 rel_year, group, mean 以及去均值后的 y（相对事件前均值的偏离）。
    去均值是为了让三条线在 t<0 都锚定在 0 附近，方便看出分化。
    """
    d = df.copy()
    d["rel_year"] = d["year"] - event_year
    d = d[(d["rel_year"] >= -window) & (d["rel_year"] <= window)]
    if d.empty or group_col not in d.columns:
        return pd.DataFrame(columns=["rel_year", group_col, "mean", "y", "n"])

    g = d.groupby(["rel_year", group_col], as_index=False).agg(
        mean=(outcome, "mean"), n=(outcome, "size")
    )
    if demean:
        base = g[g["rel_year"] < 0].groupby(group_col)["mean"].mean()
        g["y"] = g.apply(lambda r: r["mean"] - base.get(r[group_col], r["mean"]), axis=1)
    else:
        g["y"] = g["mean"]
    return g


def diff_in_diff(df: pd.DataFrame, outcome: str, event_year: int, window: int = 3) -> dict:
    """高暴露组相对低暴露组的事后平均变化差（简化 DID 估计，不含协变量）。"""
    d = df.copy()
    d["rel_year"] = d["year"] - event_year
    d = d[(d["rel_year"] >= -window) & (d["rel_year"] <= window)]
    d = d[d["group"].isin([GROUP_HIGH, GROUP_LOW])]
    if d.empty:
        return {"estimate": np.nan, "n": 0}

    d["post"] = (d["rel_year"] >= 0).astype(int)
    d["treat"] = (d["group"] == GROUP_HIGH).astype(int)
    cells = d.groupby(["treat", "post"])[outcome].mean()

    def cell(tr, po):
        return cells.get((tr, po), np.nan)

    est = (cell(1, 1) - cell(1, 0)) - (cell(0, 1) - cell(0, 0))
    return {"estimate": est, "n": int(len(d))}


def industry_timeseries(df: pd.DataFrame, outcome: str, industries: list) -> pd.DataFrame:
    d = df[df["industry_name"].isin(industries)]
    return d.groupby(["year", "industry_name"], as_index=False)[outcome].mean()


def describe(df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    g = df.groupby("group", as_index=False).agg(
        firms=("stkcd", "nunique"),
        mean=(outcome, "mean"),
        sd=(outcome, "std"),
    )
    g["order"] = g["group"].map({g_: i for i, g_ in enumerate(GROUP_ORDER)})
    return g.sort_values("order").drop(columns=["order"])
