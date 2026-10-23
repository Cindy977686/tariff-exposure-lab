"""构建 tariff.csv：美国对华关税税率（HS2 × 年）。

数据来源
--------
1) MFN 法定税率：World Bank WITS / UNCTAD TRAINS，reporter=USA，partner=World，
   datatype=reported，TARIFFTYPE=MFN，HS6 明细（data/source/wits/us_YYYY.xml）。
   MFN 税率不区分伙伴国，美国对中国适用的基础税率即该 MFN 税率。

2) Section 301 加征：USTR 四张清单的 HTS 明细
   （data/source/section301/*.xlsx，转自 hts.usitc.gov Chapter 99 附注）。
   - List 1  9903.88.01  25%  2018-07-06 起
   - List 2  9903.88.02  25%  2018-08-23 起
   - List 3  9903.88.03/04  10% 2018-09-24 起，2019-05-10 升至 25%
   - List 4A 9903.88.15  15%  2019-09-01 起，2020-02-14 降至 7.5%

3) 2025 年额外加征（IEEPA 芬太尼关税 20% + 对等关税 10%，2025-05-12 日内瓦
   联合声明后水平）。WITS 尚无 2024/2025 年数据，2024–2025 的 MFN 基础沿用
   2023 年（最近可得年份），在报告的数据说明中注明。

汇总口径
--------
HS2 税率 = Σ(HS6 税率 × HS6 税目数) / Σ(HS6 税目数)，即税目加权平均，
与 WITS 页面 "MFN Tariff" 的 Simple Average 口径一致但更稳健。

输出
----
data/raw/tariff.csv           year, hs_code(2 位), partner, tariff_rate(百分数)
data/processed/tariff_hs6.csv HS6 明细，供附录与复核用
"""

import os
import re
import xml.etree.ElementTree as ET

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC_WITS = os.path.join(ROOT, "data", "source", "wits")
SRC_301 = os.path.join(ROOT, "data", "source", "section301")
OUT_RAW = os.path.join(ROOT, "data", "raw", "tariff.csv")
OUT_DETAIL = os.path.join(ROOT, "data", "processed", "tariff_hs6.csv")

NS = {
    "m": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/message",
    "s": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/structurespecific",
}

# 各年 301 加征税率（年末生效水平，单位：小数）
SCHEDULE_301 = {
    2018: {"L1": 0.25, "L2": 0.25, "L3": 0.10},
    2019: {"L1": 0.25, "L2": 0.25, "L3": 0.25, "L4A": 0.15},
    2020: {"L1": 0.25, "L2": 0.25, "L3": 0.25, "L4A": 0.075},
    2021: {"L1": 0.25, "L2": 0.25, "L3": 0.25, "L4A": 0.075},
    2022: {"L1": 0.25, "L2": 0.25, "L3": 0.25, "L4A": 0.075},
    2023: {"L1": 0.25, "L2": 0.25, "L3": 0.25, "L4A": 0.075},
    2024: {"L1": 0.25, "L2": 0.25, "L3": 0.25, "L4A": 0.075},
    2025: {"L1": 0.25, "L2": 0.25, "L3": 0.25, "L4A": 0.075},
}

# 2025 年叠加的全口径关税（IEEPA 芬太尼 20% + 对等关税 10%）
SURCHARGE_2025 = 0.30

LIST_FILE = {
    "L1": "99038801.xlsx",
    "L2": "99038802.xlsx",
    "L3": "99038803.xlsx",
    "L3B": "99038804.xlsx",
    "L4A": "99038815.xlsx",
}


def _hs6_from_any(x):
    """从 10 位/8 位/带点的 HTS 码中取前 6 位。"""
    if pd.isna(x):
        return None
    s = re.sub(r"\D", "", str(x))
    if len(s) < 6:
        return None
    return s[:6]


def load_301_list(fname):
    """读出某张 301 清单里的全部 HS6 码。"""
    path = os.path.join(SRC_301, fname)
    xls = pd.ExcelFile(path)
    # 优先用 10 位实际编码页（Actual HS），它最贴近真实征税范围
    sheet = next((s for s in xls.sheet_names if "actual" in s.lower()), xls.sheet_names[0])
    df = xls.parse(sheet, header=0, dtype=str)
    if df.shape[1] == 0:
        return set()
    col = df.columns[0]
    codes = df[col].map(_hs6_from_any).dropna()
    return set(codes)


def load_301_all():
    """四张清单必须分开保存：2018 年 List 3 只加 10%，与 List 1/2 的 25% 不同。"""
    l1 = load_301_list(LIST_FILE["L1"])
    l2 = load_301_list(LIST_FILE["L2"])
    l3 = load_301_list(LIST_FILE["L3"]) | load_301_list(LIST_FILE["L3B"])
    l4a = load_301_list(LIST_FILE["L4A"])
    return {"L1": l1, "L2": l2, "L3": l3, "L4A": l4a}


def parse_wits(path):
    """解析 SDMX XML，返回 HS6 明细表。"""
    rows = []
    for ev, el in ET.iterparse(path, events=("end",)):
        if not el.tag.endswith("Series"):
            continue
        prod = el.get("PRODUCTCODE")
        obs = el.find("s:Obs", NS)
        if obs is None:
            # 有些文件不带命名空间前缀
            obs = el.find("Obs")
        if prod and obs is not None:
            try:
                val = float(obs.get("OBS_VALUE"))
            except (TypeError, ValueError):
                val = None
            try:
                lines = float(obs.get("TOTALNOOFLINES") or 1)
            except ValueError:
                lines = 1.0
            rows.append((prod, val, lines))
        el.clear()
    df = pd.DataFrame(rows, columns=["hs6", "mfn", "lines"])
    return df.dropna(subset=["mfn"])


def build():
    lists = load_301_all()
    print(
        "301 清单 HS6 数："
        + "，".join(f"{k} {len(v)} 个" for k, v in lists.items())
    )

    detail_frames = []
    for year in range(2015, 2026):
        # 2024/2025 的 MFN 基础用 2023 年（WITS 尚无 2024 数据）
        src_year = year if year <= 2023 else 2023
        path = os.path.join(SRC_WITS, f"us_{src_year}.xml")
        if not os.path.exists(path):
            print(f"跳过 {year}：缺少 {path}")
            continue

        mfn = parse_wits(path)
        mfn["hs6"] = mfn["hs6"].astype(str).str.zfill(6)

        # 该年 301 加征（小数）：逐张清单匹配，List 3 在 2018 年只有 10%
        add = pd.Series(0.0, index=mfn.index)
        sched = SCHEDULE_301.get(year, {})
        for key, rate in sched.items():
            codes = lists[key]
            add = add + mfn["hs6"].isin(codes).astype(float) * rate
        if year == 2025:
            add = add + SURCHARGE_2025

        mfn["add"] = add
        mfn["year"] = year
        mfn["rate"] = mfn["mfn"] + mfn["add"] * 100.0  # 转成百分数
        mfn["hs2"] = mfn["hs6"].str[:2]
        detail_frames.append(mfn.copy())

    detail = pd.concat(detail_frames, ignore_index=True)

    # HS2 = 税目加权平均
    detail["w_rate"] = detail["rate"] * detail["lines"]
    h2 = (
        detail.groupby(["year", "hs2"], as_index=False)
        .agg(w_rate=("w_rate", "sum"), lines=("lines", "sum"))
    )
    h2["tariff_rate"] = (h2["w_rate"] / h2["lines"]).round(3)
    h2 = h2.rename(columns={"hs2": "hs_code"})
    h2["partner"] = "US"
    out = h2[["year", "hs_code", "partner", "tariff_rate"]].sort_values(["year", "hs_code"])

    os.makedirs(os.path.dirname(OUT_RAW), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_DETAIL), exist_ok=True)
    out.to_csv(OUT_RAW, index=False, encoding="utf-8-sig")
    detail[["year", "hs6", "mfn", "add", "rate"]].to_csv(
        OUT_DETAIL, index=False, encoding="utf-8-sig"
    )
    print(f"写出 {OUT_RAW}：{len(out)} 行，{out['year'].nunique()} 个年份")
    print(f"写出 {OUT_DETAIL}：{len(detail)} 行")
    return out


if __name__ == "__main__":
    df = build()
    print("\n各年平均税率（全部 HS2 简单平均）：")
    print(df.groupby("year")["tariff_rate"].mean().round(2).to_string())
    print("\n2017 vs 2019 变化最大的 10 个 HS2：")
    p = df.pivot(index="hs_code", columns="year", values="tariff_rate")
    if 2017 in p.columns and 2019 in p.columns:
        d = (p[2019] - p[2017]).dropna().sort_values(ascending=False).head(10)
        print(d.round(2).to_string())
