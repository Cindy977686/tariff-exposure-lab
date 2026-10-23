"""数据加载与降级。

优先读 data/raw 下的真实数据（文件名与字段见 README）；读不到就回落到
合成数据，保证任何时候 `streamlit run app.py` 都能打开界面。
"""

import os
from dataclasses import dataclass, field

import pandas as pd

from . import demo_data

TRADE_FILE = "trade.csv"
TARIFF_FILE = "tariff.csv"
FIRM_FILE = "firms.csv"

TRADE_COLS = ["year", "hs_code", "partner", "export_value"]
TARIFF_COLS = ["year", "hs_code", "partner", "tariff_rate"]
FIRM_COLS = [
    "stkcd", "year", "industry_code", "industry_name", "capex_ratio",
    "cash_ratio", "rd_ratio", "leverage", "roa", "size", "soe", "export_share",
]

# 常见中文表头映射，省去手工改名
ALIASES = {
    "年份": "year", "会计年度": "year", "年度": "year",
    "海关编码": "hs_code", "HS编码": "hs_code", "商品编码": "hs_code", "税则号列": "hs_code",
    "贸易伙伴": "partner", "国家": "partner", "伙伴国": "partner",
    "出口额": "export_value", "出口金额": "export_value", "出口额美元": "export_value",
    "税率": "tariff_rate", "关税税率": "tariff_rate", "最惠国税率": "tariff_rate",
    "证券代码": "stkcd", "股票代码": "stkcd", "公司代码": "stkcd",
    "行业代码": "industry_code", "证监会行业代码": "industry_code",
    "行业名称": "industry_name", "证监会行业名称": "industry_name",
    "资本支出比率": "capex_ratio", "投资率": "capex_ratio",
    "现金持有": "cash_ratio", "现金比率": "cash_ratio",
    "研发强度": "rd_ratio", "研发投入占比": "rd_ratio",
    "资产负债率": "leverage", "杠杆率": "leverage",
    "总资产收益率": "roa", "资产回报率": "roa",
    "企业规模": "size", "上市年限": "age",
    "国有企业": "soe", "产权性质": "soe",
    "出口依存度": "export_share",
    "SA指数": "sa_index", "融资约束": "sa_index",
}


@dataclass
class Dataset:
    trade: pd.DataFrame
    tariffs: pd.DataFrame
    firms: pd.DataFrame
    mode: str
    warnings: list = field(default_factory=list)

    @property
    def is_real(self) -> bool:
        return self.mode == "real"


def _read(data_dir: str, name: str):
    path = os.path.join(data_dir, name)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(path, encoding="gbk")
        except Exception:
            return None
    except Exception:
        return None
    if df.empty:
        return None
    df = df.rename(columns={c: c.strip() for c in df.columns})
    return df.rename(columns={k: v for k, v in ALIASES.items() if k in df.columns})


def _check(df, cols, label, warnings):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        warnings.append(f"{label} 缺少字段：{', '.join(miss)}")
        return False
    return True


def load(data_dir: str, mode: str = "auto") -> Dataset:
    """mode: auto | real | demo"""
    warnings = []

    if mode in ("auto", "real"):
        trade = _read(data_dir, TRADE_FILE)
        tariffs = _read(data_dir, TARIFF_FILE)
        firms = _read(data_dir, FIRM_FILE)
        ok = all(x is not None for x in (trade, tariffs, firms))
        if ok:
            ok = all([
                _check(trade, TRADE_COLS, "trade.csv", warnings),
                _check(tariffs, TARIFF_COLS, "tariff.csv", warnings),
                _check(firms, FIRM_COLS, "firms.csv", warnings),
            ])
        if ok:
            return Dataset(trade, tariffs, firms, "real", warnings)
        if mode == "real":
            warnings.append("真实数据不可用，已回落到合成数据。")

    d_trade, d_tariff, d_firms, _ = demo_data.generate()
    if not warnings:
        warnings.append("当前为合成演示数据，结论无经济含义，仅用于验证管道。")
    return Dataset(d_trade, d_tariff, d_firms, "demo", warnings)


def schema_hint() -> dict:
    return {
        "trade.csv": TRADE_COLS,
        "tariff.csv": TARIFF_COLS,
        "firms.csv": FIRM_COLS,
    }
