"""贸易政策冲击事件定义。

每个事件是后续暴露度计算与事件研究的锚点。event_year 即事件年（t=0），
事件研究窗口默认取 [event_year - 3, event_year + 3]。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ShockEvent:
    key: str
    name: str
    year: int
    month: int
    channel: str
    description: str
    base_year: int  # 暴露度的基期：当年无税变时取冲击前基准年，度量存量暴露


EVENTS = [
    ShockEvent(
        key="us2018",
        name="2018 中美贸易摩擦",
        year=2018,
        month=7,
        channel="关税",
        description="2018 年 7 月起美国对华多轮加征关税，覆盖约 3600 亿美元商品，"
        "平均税率由 3% 升至 19% 左右。",
        base_year=2017,
    ),
    ShockEvent(
        key="covid2020",
        name="2020 疫情冲击",
        year=2020,
        month=2,
        channel="供给 + 需求",
        description="停工停产叠海外订单中断，供应链与外需同时受挫。当年税率未变，"
        "暴露度取相对 2017 年的存量水平，检验存量暴露更高的行业是否更脆弱。",
        base_year=2017,
    ),
    ShockEvent(
        key="tariff2025",
        name="2025 新一轮关税",
        year=2025,
        month=4,
        channel="关税",
        description="2025 年 4 月美国对等关税落地，对华平均税率进一步抬升，"
        "是检验企业调整能力的最新样本。",
        base_year=2024,
    ),
]

EVENT_MAP = {e.key: e for e in EVENTS}

DEFAULT_EVENT = "us2018"


def get_event(key: str) -> ShockEvent:
    return EVENT_MAP.get(key, EVENT_MAP[DEFAULT_EVENT])


def event_options() -> list:
    return [(e.key, e.name) for e in EVENTS]
