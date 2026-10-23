"""关税暴露与企业韧性 —— 交互式研究看板。

运行：streamlit run app.py
"""

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import narrative
from core.data_loader import load, schema_hint
from core.events import EVENTS, event_options, get_event
from core.exposure import (
    GROUP_HIGH, GROUP_LOW, GROUP_MID, GROUP_ORDER,
    assign_groups, compute_exposure, describe, diff_in_diff, event_study,
)
from core.mapping import coverage_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

C_HIGH, C_MID, C_LOW = "#B4453C", "#C08A3E", "#3E7C71"
COLOR_MAP = {GROUP_HIGH: C_HIGH, GROUP_MID: C_MID, GROUP_LOW: C_LOW}

OUTCOMES = {
    "capex_ratio": "投资率（资本支出 / 总资产）",
    "cash_ratio": "现金持有（现金 / 总资产）",
    "rd_ratio": "研发强度（研发支出 / 营业收入）",
    "leverage": "资产负债率",
    "roa": "总资产收益率",
}

st.set_page_config(page_title="关税暴露与企业韧性", page_icon="◧", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1240px;}
    h1 {font-size: 1.55rem !important; letter-spacing: -0.01em;}
    .tag {display:inline-block; font-size:0.72rem; padding:2px 8px; border-radius:4px;
          border:1px solid #d5d3cb; color:#5f5e5a; margin-right:6px;}
    .note {font-size:0.8rem; color:#7a7873; line-height:1.7;}
    .big {font-size:1.45rem; font-weight:600; color:#2c2c2a;}
    </style>
    """,
    unsafe_allow_html=True,
)

_VER = tuple(int(x) for x in st.__version__.split(".")[:2])


def _plot(fig, key=None):
    cfg = {"displayModeBar": False}
    if _VER >= (1, 49):
        st.plotly_chart(fig, width="stretch", config=cfg, key=key)
    else:
        st.plotly_chart(fig, use_container_width=True, config=cfg, key=key)


def _df(data):
    try:
        st.dataframe(data, hide_index=True, width="stretch")
    except TypeError:
        st.dataframe(data, hide_index=True, use_container_width=True)


def _base_layout(fig, height=380, title=""):
    fig.update_layout(
        height=height,
        title=dict(text=title, font=dict(size=13, color="#2c2c2a"), x=0, xanchor="left"),
        margin=dict(l=10, r=10, t=42 if title else 16, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="-apple-system, 'Microsoft YaHei', sans-serif", size=12, color="#3d3c38"),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="#d5d3cb", font_size=12),
    )
    fig.update_xaxes(showgrid=False, linecolor="#e2e0d8", zeroline=False)
    fig.update_yaxes(gridcolor="#f0eee7", linewidth=0, zeroline=False)
    return fig


@st.cache_data(show_spinner=False)
def get_data(mode: str):
    return load(DATA_DIR, mode)


@st.cache_data(show_spinner=False)
def exposure_by_event(trade, tariffs, event_year: int, base_year: int):
    return compute_exposure(trade, tariffs, event_year, base_year=base_year)


def main():
    st.markdown("## 关税暴露与企业韧性")
    st.markdown(
        '<span class="tag">赛道四 · 数据分析与智能洞察</span>'
        '<span class="tag">宏观冲击 → 企业反应</span>'
        '<span class="tag">可交互 · 可复现</span>',
        unsafe_allow_html=True,
    )
    st.write("")

    with st.sidebar:
        st.markdown("### 参数")
        ev_key = st.selectbox(
            "冲击事件", [k for k, _ in event_options()],
            format_func=lambda k: get_event(k).name, index=0,
        )
        mode_label = st.radio("数据模式", ["自动", "合成演示", "真实数据"], index=0,
                              help="真实数据放在 data/raw/ 下，字段见「数据与方法」页。")
        mode = {"自动": "auto", "合成演示": "demo", "真实数据": "real"}[mode_label]

        st.divider()
        st.markdown("### 解读层")
        use_llm = st.toggle("启用大模型解读", value=False,
                            help="需配置 OpenAI 兼容接口；失败自动回落到模板。")
        cfg = narrative.env_config()
        api_base = api_key = model = ""
        if use_llm:
            api_base = st.text_input("接口地址", value=cfg["api_base"],
                                     placeholder="https://api.deepseek.com/v1")
            api_key = st.text_input("API Key", value=cfg["api_key"], type="password")
            model = st.text_input("模型名", value=cfg["model"], placeholder="deepseek-chat")

        st.divider()
        st.caption("数据口径")
        window = st.slider("事件窗口（年）", 2, 5, 3)

    ds = get_data(mode)
    event = get_event(ev_key)

    for w in ds.warnings:
        st.caption(f"· {w}")

    exp = exposure_by_event(ds.trade, ds.tariffs, event.year, event.base_year)
    if exp.empty:
        st.error("暴露度计算为空，请检查 year / hs_code / partner 字段是否与事件年份对得上。")
        return

    firm = assign_groups(ds.firms, exp)
    firm = firm[firm[list(OUTCOMES)].notna().all(axis=1)]

    t1, t2, t3, t4 = st.tabs(["冲击层", "响应层", "解读层", "数据与方法"])

    with t1:
        render_shock(ds, exp, firm, event)
    with t2:
        render_response(firm, exp, event, window)
    with t3:
        render_narrative(firm, exp, event, use_llm, api_base, api_key, model)
    with t4:
        render_docs(ds, exp)


def render_shock(ds, exp, firm, event):
    top = exp.iloc[0]
    delta = exp["exposure"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("覆盖行业", f"{exp['industry_code'].nunique()} 个")
    c2.metric("覆盖公司", f"{int(firm['stkcd'].nunique())} 家")
    c3.metric("最高暴露行业", top["industry_name"], f"{top['exposure']:+.2f} pp")
    c4.metric("平均税率变动", f"{delta:+.2f} pp", f"{event.year} 年")

    st.markdown(f"**{event.name}** · {event.channel}型冲击 · {event.description}")

    left, right = st.columns([1.35, 1])

    with left:
        d = exp.sort_values("exposure").tail(14)
        fig = go.Figure(go.Bar(
            x=d["exposure"], y=d["industry_name"], orientation="h",
            marker=dict(color=[C_HIGH if v >= d["exposure"].median() else "#C9A67E" for v in d["exposure"]],
                        line=dict(width=0)),
            text=[f"{v:+.1f}" for v in d["exposure"]],
            textposition="outside", hovertemplate="%{y}：%{x:+.2f} pp<extra></extra>",
        ))
        _base_layout(fig, 420, "行业关税暴露度（出口加权，百分点）")
        fig.update_xaxes(title_text="暴露度（pp）")
        _plot(fig, "bar_exp")

    with right:
        rows = []
        for e in EVENTS:
            x = exposure_by_event(ds.trade, ds.tariffs, e.year, e.base_year)
            if x.empty:
                continue
            for r in x.itertuples():
                rows.append((e.name, r.industry_name, r.exposure))
        hm = pd.DataFrame(rows, columns=["event", "industry", "v"]).pivot(
            index="industry", columns="event", values="v"
        )
        if not hm.empty:
            order = exp.set_index("industry_name")["exposure"].reindex(hm.index).sort_values(ascending=False)
            hm = hm.loc[order.index]
            fig = go.Figure(go.Heatmap(
                z=hm.values, x=list(hm.columns), y=list(hm.index),
                colorscale=[[0, "#EEF4F2"], [0.5, "#EBD9B8"], [1, "#B4453C"]],
                colorbar=dict(title="pp", thickness=10, len=0.8),
                hovertemplate="%{y} · %{x}<br>暴露度 %{z:+.2f} pp<extra></extra>",
            ))
            _base_layout(fig, 420, "各事件的暴露度排序是否稳定")
            fig.update_yaxes(showticklabels=False)
            _plot(fig, "heat")

    st.markdown(
        '<p class="note">暴露度 = Σ（基期出口占比 × 关税变动）。用基期权重，避免企业调整出口结构后反向影响权重。</p>',
        unsafe_allow_html=True,
    )


def render_response(firm, exp, event, window):
    c1, c2 = st.columns([1, 3])
    with c1:
        outcome = st.selectbox("结果变量", list(OUTCOMES.keys()),
                               format_func=lambda k: OUTCOMES[k], index=0)

    es = event_study(firm, outcome, event.year, window=window)
    did = diff_in_diff(firm, outcome, event.year, window=window)

    left, right = st.columns([1.5, 1])

    with left:
        if es.empty:
            st.info("窗口内没有足够观测。")
        else:
            fig = go.Figure()
            for g in GROUP_ORDER:
                d = es[es["group"] == g].sort_values("rel_year")
                if d.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=d["rel_year"], y=d["y"] * 100, name=g, mode="lines+markers",
                    line=dict(color=COLOR_MAP[g], width=2),
                    marker=dict(size=6, color=COLOR_MAP[g]),
                    hovertemplate="t=%{x}<br>%{y:+.2f} pp<extra>" + g + "</extra>",
                ))
            fig.add_vline(x=0, line_width=1, line_dash="dot", line_color="#9a978e")
            fig.add_annotation(x=0, y=1, yref="paper", text=event.name.split(" ")[0],
                               showarrow=False, font=dict(size=11, color="#9a978e"), yshift=8)
            fig.add_hline(y=0, line_width=1, line_color="#d8d5cb")
            _base_layout(fig, 400, f"{OUTCOMES[outcome]}：相对冲击前均值的偏离")
            fig.update_xaxes(title_text="相对事件年（0 = 冲击当年）", dtick=1)
            fig.update_yaxes(title_text="百分点")
            fig.update_layout(legend=dict(orientation="h", y=-0.18, x=0))
            _plot(fig, "es")

    with right:
        est = did["estimate"]
        st.markdown("**高暴露 − 低暴露（事后）**")
        st.markdown(f'<div class="big">{est * 100:+.2f} pp</div>' if pd.notna(est) else "—",
                    unsafe_allow_html=True)
        st.caption(f"简化 DID 估计，观测 {did['n']} 条；未含协变量，标准误待聚类。")

        desc = describe(firm, outcome)
        show = desc.copy()
        show["mean"] = (show["mean"] * 100).round(2)
        show["sd"] = (show["sd"] * 100).round(2)
        show.columns = ["分组", "公司数", "均值(%)", "标准差"]
        _df(show)

    st.divider()

    d = firm.copy()
    d["rel"] = d["year"] - event.year
    pre = d[d["rel"].between(-window, -1)].groupby("industry_name")[outcome].mean()
    post = d[d["rel"].between(0, window)].groupby("industry_name")[outcome].mean()
    chg = (post - pre).rename("chg")
    sc = pd.concat([exp.set_index("industry_name")["exposure"], chg], axis=1).dropna()

    if len(sc) >= 3:
        a, b = np.polyfit(sc["exposure"], sc["chg"], 1)
        xs = np.linspace(sc["exposure"].min(), sc["exposure"].max(), 50)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sc["exposure"], y=sc["chg"] * 100, mode="markers",
            marker=dict(size=12, color="#5C6E8A", opacity=0.85, line=dict(width=0)),
            text=sc.index, hovertemplate="%{text}<br>暴露度 %{x:+.2f} pp<br>变化 %{y:+.2f} pp<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=xs, y=(a * xs + b) * 100, mode="lines", name="拟合",
            line=dict(color="#B4453C", width=1.5, dash="dash"), hoverinfo="skip",
        ))
        _base_layout(fig, 360, "行业层面：暴露度越高，冲击后该指标降得越多")
        fig.update_xaxes(title_text="关税暴露度（pp）")
        fig.update_yaxes(title_text="冲击前后变化（pp）")
        _plot(fig, "scatter")
        st.markdown(
            f'<p class="note">拟合斜率 {a * 100:+.2f}，仅 {len(sc)} 个行业点，图用于展示方向，不构成统计推断。</p>',
            unsafe_allow_html=True,
        )


def render_narrative(firm, exp, event, use_llm, api_base, api_key, model):
    es = event_study(firm, "capex_ratio", event.year)
    did_c = diff_in_diff(firm, "capex_ratio", event.year)["estimate"]
    did_k = diff_in_diff(firm, "cash_ratio", event.year)["estimate"]
    desc = describe(firm, "capex_ratio")

    ctx = narrative.build_context(
        event, exp, did_c, did_k, desc, OUTCOMES["capex_ratio"], firm["stkcd"].nunique()
    )
    text, source = narrative.generate(ctx, use_llm, api_base, api_key, model)

    st.markdown("**自动生成的观察**")
    st.markdown(
        f'<div style="background:#FBFAF6;border:1px solid #E8E5DA;border-radius:10px;'
        f'padding:20px 22px;line-height:1.9;font-size:0.95rem;">'
        f'{text.replace(chr(10) + chr(10), "<br><br>")}</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"生成方式：{source}")

    if use_llm:
        st.info(
            "大模型只拿到上面这张数字表，系统提示词禁止它引入任何未给出的数据。"
            "演示时如果接口不通，会自动显示模板版本，不会开天窗。",
            icon="ⓘ",
        )

    st.divider()
    st.markdown("**决赛被追问时怎么说**")
    qa = [
        ("为什么用行业层面暴露度，而不是企业层面？",
         "海关企业级数据公开版本只到 2016 年，覆盖不到 2018 与 2025 两次冲击。"
         "所以基线用行业暴露度匹配上市公司，是数据可得性下的次优选择；"
         "论文阶段会用 HS6 加 BEC 中间品分类再细化。"),
        ("这是相关还是因果？",
         "本文不主张因果。暴露度按基期出口权重构造，事前与结果变量无同期关联，"
         "但窗口内存在其他政策叠加，因此只做方向性解读。"),
        ("为什么高暴露组投资下降更明显？",
         "两条渠道：一是外需预期恶化直接压低资本支出；"
         "二是银行对高暴露行业收紧授信，融资约束放大投资收缩。"
         "现金持有同步上升支持预防性储蓄的解释。"),
    ]
    for q, a in qa:
        with st.expander(q):
            st.write(a)


def render_docs(ds, exp):
    st.markdown("**数据状态**")
    c1, c2, c3 = st.columns(3)
    c1.metric("数据来源", "真实数据" if ds.is_real else "合成演示")
    c2.metric("企业观测", f"{len(ds.firms):,}")
    c3.metric("行业数", f"{exp['industry_code'].nunique()}")

    if not ds.is_real:
        st.warning(
            "当前是合成数据，所有结论都没有经济含义。它的唯一用途是证明"
            "「数据进 → 计算 → 出图 → 出文字」这条链路是通的。",
            icon="⚠",
        )

    st.markdown("**真实数据放这里**：`data/raw/`，文件名与字段如下（支持常见中文表头自动识别）")
    _df(pd.DataFrame(
        [{"文件名": k, "必需字段": "、".join(v)} for k, v in schema_hint().items()]
    ))

    st.markdown("**行业覆盖**")
    cov = coverage_report(ds.trade)
    cov["share"] = (cov["share"] * 100).round(2)
    cov = cov.rename(columns={"industry_code": "行业代码", "industry_name": "行业名称",
                              "export_value": "出口额", "n_hs": "HS 章数", "share": "占比(%)"})
    _df(cov.head(20))

    st.markdown("**方法说明**")
    st.markdown(
        """
- **暴露度**：`Exposure_j = Σ_k (出口_kj / 出口_j) × (事件年税率 − 基期税率)`，权重取事件前一年。
- **分组**：按行业暴露度三分位划分高 / 中 / 低暴露组。
- **事件研究**：`rel_year = 年份 − 事件年`，窗口默认 ±3 年，纵轴为相对冲击前均值的偏离。
- **DID**：高暴露组与低暴露组的事前事后均值差之差，用于给一个方向性的数，不做显著性主张。
- **匹配链**：HS 编码 → 证监会行业 → 上市公司。HS6 可先截前两位归章，再接自定义映射表。
        """
    )
    st.download_button(
        "下载暴露度结果（CSV）",
        exp.to_csv(index=False).encode("utf-8-sig"),
        file_name="industry_exposure.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
