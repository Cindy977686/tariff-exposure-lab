"""解读层：把数字变成人话。

默认走模板生成——所有数字都来自实际计算结果，不会编造，现场演示最安全。
配置了 HiAgent 或任意 OpenAI 兼容接口后可以切成大模型生成，失败自动回落。
"""

import json
import os
import urllib.error
import urllib.request

SYSTEM_PROMPT = (
    "你是一名经贸与公司金融方向的研究助理。你只能使用用户给出的数字，"
    "不得引入任何未提供的统计数据、公司名称或政策细节。"
    "如果某项数字缺失，就明确说该项尚未计算。输出简体中文，分三段："
    "第一段说冲击落在哪里；第二段说企业怎么反应；第三段给一句方法上的保留意见。"
    "每段不超过 80 字，不要用列表，不要加标题。"
)


def build_context(event, exposure, did_capex, did_cash, group_desc, outcome_label, n_firms):
    top = exposure.head(3)
    ctx = {
        "event_name": event.name,
        "event_desc": event.description,
        "year": event.year,
        "channel": event.channel,
        "n_industries": int(exposure["industry_code"].nunique()) if not exposure.empty else 0,
        "n_firms": int(n_firms),
        "outcome_label": outcome_label,
        "top_industries": [
            {"name": r.industry_name, "exposure": round(float(r.exposure), 3)}
            for r in top.itertuples()
        ],
        "did_capex": None if did_capex is None else round(float(did_capex), 4),
        "did_cash": None if did_cash is None else round(float(did_cash), 4),
        "groups": [],
    }
    if group_desc is not None and not group_desc.empty:
        for r in group_desc.itertuples():
            ctx["groups"].append({
                "group": r.group,
                "firms": int(r.firms),
                "mean": round(float(r.mean), 4),
            })
    return ctx


def _pct(x):
    if x is None:
        return "尚未计算"
    return f"{x * 100:+.2f} 个百分点"


def template_text(ctx) -> str:
    top = ctx["top_industries"]
    top_txt = "、".join(f"{t['name']}（{t['exposure']:+.2f} 个百分点）" for t in top) or "暂无"
    groups = ctx["groups"]
    g_txt = "；".join(f"{g['group']}组均值 {g['mean'] * 100:.2f}%" for g in groups) or "暂无分组结果"

    p1 = (
        f"以 {ctx['event_name']} 为冲击时点，按基期出口权重加权的关税变动计算，"
        f"暴露度最高的行业是 {top_txt}。覆盖 {ctx['n_industries']} 个行业、{ctx['n_firms']} 家公司。"
    )
    p2 = (
        f"高暴露组相对低暴露组，{ctx['outcome_label']}在冲击后平均变化 {_pct(ctx['did_capex'])}，"
        f"现金持有变化 {_pct(ctx['did_cash'])}。分组水平为：{g_txt}。"
        f"如果投资下降而现金上升同时出现，通常指向预防性储蓄而非单纯的需求收缩。"
    )
    p3 = (
        "需要保留的是：暴露度按行业层面构造，组内企业同值，标准误若不做行业聚类会偏小；"
        "同时该窗口内可能存在其他政策叠加，因果解释应限于相关性表述。"
    )
    return "\n\n".join([p1, p2, p3])


def llm_text(ctx, api_base: str, api_key: str, model: str, timeout: int = 30):
    """调用 OpenAI 兼容接口。失败返回 (None, 错误信息)。"""
    if not api_base or not api_key:
        return None, "未配置接口"

    url = api_base.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"

    payload = {
        "model": model or "deepseek-chat",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(ctx, ensure_ascii=False, indent=2)},
        ],
        "temperature": 0.2,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip(), None
    except urllib.error.HTTPError as e:
        return None, f"接口返回 {e.code}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def generate(ctx, use_llm: bool = False, api_base: str = "", api_key: str = "", model: str = ""):
    """返回 (文本, 来源标签)。"""
    if use_llm:
        text, err = llm_text(ctx, api_base, api_key, model)
        if text:
            return text, f"大模型（{model or 'default'}）"
        fallback = template_text(ctx)
        return fallback, f"模板（大模型不可用：{err}）"
    return template_text(ctx), "模板"


def env_config():
    return {
        "api_base": os.environ.get("TEL_API_BASE", ""),
        "api_key": os.environ.get("TEL_API_KEY", ""),
        "model": os.environ.get("TEL_MODEL", ""),
    }
