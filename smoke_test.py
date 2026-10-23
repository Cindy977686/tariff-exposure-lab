"""不启动界面，直接跑一遍计算链，确认每个环节都有数。"""

import sys

from core import demo_data, narrative
from core.events import EVENTS
from core.exposure import assign_groups, compute_exposure, describe, diff_in_diff, event_study, GROUP_ORDER
from core.mapping import coverage_report


def main():
    trade, tariffs, firms, meta = demo_data.generate(seed=42, n_firms=600)
    print(f"[data] trade={len(trade)} tariff={len(tariffs)} firms={len(firms)}  ({meta['source']})")

    cov = coverage_report(trade)
    print(f"[map ] {len(cov)} industries, top={cov.iloc[0]['industry_name']} {cov.iloc[0]['share']:.1%}")

    for ev in EVENTS:
        exp = compute_exposure(trade, tariffs, ev.year, base_year=ev.base_year)
        assert not exp.empty, f"{ev.name} 暴露度为空"
        f = assign_groups(firms, exp)
        groups = sorted(f["group"].unique())
        assert set(groups) <= set(GROUP_ORDER), groups

        es = event_study(f, "capex_ratio", ev.year)
        did_c = diff_in_diff(f, "capex_ratio", ev.year)
        did_k = diff_in_diff(f, "cash_ratio", ev.year)
        pre = es[es["rel_year"] < 0]["y"].abs().mean()

        print(
            f"[ev  ] {ev.name:<14} ind={len(exp):2d} "
            f"max={exp.iloc[0]['industry_name']}({exp.iloc[0]['exposure']:+.2f}) "
            f"did_capex={did_c['estimate']*100:+.2f}pp did_cash={did_k['estimate']*100:+.2f}pp "
            f"pre_abs={pre*100:.2f}pp n={did_c['n']}"
        )

        desc = describe(f, "capex_ratio")
        assert len(desc) >= 2

    exp = compute_exposure(trade, tariffs, 2018, base_year=2017)
    f = assign_groups(firms, exp)
    ctx = narrative.build_context(
        EVENTS[0], exp, diff_in_diff(f, "capex_ratio", 2018)["estimate"],
        diff_in_diff(f, "cash_ratio", 2018)["estimate"], describe(f, "capex_ratio"),
        "投资率", f["stkcd"].nunique(),
    )
    text, src = narrative.generate(ctx, use_llm=False)
    assert len(text) > 80
    print(f"\n[narr] source={src}")
    print(text[:220] + " ...")
    print("\nOK 全链路通过")


if __name__ == "__main__":
    sys.exit(main())
