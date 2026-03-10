"""
回测可视化模块 - 生成专业的绩效报告图表
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# 设置中文字体（Linux 兼容）
plt.rcParams["font.family"] = ["DejaVu Sans", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def plot_backtest_results(
    equity_curve: pd.Series,
    trades_df: pd.DataFrame,
    metrics: dict,
    initial_capital: float,
    save_path: str = "reports/backtest_report.png",
    title: str = "A股量化策略回测报告",
) -> None:
    """
    生成4格综合回测报告图
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    nav = equity_curve / initial_capital
    daily_returns = nav.pct_change().dropna()

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    # ── 图1：净值曲线 + 回撤 ──
    ax1 = fig.add_subplot(3, 2, (1, 2))
    ax1.plot(nav.index, nav.values, color="#1f77b4", linewidth=1.5, label="策略净值")
    ax1.axhline(1.0, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

    rolling_max = nav.cummax()
    drawdown = (nav - rolling_max) / rolling_max
    ax1_twin = ax1.twinx()
    ax1_twin.fill_between(drawdown.index, drawdown.values, 0,
                           color="#d62728", alpha=0.25, label="回撤")
    ax1_twin.set_ylabel("回撤", color="#d62728", fontsize=9)
    ax1_twin.tick_params(axis="y", labelcolor="#d62728")
    ax1_twin.set_ylim(-0.5, 0.05)

    ax1.set_title("净值曲线 & 回撤", fontsize=12, pad=8)
    ax1.set_ylabel("净值", fontsize=9)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # ── 图2：月度收益热力图 ──
    ax2 = fig.add_subplot(3, 2, 3)
    monthly = nav.resample("ME").last().pct_change().dropna() * 100
    if not monthly.empty:
        monthly_df = monthly.reset_index()
        monthly_df.columns = ["date", "ret"]
        monthly_df["year"] = monthly_df["date"].dt.year
        monthly_df["month"] = monthly_df["date"].dt.month
        pivot = monthly_df.pivot(index="year", columns="month", values="ret")
        pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:len(pivot.columns)]

        cmap = plt.cm.RdYlGn
        vmax = max(abs(pivot.values[~np.isnan(pivot.values)]).max(), 5)
        im = ax2.imshow(pivot.values, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        ax2.set_xticks(range(len(pivot.columns)))
        ax2.set_xticklabels(pivot.columns, fontsize=7)
        ax2.set_yticks(range(len(pivot.index)))
        ax2.set_yticklabels(pivot.index.astype(str), fontsize=8)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax2.text(j, i, f"{val:.1f}%", ha="center", va="center",
                             fontsize=6.5, color="black" if abs(val) < vmax * 0.6 else "white")
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    ax2.set_title("月度收益热力图 (%)", fontsize=11, pad=8)

    # ── 图3：收益分布 ──
    ax3 = fig.add_subplot(3, 2, 4)
    if not trades_df.empty:
        pnl_pcts = trades_df["pnl_pct"] * 100
        wins = pnl_pcts[pnl_pcts > 0]
        losses = pnl_pcts[pnl_pcts <= 0]
        ax3.hist(wins.values, bins=25, color="#2ca02c", alpha=0.7, label=f"盈利({len(wins)}笔)")
        ax3.hist(losses.values, bins=25, color="#d62728", alpha=0.7, label=f"亏损({len(losses)}笔)")
        ax3.axvline(0, color="black", linewidth=1.2)
        ax3.axvline(pnl_pcts.mean(), color="navy", linewidth=1.2,
                    linestyle="--", label=f"均值 {pnl_pcts.mean():.2f}%")
        ax3.legend(fontsize=8)
    ax3.set_title("单笔收益分布", fontsize=11, pad=8)
    ax3.set_xlabel("收益率 (%)", fontsize=9)
    ax3.set_ylabel("频次", fontsize=9)
    ax3.grid(True, alpha=0.3)

    # ── 图4：累计收益曲线（按时间排序的交易） ──
    ax4 = fig.add_subplot(3, 2, 5)
    if not trades_df.empty and "exit_date" in trades_df.columns:
        sorted_trades = trades_df.sort_values("exit_date").copy()
        sorted_trades["cum_pnl"] = sorted_trades["pnl"].cumsum()
        colors = ["#2ca02c" if w else "#d62728" for w in sorted_trades["is_win"]]
        ax4.bar(range(len(sorted_trades)), sorted_trades["pnl_pct"] * 100,
                color=colors, alpha=0.7, width=0.8)
        ax4.axhline(0, color="black", linewidth=0.8)
    ax4.set_title("逐笔交易收益 (%)", fontsize=11, pad=8)
    ax4.set_xlabel("交易序号", fontsize=9)
    ax4.set_ylabel("收益率 (%)", fontsize=9)
    ax4.grid(True, alpha=0.3)

    # ── 图5：关键指标文本卡片 ──
    ax5 = fig.add_subplot(3, 2, 6)
    ax5.axis("off")
    wr = metrics.get("win_rate_pct", 0)
    sr = metrics.get("sharpe_ratio", 0)
    md = metrics.get("max_drawdown_pct", 0)
    ar = metrics.get("annual_return_pct", 0)

    info_text = (
        f"{'='*38}\n"
        f"  关键绩效指标汇总\n"
        f"{'='*38}\n"
        f"  胜率          : {wr:.2f}%  {'✓' if wr >= 80 else '△'}\n"
        f"  年化收益率    : {ar:.2f}%\n"
        f"  最大回撤      : -{md:.2f}%\n"
        f"  夏普比率      : {sr:.3f}  {'✓' if sr >= 1.5 else '△'}\n"
        f"  卡尔玛比率    : {metrics.get('calmar_ratio', 0):.3f}\n"
        f"  盈亏比        : {metrics.get('profit_loss_ratio', 0):.2f}\n"
        f"  期望值        : {metrics.get('expectancy_pct', 0):.2f}%\n"
        f"  盈利因子      : {metrics.get('profit_factor', 0):.2f}\n"
        f"  总交易次数    : {metrics.get('total_trades', 0)}\n"
        f"  平均持仓天数  : {metrics.get('avg_hold_days', 0):.1f}天\n"
        f"  月度胜率      : {metrics.get('monthly_win_rate_pct', 0):.2f}%\n"
        f"{'='*38}\n"
        f"  初始资金: ¥{initial_capital/10000:.0f}万\n"
        f"  期末权益: ¥{metrics.get('final_equity', 0)/10000:.2f}万\n"
    )
    ax5.text(0.05, 0.95, info_text, transform=ax5.transAxes,
             fontsize=9.5, verticalalignment="top",
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f4ff",
                       edgecolor="#334d80", linewidth=1.5))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"[可视化] 报告已保存: {save_path}")
