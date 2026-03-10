"""
回测绩效指标计算模块

指标体系：
  胜率、盈亏比、期望值、年化收益、最大回撤、夏普比率、
  卡尔玛比率、索提诺比率、信息比率等
"""

import numpy as np
import pandas as pd
from typing import Optional


def compute_metrics(
    trades_df: pd.DataFrame,
    equity_curve: pd.Series,
    initial_capital: float,
    risk_free_rate: float = 0.02,
    benchmark: Optional[pd.Series] = None,
) -> dict:
    """
    计算全套回测绩效指标

    Parameters
    ----------
    trades_df      : 交易明细 DataFrame
    equity_curve   : 净值曲线（原始金额）
    initial_capital: 初始资金
    risk_free_rate : 年化无风险利率
    benchmark      : 基准净值序列（用于计算信息比率）

    Returns
    -------
    指标字典
    """
    if trades_df.empty:
        return _empty_metrics()

    # ── 净值归一化 ──
    nav = equity_curve / initial_capital  # 净值

    # ── 交易统计 ──
    total_trades = len(trades_df)
    win_trades = trades_df["is_win"].sum()
    loss_trades = total_trades - win_trades
    win_rate = win_trades / total_trades if total_trades > 0 else 0

    avg_win = trades_df.loc[trades_df["is_win"], "pnl_pct"].mean() if win_trades > 0 else 0
    avg_loss = trades_df.loc[~trades_df["is_win"], "pnl_pct"].mean() if loss_trades > 0 else 0
    profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    # 期望值 = 胜率 * 平均盈 - 败率 * 平均亏
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    # 最大单笔盈/亏
    max_win_pct = trades_df["pnl_pct"].max()
    max_loss_pct = trades_df["pnl_pct"].min()

    # 平均持仓天数
    avg_hold_days = trades_df["hold_days"].mean()

    # 退出原因分布
    exit_dist = trades_df["exit_reason"].value_counts().to_dict()

    # ── 资金曲线指标 ──
    total_days = len(nav)
    trading_years = total_days / 252

    # 年化收益率
    final_nav = nav.iloc[-1]
    annual_return = (final_nav ** (1 / trading_years) - 1) if trading_years > 0 else 0

    # 总收益率
    total_return = final_nav - 1

    # 日收益率
    daily_returns = nav.pct_change().dropna()

    # 年化波动率
    annual_vol = daily_returns.std() * np.sqrt(252)

    # 夏普比率
    rf_daily = risk_free_rate / 252
    excess_returns = daily_returns - rf_daily
    sharpe = (excess_returns.mean() / daily_returns.std() * np.sqrt(252)
              if daily_returns.std() > 0 else 0)

    # 索提诺比率（只考虑下行风险）
    downside_returns = daily_returns[daily_returns < rf_daily]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino = ((annual_return - risk_free_rate) / downside_std
               if downside_std > 0 else 0)

    # 最大回撤
    rolling_max = nav.cummax()
    drawdown = (nav - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    max_drawdown_pct = abs(max_drawdown)

    # 最大回撤持续时间
    dd_duration = _max_drawdown_duration(drawdown)

    # 卡尔玛比率
    calmar = annual_return / max_drawdown_pct if max_drawdown_pct > 0 else float("inf")

    # 信息比率（相对基准）
    information_ratio = _information_ratio(daily_returns, benchmark)

    # 盈利因子（总盈利 / 总亏损）
    total_profit = trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum()
    total_loss = abs(trades_df.loc[trades_df["pnl"] < 0, "pnl"].sum())
    profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

    # ── 月度统计 ──
    monthly_returns = _compute_monthly_returns(nav)
    win_months = (monthly_returns > 0).sum()
    total_months = len(monthly_returns)
    monthly_win_rate = win_months / total_months if total_months > 0 else 0

    return {
        # 胜率与盈亏
        "total_trades": int(total_trades),
        "win_trades": int(win_trades),
        "loss_trades": int(loss_trades),
        "win_rate": round(win_rate, 4),
        "win_rate_pct": round(win_rate * 100, 2),
        "avg_win_pct": round(avg_win * 100, 2),
        "avg_loss_pct": round(avg_loss * 100, 2),
        "profit_loss_ratio": round(profit_loss_ratio, 2),
        "expectancy_pct": round(expectancy * 100, 2),
        "max_win_pct": round(max_win_pct * 100, 2),
        "max_loss_pct": round(max_loss_pct * 100, 2),
        "profit_factor": round(profit_factor, 2),
        "avg_hold_days": round(avg_hold_days, 1),

        # 收益
        "total_return_pct": round(total_return * 100, 2),
        "annual_return_pct": round(annual_return * 100, 2),
        "final_equity": round(equity_curve.iloc[-1], 2),

        # 风险
        "annual_vol_pct": round(annual_vol * 100, 2),
        "max_drawdown_pct": round(max_drawdown_pct * 100, 2),
        "max_drawdown_duration_days": dd_duration,

        # 风险调整收益
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "calmar_ratio": round(calmar, 3),
        "information_ratio": round(information_ratio, 3),

        # 月度
        "monthly_win_rate_pct": round(monthly_win_rate * 100, 2),

        # 退出原因
        "exit_reasons": exit_dist,
    }


def _empty_metrics() -> dict:
    return {
        "total_trades": 0, "win_rate_pct": 0, "annual_return_pct": 0,
        "max_drawdown_pct": 0, "sharpe_ratio": 0, "message": "无交易记录",
    }


def _max_drawdown_duration(drawdown: pd.Series) -> int:
    """计算最大回撤持续天数"""
    in_dd = drawdown < 0
    if not in_dd.any():
        return 0
    max_dur = 0
    cur_dur = 0
    for v in in_dd:
        cur_dur = cur_dur + 1 if v else 0
        max_dur = max(max_dur, cur_dur)
    return max_dur


def _information_ratio(
    daily_returns: pd.Series,
    benchmark: Optional[pd.Series] = None,
) -> float:
    if benchmark is None:
        return 0.0
    bench_ret = benchmark.pct_change().dropna()
    # 对齐
    common = daily_returns.index.intersection(bench_ret.index)
    if len(common) < 10:
        return 0.0
    excess = daily_returns.loc[common] - bench_ret.loc[common]
    tracking_error = excess.std() * np.sqrt(252)
    return float((excess.mean() * 252) / tracking_error) if tracking_error > 0 else 0.0


def _compute_monthly_returns(nav: pd.Series) -> pd.Series:
    """计算月度收益率"""
    monthly = nav.resample("ME").last()
    return monthly.pct_change().dropna()


def print_metrics_report(metrics: dict, title: str = "回测绩效报告") -> None:
    """打印格式化绩效报告"""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)

    # 胜率高亮
    wr = metrics.get("win_rate_pct", 0)
    wr_flag = "✅" if wr >= 80 else ("⚠️" if wr >= 65 else "❌")

    print(f"\n📊 【交易统计】")
    print(f"  总交易次数     : {metrics.get('total_trades', 0)}")
    print(f"  胜率           : {wr:.2f}% {wr_flag}")
    print(f"  盈利次数       : {metrics.get('win_trades', 0)}")
    print(f"  亏损次数       : {metrics.get('loss_trades', 0)}")
    print(f"  平均盈利       : +{metrics.get('avg_win_pct', 0):.2f}%")
    print(f"  平均亏损       : {metrics.get('avg_loss_pct', 0):.2f}%")
    print(f"  盈亏比         : {metrics.get('profit_loss_ratio', 0):.2f}")
    print(f"  期望值         : {metrics.get('expectancy_pct', 0):.2f}%")
    print(f"  盈利因子       : {metrics.get('profit_factor', 0):.2f}")
    print(f"  最大单笔盈利   : +{metrics.get('max_win_pct', 0):.2f}%")
    print(f"  最大单笔亏损   : {metrics.get('max_loss_pct', 0):.2f}%")
    print(f"  平均持仓天数   : {metrics.get('avg_hold_days', 0):.1f} 天")

    print(f"\n💰 【收益指标】")
    print(f"  期末权益       : ¥{metrics.get('final_equity', 0):,.2f}")
    print(f"  总收益率       : {metrics.get('total_return_pct', 0):.2f}%")
    print(f"  年化收益率     : {metrics.get('annual_return_pct', 0):.2f}%")
    print(f"  月度胜率       : {metrics.get('monthly_win_rate_pct', 0):.2f}%")

    print(f"\n⚠️  【风险指标】")
    print(f"  年化波动率     : {metrics.get('annual_vol_pct', 0):.2f}%")
    print(f"  最大回撤       : -{metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"  最大回撤天数   : {metrics.get('max_drawdown_duration_days', 0)} 天")

    print(f"\n🎯 【风险调整收益】")
    sr = metrics.get("sharpe_ratio", 0)
    sr_flag = "✅" if sr >= 1.5 else ("⚠️" if sr >= 1.0 else "❌")
    print(f"  夏普比率       : {sr:.3f} {sr_flag}")
    print(f"  索提诺比率     : {metrics.get('sortino_ratio', 0):.3f}")
    print(f"  卡尔玛比率     : {metrics.get('calmar_ratio', 0):.3f}")
    print(f"  信息比率       : {metrics.get('information_ratio', 0):.3f}")

    exit_reasons = metrics.get("exit_reasons", {})
    if exit_reasons:
        print(f"\n🚪 【退出原因分布】")
        for reason, cnt in sorted(exit_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason:<20}: {cnt}")

    print(f"\n{sep}\n")
