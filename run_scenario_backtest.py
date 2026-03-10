"""
场景对比回测：2025牛市 vs 2026初高波动

说明：由于外网不通无法获取真实行情，使用调参仿真数据模拟两种市场体制：
  - 2025牛市：强趋势、低波动、高胜率预期
  - 2026初波动：高波动、体制频繁切换、测试策略抗波动性
"""

import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("scenario")

import numpy as np
import pandas as pd

from config import DATA_CONFIG, STRATEGY_CONFIG, BACKTEST_CONFIG
from strategies.multi_factor import MultiFactorStrategy
from backtest.engine import BacktestEngine
from backtest.metrics import compute_metrics, print_metrics_report
from data.fetcher import load_cache


def generate_scenario_data(
    stock_code: str,
    start_date: str,
    end_date: str,
    scenario: str = "bull",
    seed: int = 42,
) -> pd.DataFrame:
    """
    生成场景化仿真数据

    scenario:
      "bull_2025"  - 2025牛市：强趋势，μ=+0.15%/日，低波动，牛市主导（持续180日+）
      "volatile_2026" - 2026初波动：高波动σ=2.5%，频繁体制切换，散户情绪剧烈
    """
    rng = np.random.default_rng(seed + sum(ord(c) for c in stock_code) % 997)
    dates = pd.bdate_range(start=start_date, end=end_date, freq="B")
    n = len(dates)

    if scenario == "bull_2025":
        # ── 2025牛市参数 ──
        # 真实2025：DeepSeek行情带动科技股，沪指从3200→3400+，局部标的+30%
        bull_mu = 0.0015       # +0.15%/日 ≈ +38%/年（强牛）
        bull_sigma = 0.012     # 1.2%/日（低波动稳涨）
        bear_mu = -0.0003      # 熊市也较温和（整体向上）
        bear_sigma = 0.015
        p_bull_to_bear = 1/180  # 牛市平均持续180天（半年）
        p_bear_to_bull = 1/20   # 熊市短暂（20天快速切回）
        rho = 0.30             # 强动量（追涨惯性强）
        jump_prob = 0.015
        jump_mean_bull = 0.020  # 牛市暴涨事件（+2%跳）
        jump_mean_bear = -0.010
        jump_std = 0.020
        initial_price = 100.0

    elif scenario == "volatile_2026":
        # ── 2026初波动参数 ──
        # 真实2026初：政策预期分歧大，关税战扰动，体制频繁切换
        bull_mu = 0.0008       # +0.08%/日（趋势较弱）
        bull_sigma = 0.022     # 2.2%/日（高波动）
        bear_mu = -0.0010      # 更深的熊市阶段
        bear_sigma = 0.028     # 熊市波动更高
        p_bull_to_bear = 1/30  # 牛市仅持续30天（频繁切换）
        p_bear_to_bull = 1/25  # 熊市也短（双向剧烈）
        rho = 0.20             # 动量较弱（方向不明）
        jump_prob = 0.035      # 更频繁的跳扩散（政策消息冲击）
        jump_mean_bull = 0.015
        jump_mean_bear = -0.025  # 熊市暴跌事件更猛
        jump_std = 0.030
        initial_price = 100.0

    else:
        raise ValueError(f"未知场景: {scenario}")

    # ── 体制切换（马尔可夫链）──
    regimes = np.zeros(n, dtype=int)
    regimes[0] = 1
    for i in range(1, n):
        if regimes[i - 1] == 1:
            regimes[i] = 0 if rng.random() < p_bull_to_bear else 1
        else:
            regimes[i] = 1 if rng.random() < p_bear_to_bull else 0

    bull_days = regimes.sum()
    logger.debug(f"[{stock_code}] {scenario}: 牛市天数={bull_days}/{n} ({bull_days/n*100:.0f}%)")

    # ── 收益率（AR(1)动量 + 跳扩散）──
    z = rng.standard_normal(n)
    momentum = np.zeros(n)
    for i in range(1, n):
        momentum[i] = rho * momentum[i - 1] + np.sqrt(1 - rho ** 2) * z[i]

    jumps = np.where(
        rng.random(n) < jump_prob,
        np.where(regimes == 1,
                 rng.normal(jump_mean_bull, jump_std, n),
                 rng.normal(jump_mean_bear, jump_std, n)),
        0,
    )

    mu_t = np.where(regimes == 1, bull_mu, bear_mu)
    sigma_t = np.where(regimes == 1, bull_sigma, bear_sigma)
    log_returns = mu_t + sigma_t * momentum + jumps
    log_returns = np.clip(log_returns, -0.10, 0.10)  # 涨跌停

    close = initial_price * np.exp(np.cumsum(log_returns))

    daily_vol_pct = sigma_t + 0.005
    high = close * (1 + daily_vol_pct * rng.beta(2, 5, n))
    low  = close * (1 - daily_vol_pct * rng.beta(2, 5, n))
    open_ = close.copy()
    open_[1:] = close[:-1] * (1 + rng.normal(0, 0.005, n - 1))

    base_vol = 5_000_000
    regime_vol_factor = np.where(regimes == 1, 1.5, 0.7)
    price_chg_factor = 1 + 3 * np.abs(log_returns)
    volume = (base_vol * regime_vol_factor * price_chg_factor *
              rng.lognormal(0, 0.3, n)).astype(int)

    df = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume, "amount": volume * close,
        "turnover": volume / 1e8 * 100,
        "pct_chg": pd.Series(close).pct_change().fillna(0).values * 100,
        "regime": regimes,
    }, index=dates)
    df.index.name = "date"
    return df.round(4)


def run_scenario(scenario_name: str, start: str, end: str, scenario_key: str) -> dict:
    """
    运行单个场景回测

    策略指标需要至少120日预热（MA120, ATR等），因此数据从 start 往前延伸
    120个交易日（约6个月），但只在 [start, end] 窗口内回测交易。
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"  场景：{scenario_name}  [{start} → {end}]")
    logger.info(f"{'='*60}")

    stock_pool = DATA_CONFIG["stock_pool"]
    strategy = MultiFactorStrategy(STRATEGY_CONFIG)
    engine = BacktestEngine(BACKTEST_CONFIG, STRATEGY_CONFIG)

    # 预热期：往前延伸 280 个交易日（含52周高低点等指标），确保所有指标无NaN
    warmup_start = (pd.Timestamp(start) - pd.offsets.BDay(280)).strftime("%Y-%m-%d")

    # 生成数据（预热 + 目标区间）：优先读真实缓存，无缓存则用仿真
    stock_data = {}
    real_count = 0
    for i, code in enumerate(stock_pool):
        real = load_cache(code, warmup_start, end)
        if not real.empty and len(real) >= 280:
            stock_data[code] = real
            real_count += 1
        else:
            stock_data[code] = generate_scenario_data(
                code, warmup_start, end, scenario=scenario_key, seed=42 + i * 7
            )
    if real_count > 0:
        logger.info(f"  真实数据：{real_count} 只，仿真数据：{len(stock_pool)-real_count} 只")

    # 生成信号（全区间计算指标）
    signals_dict_full = {}
    for code, df in stock_data.items():
        if len(df) < 60:
            continue
        sig_df = strategy.generate_signals(df)
        if not sig_df.empty:
            signals_dict_full[code] = sig_df

    # 裁剪到目标区间（只回测 [start, end]）
    signals_dict = {}
    total_signals = 0
    for code, sig_df in signals_dict_full.items():
        sig_window = sig_df.loc[start:end]
        if not sig_window.empty:
            signals_dict[code] = sig_window
            total_signals += (sig_window["signal"] == 1).sum()

    logger.info(f"信号汇总：{len(signals_dict)} 只股票，共 {total_signals} 个买入点")

    if not signals_dict:
        logger.warning("无有效信号，跳过")
        return {}

    # 回测
    initial_capital = BACKTEST_CONFIG["initial_capital"]
    trades, equity, trade_df = engine.run_portfolio(signals_dict)
    metrics = compute_metrics(trade_df, equity, initial_capital)

    print_metrics_report(metrics, f"【{scenario_name}】组合回测结果")

    # 各股票胜率
    overall_wins, overall_total = 0, 0
    stock_results = []
    for code in stock_pool:
        if code not in signals_dict:
            continue
        sub_trades, sub_eq = engine.run_single(signals_dict[code], code)
        if not sub_trades:
            continue
        sub_df = engine._trades_to_df(sub_trades)
        w = sub_df["is_win"].sum()
        t = len(sub_df)
        overall_wins += w
        overall_total += t
        stock_results.append({
            "code": code,
            "trades": t,
            "win_rate": w / t * 100,
            "avg_ret": sub_df["pnl_pct"].mean() * 100,
            "total_pnl": sub_df["pnl"].sum(),
        })

    overall_wr = overall_wins / overall_total * 100 if overall_total > 0 else 0

    print(f"\n  各股胜率（{scenario_name}）")
    print(f"  {'代码':<8} {'次数':>4} {'胜率':>7} {'均收益':>8} {'总盈亏':>12}")
    print("  " + "-"*45)
    for r in sorted(stock_results, key=lambda x: -x["win_rate"])[:10]:
        flag = "✅" if r["win_rate"] >= 80 else ("⚠️" if r["win_rate"] >= 65 else "❌")
        print(f"  {r['code']:<8} {r['trades']:>4}  {r['win_rate']:>6.1f}%  {r['avg_ret']:>+7.2f}%  ¥{r['total_pnl']:>10,.0f} {flag}")
    print("  " + "-"*45)
    print(f"  综合胜率: {overall_wr:.2f}%  总交易: {overall_total}笔")

    metrics["overall_win_rate"] = overall_wr
    metrics["total_trades_all_stocks"] = overall_total
    metrics["scenario"] = scenario_name
    return metrics


def main():
    print("\n" + "="*65)
    print("  A股回测：场景对比分析")
    print("  基准策略：回调反弹（TP=3%, SL=8%, 仓位28%）")
    print("="*65)

    # 场景1：历史基准（2020-2024，已知结果）
    # 场景2：2025牛市（2025-01-01 → 2025-02-28）
    # 场景3：2026初波动（2026-01-01 → 2026-02-28）

    results = {}

    # 2025 1-2月（牛市）
    results["bull_2025"] = run_scenario(
        "2025年1-2月（牛市）",
        "2025-01-01", "2025-02-28",
        scenario_key="bull_2025"
    )

    # 2026 1-2月（高波动）
    results["volatile_2026"] = run_scenario(
        "2026年1-2月（高波动）",
        "2026-01-01", "2026-02-28",
        scenario_key="volatile_2026"
    )

    # 对比汇总
    print("\n" + "="*65)
    print("  场景对比汇总")
    print("="*65)
    print(f"  {'指标':<18} {'2025牛市（1-2月）':>18} {'2026初波动（1-2月）':>20}")
    print("  " + "-"*58)

    def fmt(m, key, pct=True):
        v = m.get(key, 0)
        return f"{v:+.2f}%" if pct else f"{v:.2f}"

    b = results.get("bull_2025", {})
    v = results.get("volatile_2026", {})

    rows = [
        ("总交易次数", "total_trades", False),
        ("综合胜率", "overall_win_rate", True),
        ("组合胜率", "win_rate_pct", True),
        ("年化收益率", "annual_return_pct", True),
        ("总收益率", "total_return_pct", True),
        ("最大回撤", "max_drawdown_pct", True),
        ("夏普比率", "sharpe_ratio", False),
        ("索提诺比率", "sortino_ratio", False),
        ("期望值/笔", "expectancy_pct", True),
        ("盈利因子", "profit_factor", False),
    ]

    for name, key, is_pct in rows:
        bv = b.get(key, 0)
        vv = v.get(key, 0)
        bs = f"{bv:+.2f}%" if is_pct else f"{bv:.3f}"
        vs = f"{vv:+.2f}%" if is_pct else f"{vv:.3f}"
        print(f"  {name:<18} {bs:>18} {vs:>20}")

    print("  " + "-"*58)
    print(f"\n  关键结论：")

    b_wr = b.get("win_rate_pct", 0)
    v_wr = v.get("win_rate_pct", 0)
    b_ar = b.get("annual_return_pct", 0)
    v_ar = v.get("annual_return_pct", 0)

    if b_wr >= 80:
        print(f"  ✅ 牛市胜率 {b_wr:.1f}% — 策略在趋势行情中表现优异")
    else:
        print(f"  ⚠️  牛市胜率 {b_wr:.1f}% — 低于预期（可能样本量太少）")

    if v_wr >= 75:
        print(f"  ✅ 高波动胜率 {v_wr:.1f}% — 回调策略对波动有一定抗性")
    elif v_wr >= 65:
        print(f"  ⚠️  高波动胜率 {v_wr:.1f}% — 略有下降，属正常衰退")
    else:
        print(f"  ❌ 高波动胜率 {v_wr:.1f}% — 需谨慎，波动市场策略失效风险高")

    if b_ar > v_ar:
        diff = b_ar - v_ar
        print(f"  📊 牛市年化比波动市高 {diff:.1f}pct — 策略具有明显的趋势偏好")

    print(f"\n  ⚠️  注：基于仿真数据，真实市场结果会有偏差")
    print(f"       建议：真实数据接入后(tushare/akshare)重跑本脚本验证")
    print("="*65)


if __name__ == "__main__":
    main()
