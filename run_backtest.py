"""
完整回测分析脚本
同时展示：纯技术面策略 vs ML增强策略
"""

import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("backtest")

import sys
import numpy as np
import pandas as pd

from config import DATA_CONFIG, STRATEGY_CONFIG, BACKTEST_CONFIG, ML_CONFIG
from data.fetcher import generate_synthetic_data, load_cache
from strategies.multi_factor import MultiFactorStrategy
from strategies.ml_filter import MLSignalFilter, build_features, build_labels
from backtest.engine import BacktestEngine
from backtest.metrics import compute_metrics, print_metrics_report
from backtest.visualizer import plot_backtest_results


def run_full_backtest():
    initial_capital = BACKTEST_CONFIG["initial_capital"]
    bt_engine = BacktestEngine(BACKTEST_CONFIG, STRATEGY_CONFIG)
    strategy = MultiFactorStrategy(STRATEGY_CONFIG)

    # ── 1. 加载数据（优先真实缓存，fallback仿真）──
    start = DATA_CONFIG["start_date"]
    end   = DATA_CONFIG["end_date"]
    real_count = 0
    stock_data = {}
    for i, code in enumerate(DATA_CONFIG["stock_pool"]):
        df = load_cache(code, start, end)
        if not df.empty and len(df) >= 60:
            stock_data[code] = df
            real_count += 1
        else:
            stock_data[code] = generate_synthetic_data(code, start, end, seed=42 + i * 7)
    src = f"（{real_count}只真实数据 + {len(stock_data)-real_count}只仿真数据）" if real_count else "（仿真数据）"
    logger.info(f"数据准备完成：{len(stock_data)} 只股票，每只约 {len(list(stock_data.values())[0])} 个交易日 {src}")

    # ── 2. 技术面信号生成 ──
    logger.info("计算多因子技术信号...")
    signals_dict = {}
    total_signals = 0
    for code, df in stock_data.items():
        sig_df = strategy.generate_signals(df)
        if not sig_df.empty:
            signals_dict[code] = sig_df
            cnt = (sig_df["signal"] == 1).sum()
            total_signals += cnt
    logger.info(f"技术信号汇总：{len(signals_dict)} 只股票，共 {total_signals} 个买入点")

    # ── 3. 纯技术策略回测 ──
    logger.info("=" * 50)
    logger.info("运行【纯技术面策略】回测...")
    trades_tech, equity_tech, trade_df_tech = bt_engine.run_portfolio(signals_dict)
    metrics_tech = compute_metrics(trade_df_tech, equity_tech, initial_capital)
    print_metrics_report(metrics_tech, "【纯技术面策略】组合回测结果（2020-2024）")

    # ── 4. ML过滤器训练 ──
    logger.info("=" * 50)
    logger.info("训练 XGBoost ML 信号过滤器...")
    ml_filter = MLSignalFilter(ML_CONFIG)

    # 使用前70%数据训练，后30%验证（时间序列）
    train_dict = {}
    for code, df in signals_dict.items():
        cutoff = int(len(df) * 0.7)
        train_dict[code] = df.iloc[:cutoff]

    ml_result = ml_filter.train(train_dict, forward_days=5, threshold=0.025)
    logger.info(f"ML训练完成: ACC={ml_result.get('oof_accuracy', 0):.3f}, AUC={ml_result.get('oof_auc', 0):.3f}")

    # ── 5. 在验证集上过滤信号 ──
    signals_ml = {}
    ml_filter_stats = []
    for code, df in signals_dict.items():
        try:
            filtered = ml_filter.filter_signals(df)
            before = (df["signal"] == 1).sum()
            after = (filtered["signal"] == 1).sum()
            ml_filter_stats.append({"code": code, "before": before, "after": after})
            signals_ml[code] = filtered
        except Exception as e:
            signals_ml[code] = df
            logger.warning(f"[{code}] ML过滤失败: {e}")

    filter_df = pd.DataFrame(ml_filter_stats)
    logger.info(f"ML过滤效果: 总信号 {filter_df['before'].sum()} → {filter_df['after'].sum()}")

    # ── 6. ML增强策略回测 ──
    logger.info("运行【ML增强策略】回测...")
    trades_ml, equity_ml, trade_df_ml = bt_engine.run_portfolio(signals_ml)
    if trades_ml:
        metrics_ml = compute_metrics(trade_df_ml, equity_ml, initial_capital)
        print_metrics_report(metrics_ml, "【ML增强策略】组合回测结果（2020-2024）")
    else:
        logger.warning("ML增强策略无交易记录（阈值可能过高），使用纯技术策略结果")
        metrics_ml = metrics_tech
        trade_df_ml = trade_df_tech
        equity_ml = equity_tech

    # ── 7. 单股深度分析 ──
    logger.info("=" * 50)
    logger.info("单股详细分析（前5只成交量最多的股票）...")
    if not trade_df_tech.empty:
        top_codes = (
            trade_df_tech.groupby("stock_code")["pnl"].sum()
            .nlargest(5).index.tolist()
        )
        for code in top_codes[:3]:
            if code in signals_dict:
                sub_trades, sub_equity = bt_engine.run_single(signals_dict[code], code)
                if sub_trades:
                    sub_trade_df = bt_engine._trades_to_df(sub_trades)
                    sub_metrics = compute_metrics(sub_trade_df, sub_equity, initial_capital)
                    wr = sub_metrics.get("win_rate_pct", 0)
                    ar = sub_metrics.get("annual_return_pct", 0)
                    logger.info(f"  [{code}] 胜率={wr:.1f}%  年化={ar:.1f}%  交易={sub_metrics.get('total_trades',0)}笔")

    # ── 8. 各股票胜率汇总 ──
    print("\n" + "=" * 65)
    print("  各股票回测胜率汇总（纯技术面策略）")
    print("=" * 65)
    print(f"  {'股票代码':<10} {'交易次数':>8} {'胜率':>8} {'平均收益':>10} {'总盈亏':>14}")
    print("  " + "-" * 55)

    overall_wins = 0
    overall_trades = 0
    for code in DATA_CONFIG["stock_pool"]:
        if code not in signals_dict:
            continue
        sub, sub_eq = bt_engine.run_single(signals_dict[code], code)
        if not sub:
            continue
        sub_df = bt_engine._trades_to_df(sub)
        wins = sub_df["is_win"].sum()
        total = len(sub_df)
        overall_wins += wins
        overall_trades += total
        wr = wins / total * 100
        avg_ret = sub_df["pnl_pct"].mean() * 100
        total_pnl = sub_df["pnl"].sum()
        flag = "✅" if wr >= 80 else ("⚠️" if wr >= 65 else "❌")
        print(f"  {code:<10} {total:>8}    {wr:>6.1f}%  {avg_ret:>+9.2f}%  ¥{total_pnl:>11,.0f}  {flag}")

    print("  " + "-" * 55)
    overall_wr = overall_wins / overall_trades * 100 if overall_trades > 0 else 0
    print(f"  {'合计/均值':<10} {overall_trades:>8}    {overall_wr:>6.1f}%")
    flag = "✅ 达标！" if overall_wr >= 80 else ("⚠️ 接近目标" if overall_wr >= 70 else "❌ 需优化")
    print(f"\n  综合胜率: {overall_wr:.2f}%  {flag}")
    print("=" * 65)

    # ── 9. 可视化 ──
    logger.info("生成可视化报告...")
    try:
        # 纯技术策略图
        plot_backtest_results(
            equity_tech, trade_df_tech, metrics_tech, initial_capital,
            save_path="reports/tech_strategy_report.png",
            title="A股多因子技术策略 回测报告（2020-2024）",
        )
        # ML增强策略图
        if not trade_df_ml.empty:
            plot_backtest_results(
                equity_ml, trade_df_ml, metrics_ml, initial_capital,
                save_path="reports/ml_enhanced_report.png",
                title="A股多因子+ML增强策略 回测报告（2020-2024）",
            )
    except Exception as e:
        logger.warning(f"可视化失败: {e}")

    # ── 10. 年度收益明细 ──
    print("\n" + "=" * 65)
    print("  年度收益明细（15万限额，真实子弹约束）")
    print("=" * 65)
    print(f"  {'年份':<6} {'期初资金':>10} {'期末资金':>10} {'年度收益':>8} {'年度净利':>10}")
    print("  " + "-" * 50)
    total_start = equity_tech.iloc[0]
    for yr in range(2020, 2026):
        yr_eq = equity_tech[equity_tech.index.year == yr]
        if len(yr_eq) == 0:
            continue
        yr_start = yr_eq.iloc[0]
        yr_end   = yr_eq.iloc[-1]
        yr_ret   = (yr_end / yr_start - 1) * 100
        yr_pnl   = yr_end - yr_start
        flag = "✅" if yr_ret >= 8 else ("⚠️" if yr_ret >= 0 else "❌")
        print(f"  {yr:<6} {yr_start:>10,.0f} {yr_end:>10,.0f} {yr_ret:>+7.1f}%  ¥{yr_pnl:>+9,.0f}  {flag}")
    total_ret  = (equity_tech.iloc[-1] / total_start - 1) * 100
    total_pnl  = equity_tech.iloc[-1] - total_start
    print("  " + "-" * 50)
    print(f"  {'合计':<6} {total_start:>10,.0f} {equity_tech.iloc[-1]:>10,.0f} {total_ret:>+7.1f}%  ¥{total_pnl:>+9,.0f}")
    print(f"\n  资金约束说明：初始15万，单笔固定仓位≈4.2万，最多同时持仓3只（≈12.6万）")
    print(f"  每笔亏损上限：-{STRATEGY_CONFIG['stop_loss']*100:.0f}%（≈¥{initial_capital*BACKTEST_CONFIG['position_size']*STRATEGY_CONFIG['stop_loss']:,.0f}），"
          f"止盈目标：+{STRATEGY_CONFIG['take_profit']*100:.0f}%（≈¥{initial_capital*BACKTEST_CONFIG['position_size']*STRATEGY_CONFIG['take_profit']:,.0f}）")

    # ── 11. 最终评级 ──
    print("\n" + "=" * 65)
    print("  策略综合评级")
    print("=" * 65)
    m = metrics_tech
    criteria = [
        ("胜率",       overall_wr,                    80,   "%",  "≥80%"),
        ("年化收益",   m.get("annual_return_pct", 0),  8,   "%",  "≥8%"),
        ("最大回撤",   m.get("max_drawdown_pct", 0),  15,   "%",  "≤15%", True),
        ("盈亏比",     m.get("profit_loss_ratio", 0),  0.3, "",   "≥0.3（TP/SL=3%/8%）"),
        ("夏普比率",   m.get("sharpe_ratio", 0),       0.8, "",   "≥0.8"),
    ]
    passed_cnt = 0
    for row in criteria:
        name, val, threshold, unit, desc = row[:5]
        invert = row[5] if len(row) > 5 else False
        passed = (val <= threshold) if invert else (val >= threshold)
        passed_cnt += int(passed)
        icon = "✅" if passed else "❌"
        print(f"  {icon} {name:<10}: {val:.2f}{unit}  （目标：{desc}）")

    grade = "优秀" if passed_cnt >= 4 else ("良好" if passed_cnt >= 3 else "需改进")
    print(f"\n  综合评级: {grade}（{passed_cnt}/5 项达标）")
    print("=" * 65)
    print("\n✅ 回测完成！报告保存在 reports/ 目录")

    return metrics_tech, overall_wr


if __name__ == "__main__":
    run_full_backtest()
