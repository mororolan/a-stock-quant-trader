"""
A股量化交易系统 - 主入口

使用方式：
  python main.py                     # 使用仿真数据全量回测
  python main.py --use-real-data     # 尝试从 akshare 获取真实数据
  python main.py --no-ml             # 不使用 ML 过滤器（纯技术面策略）
  python main.py --single 600519     # 单只股票深度分析
"""

import sys
import logging
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── 日志设置 ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

from config import DATA_CONFIG, STRATEGY_CONFIG, BACKTEST_CONFIG, ML_CONFIG
from data.fetcher import fetch_stock_pool, generate_synthetic_data
from strategies.multi_factor import MultiFactorStrategy
from strategies.ml_filter import MLSignalFilter
from backtest.engine import BacktestEngine
from backtest.metrics import compute_metrics, print_metrics_report
from backtest.visualizer import plot_backtest_results


def parse_args():
    parser = argparse.ArgumentParser(description="A股量化交易回测系统")
    parser.add_argument("--use-real-data", action="store_true",
                        help="从 akshare 获取真实A股数据（需要网络）")
    parser.add_argument("--no-ml", action="store_true",
                        help="禁用 XGBoost ML 信号过滤器")
    parser.add_argument("--single", type=str, default=None,
                        help="单只股票代码进行深度回测（如 600519）")
    parser.add_argument("--start", type=str, default=DATA_CONFIG["start_date"],
                        help="回测开始日期（YYYY-MM-DD）")
    parser.add_argument("--end", type=str, default=DATA_CONFIG["end_date"],
                        help="回测结束日期（YYYY-MM-DD）")
    return parser.parse_args()


def load_data(args) -> dict[str, pd.DataFrame]:
    """加载数据（真实 or 仿真）"""
    start = args.start
    end = args.end

    if args.use_real_data:
        logger.info("正在从 akshare 获取真实A股数据...")
        pool = [args.single] if args.single else DATA_CONFIG["stock_pool"]
        stock_data = fetch_stock_pool(pool, start, end, DATA_CONFIG["cache_dir"])
        if not stock_data:
            logger.warning("真实数据获取失败，退回到仿真数据")
            args.use_real_data = False

    if not args.use_real_data:
        logger.info("使用仿真数据（GBM + 跳扩散模型，模拟A股特性）")
        pool = [args.single] if args.single else DATA_CONFIG["stock_pool"]
        stock_data = {}
        for i, code in enumerate(pool):
            df = generate_synthetic_data(code, start, end, seed=42 + i)
            stock_data[code] = df
            logger.info(f"[{code}] 仿真数据生成完成，{len(df)} 行")

    return stock_data


def run_strategy_signals(
    stock_data: dict[str, pd.DataFrame],
    use_ml: bool = True,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """
    对所有股票计算策略信号（多因子 + ML过滤）
    """
    strategy = MultiFactorStrategy(STRATEGY_CONFIG)
    ml_filter = MLSignalFilter(ML_CONFIG) if use_ml else None

    # ── 第一步：计算技术面信号 ──
    logger.info("正在计算多因子技术信号...")
    signals_dict = {}
    indicators_dict = {}

    for code, df in stock_data.items():
        try:
            df_signal = strategy.generate_signals(df)
            if df_signal.empty:
                continue
            signals_dict[code] = df_signal
            indicators_dict[code] = df_signal
            buy_cnt = (df_signal["signal"] == 1).sum()
            logger.info(f"[{code}] 技术信号: {buy_cnt} 个买入点")
        except Exception as e:
            logger.error(f"[{code}] 信号计算失败: {e}")

    # ── 第二步：ML 训练 + 过滤 ──
    ml_result = {}
    if use_ml and ml_filter and len(indicators_dict) >= 3:
        logger.info("训练 XGBoost ML 信号过滤器...")
        try:
            ml_result = ml_filter.train(
                indicators_dict,
                forward_days=5,
                threshold=0.025,
            )
            logger.info("ML过滤器训练完成，对信号进行置信度过滤...")

            for code in list(signals_dict.keys()):
                try:
                    filtered = ml_filter.filter_signals(signals_dict[code])
                    before = (signals_dict[code]["signal"] == 1).sum()
                    after = (filtered["signal"] == 1).sum()
                    logger.info(f"[{code}] ML过滤: {before} → {after} 个买入信号")
                    signals_dict[code] = filtered
                except Exception as e:
                    logger.warning(f"[{code}] ML过滤失败，保留原信号: {e}")

            # 特征重要性
            if ml_filter.feature_importance_ is not None:
                print("\n📊 ML特征重要性 Top 10:")
                print(ml_filter.feature_importance_.head(10).to_string())
        except Exception as e:
            logger.error(f"ML训练失败: {e}")
    elif use_ml:
        logger.info("股票数量不足，跳过 ML 过滤")

    return signals_dict, ml_result


def single_stock_backtest(
    code: str,
    df_signal: pd.DataFrame,
    bt_engine: BacktestEngine,
    initial_capital: float,
    save_prefix: str = "",
) -> dict:
    """单只股票深度回测"""
    trades, equity_curve = bt_engine.run_single(df_signal, code)

    if not trades:
        print(f"[{code}] 无交易记录")
        return {}

    trade_df = bt_engine._trades_to_df(trades)
    metrics = compute_metrics(
        trade_df, equity_curve, initial_capital
    )

    print_metrics_report(metrics, f"[{code}] 单股回测报告")

    # 可视化
    save_path = f"reports/{save_prefix}{code}_backtest.png"
    try:
        plot_backtest_results(
            equity_curve, trade_df, metrics, initial_capital,
            save_path=save_path,
            title=f"股票 {code} 多因子策略回测报告",
        )
    except Exception as e:
        logger.warning(f"可视化失败: {e}")

    return metrics


def main():
    args = parse_args()

    print("\n" + "=" * 65)
    print("   A股量化交易系统 v2.0")
    print("   策略：多因子三重确认 + XGBoost ML信号过滤")
    print("=" * 65)

    use_ml = not args.no_ml

    # ── 1. 加载数据 ──
    stock_data = load_data(args)
    if not stock_data:
        logger.error("没有有效的股票数据，退出")
        sys.exit(1)

    initial_capital = BACKTEST_CONFIG["initial_capital"]
    bt_engine = BacktestEngine(BACKTEST_CONFIG, STRATEGY_CONFIG)

    # ── 2. 计算信号 ──
    signals_dict, ml_result = run_strategy_signals(stock_data, use_ml=use_ml)
    if not signals_dict:
        logger.error("没有产生有效信号，退出")
        sys.exit(1)

    # ── 3. 单只股票模式 ──
    if args.single:
        code = args.single
        if code not in signals_dict:
            logger.error(f"股票 {code} 没有信号数据")
            sys.exit(1)
        single_stock_backtest(code, signals_dict[code], bt_engine, initial_capital)
        return

    # ── 4. 组合回测 ──
    logger.info(f"开始组合回测，共 {len(signals_dict)} 只股票...")
    all_trades, portfolio_equity, trade_df = bt_engine.run_portfolio(signals_dict)

    if not all_trades:
        logger.error("组合回测无交易记录")
        sys.exit(1)

    # ── 5. 绩效分析 ──
    metrics = compute_metrics(
        trade_df, portfolio_equity, initial_capital
    )
    print_metrics_report(metrics, "A股多因子组合策略 回测报告")

    # ── 6. 按股票分析 ──
    print("\n📈 各股票绩效汇总:")
    print("-" * 75)
    stock_stats = []
    for code in signals_dict:
        sub_trades = trade_df[trade_df["stock_code"] == code]
        if sub_trades.empty:
            continue
        wins = sub_trades["is_win"].sum()
        total = len(sub_trades)
        wr = wins / total * 100
        avg_ret = sub_trades["pnl_pct"].mean() * 100
        total_pnl = sub_trades["pnl"].sum()
        stock_stats.append({
            "股票": code,
            "交易次数": total,
            "胜率%": f"{wr:.1f}",
            "平均收益%": f"{avg_ret:.2f}",
            "总盈亏": f"¥{total_pnl:,.0f}",
        })

    stats_df = pd.DataFrame(stock_stats)
    if not stats_df.empty:
        print(stats_df.sort_values("总盈亏", ascending=False).to_string(index=False))

    # ── 7. 策略关键指标总结 ──
    wr = metrics.get("win_rate_pct", 0)
    ar = metrics.get("annual_return_pct", 0)
    md = metrics.get("max_drawdown_pct", 0)
    sr = metrics.get("sharpe_ratio", 0)

    print("\n" + "=" * 65)
    print("  策略评级")
    print("=" * 65)

    criteria = [
        ("胜率", wr, 80, "%", "≥80% 为目标"),
        ("年化收益", ar, 15, "%", "≥15% 超越市场"),
        ("最大回撤", md, 20, "%", "≤20% 风控达标（越小越好）", True),
        ("夏普比率", sr, 1.5, "", "≥1.5 优秀", False),
    ]

    all_pass = True
    for name, val, threshold, unit, desc, *inv in criteria:
        invert = inv[0] if inv else False
        passed = (val <= threshold) if invert else (val >= threshold)
        all_pass = all_pass and passed
        icon = "✅" if passed else "❌"
        print(f"  {icon} {name:<10}: {val:.2f}{unit}  （{desc}）")

    if all_pass:
        print("\n  🏆 策略全面达标！可进入实盘验证阶段")
    else:
        print("\n  ⚠️  部分指标未达标，建议继续优化")

    # ── 8. ML 训练结果 ──
    if ml_result:
        print(f"\n🤖 ML 过滤器性能:")
        print(f"  训练样本数  : {ml_result.get('n_samples', 0)}")
        print(f"  OOF 准确率  : {ml_result.get('oof_accuracy', 0):.3f}")
        print(f"  OOF AUC     : {ml_result.get('oof_auc', 0):.3f}")
        print(f"  正例精确率  : {ml_result.get('oof_precision_1', 0):.3f}")

    # ── 9. 可视化 ──
    try:
        plot_backtest_results(
            portfolio_equity, trade_df, metrics, initial_capital,
            save_path="reports/portfolio_backtest_report.png",
            title="A股多因子策略 组合回测报告（2020-2024）",
        )
    except Exception as e:
        logger.warning(f"可视化生成失败: {e}")

    print("\n✅ 回测完成！查看 reports/ 目录获取详细图表报告")


if __name__ == "__main__":
    main()
