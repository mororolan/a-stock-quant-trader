"""
每日交易信号扫描器

使用方式：
  # 步骤1：指定今日关注板块（LLM分析 或 手动）
  python run_daily_scanner.py --sectors AI与科技 金融

  # 步骤2：提供真实行情数据（CSV格式）或使用仿真数据
  python run_daily_scanner.py --sectors AI与科技 --data-dir data/cache

输出：
  明日建议操作：
  ┌─────────────────────────────────────────────────────────┐
  │  买入信号：海康威视 (002415)                              │
  │  当前价：约47.2元  建议买入价：≤47.5元（收盘附近）         │
  │  止盈价：48.75元（+3%）  止损价：43.62元（-8%）           │
  │  建议仓位：28% ≈ 4.2万（约886股，按100股整数倍）          │
  │  信号强度：★★★★☆  理由：MA60趋势+RSI从低位反弹+放量      │
  └─────────────────────────────────────────────────────────┘
"""

import warnings
warnings.filterwarnings("ignore")
import os
import sys
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("scanner")

from config import SECTOR_STOCK_POOL, STRATEGY_CONFIG, ACCOUNT_CONFIG
from strategies.multi_factor import MultiFactorStrategy
from strategies.indicators import compute_all_indicators
from data.fetcher import generate_synthetic_data


# ────────────── 数据加载 ──────────────

def load_stock_data_csv(stock_code: str, data_dir: str) -> pd.DataFrame:
    """从 CSV/Parquet 文件加载行情数据"""
    for ext in [".parquet", ".csv"]:
        p = Path(data_dir) / f"{stock_code}{ext}"
        if p.exists():
            if ext == ".parquet":
                return pd.read_parquet(p)
            else:
                df = pd.read_csv(p, parse_dates=["date"], index_col="date")
                return df
    return pd.DataFrame()


def load_or_simulate(
    stock_code: str,
    data_dir: str,
    start_date: str,
    end_date: str,
    warmup_days: int = 280,
) -> pd.DataFrame:
    """加载真实数据 或 使用仿真数据（fallback）"""
    warmup_start = (pd.Timestamp(start_date) - pd.offsets.BDay(warmup_days)).strftime("%Y-%m-%d")

    # 优先用真实数据
    if data_dir:
        df = load_stock_data_csv(stock_code, data_dir)
        if not df.empty:
            return df.loc[warmup_start:end_date] if not df.empty else df

    # Fallback: 仿真数据
    seed = 42 + sum(ord(c) for c in stock_code) % 997
    df = generate_synthetic_data(stock_code, warmup_start, end_date, seed=seed)
    return df


# ────────────── 信号分析 ──────────────

def analyze_signal_strength(row: pd.Series, df: pd.DataFrame, idx: int) -> dict:
    """
    分析买入信号的强度和理由

    Returns: {score: int(1-5), reasons: list[str], warnings: list[str]}
    """
    reasons = []
    warnings = []
    score = 0

    # 条件A: MA60趋势
    if row.get("close", 0) > row.get("ma_slow", 0):
        reasons.append("价格站上MA60（中期牛市结构）")
        score += 1

    ma60_now = row.get("ma_slow", 0)
    if idx >= 10:
        ma60_prev = df.iloc[idx - 10].get("ma_slow", ma60_now)
        if ma60_now > ma60_prev:
            reasons.append("MA60向上倾斜（趋势加速）")
            score += 1

    # 条件B: RSI
    rsi = row.get("rsi", 50)
    if 45 < rsi < 65:
        reasons.append(f"RSI={rsi:.1f}（健康区间，未超买）")
        score += 1
    elif rsi >= 65:
        warnings.append(f"RSI={rsi:.1f}偏高，注意追高风险")

    # 条件C: MACD
    macd_hist = row.get("macd_hist", 0)
    if macd_hist > 0:
        reasons.append("MACD柱状量转正（动能向上）")
        score += 1

    # 条件D: 量能
    vol_ratio = row.get("vol_ratio", 1.0)
    if vol_ratio >= 1.2:
        reasons.append(f"成交量放大{vol_ratio:.1f}x（强势放量）")
        score += 1
    elif vol_ratio >= 0.9:
        reasons.append(f"量能正常（{vol_ratio:.1f}x均量）")

    # 布林带位置
    boll_pb = row.get("boll_pb", 0.5)
    if boll_pb < 0.5:
        reasons.append("价格在布林带下半区（回调到位）")
    elif boll_pb > 0.8:
        warnings.append("接近布林上轨，上升空间受限")

    return {"score": min(score, 5), "reasons": reasons, "warnings": warnings}


def generate_order_advice(
    stock_code: str,
    stock_name: str,
    current_price: float,
    signal_info: dict,
    account_cfg: dict,
) -> dict:
    """生成具体的买入操作建议"""
    tp_pct = STRATEGY_CONFIG["take_profit"]   # 0.03
    sl_pct = STRATEGY_CONFIG["stop_loss"]     # 0.08
    pos_size = account_cfg["position_size"]   # 0.28
    capital = account_cfg["total_capital"]

    budget = capital * pos_size
    shares_raw = int(budget / current_price / 100) * 100  # 100股整数倍
    actual_cost = shares_raw * current_price

    take_profit_price = current_price * (1 + tp_pct)
    stop_loss_price = current_price * (1 - sl_pct)

    # 手续费估算
    commission = max(actual_cost * 0.0003, 5.0)
    stamp_sell = actual_cost * tp_pct * 0.001  # 止盈时印花税估算

    expected_profit = actual_cost * tp_pct - commission * 2 - stamp_sell
    expected_loss = actual_cost * sl_pct + commission * 2

    stars = "★" * signal_info["score"] + "☆" * (5 - signal_info["score"])

    return {
        "stock_code": stock_code,
        "stock_name": stock_name,
        "current_price": current_price,
        "suggested_buy_price": round(current_price * 1.005, 2),  # 允许0.5%的买入偏差
        "take_profit_price": round(take_profit_price, 2),
        "stop_loss_price": round(stop_loss_price, 2),
        "shares": shares_raw,
        "actual_cost": actual_cost,
        "budget_pct": pos_size * 100,
        "expected_profit": expected_profit,
        "expected_loss": expected_loss,
        "signal_strength": signal_info["score"],
        "signal_stars": stars,
        "reasons": signal_info["reasons"],
        "warnings": signal_info["warnings"],
    }


def print_order_advice(advice: dict, rank: int) -> None:
    """打印操作建议（格式化）"""
    w = 62
    sep = "─" * w

    cur_str = f"{advice['current_price']:.2f}"
    buy_str = f"{advice['suggested_buy_price']:.2f}"
    tp_str  = f"{advice['take_profit_price']:.2f}"
    sl_str  = f"{advice['stop_loss_price']:.2f}"
    name_str = f"{advice['stock_name']} ({advice['stock_code']})"

    print(f"\n  ┌{sep}┐")
    print(f"  │  {'买入建议 #' + str(rank):<{w-2}}│")
    print(f"  ├{sep}┤")
    print(f"  │  股票：{name_str}{' '*(w-5-len(name_str))}│")
    print(f"  │  信号强度：{advice['signal_stars']}{' '*(w-8-len(advice['signal_stars']))}│")
    print(f"  ├{sep}┤")
    print(f"  │  当前收盘：{cur_str}元{' '*(w-8-len(cur_str))}│")
    buy_line = f"≤{buy_str}元（次日竞价或盘中）"
    print(f"  │  建议买入：{buy_line}{' '*(w-7-len(buy_line))}│")
    tp_line = f"{tp_str}元（+3%，挂限价卖出）"
    print(f"  │  止盈目标：{tp_line}{' '*(w-7-len(tp_line))}│")
    sl_line = f"{sl_str}元（-8%，跌破须执行）"
    print(f"  │  止损价位：{sl_line}{' '*(w-7-len(sl_line))}│")
    print(f"  ├{sep}┤")
    shares_str = f"{advice['shares']}股（{advice['budget_pct']:.0f}%仓位 ≈ {advice['actual_cost']/10000:.1f}万元）"
    print(f"  │  建议股数：{shares_str}{' '*(w-7-len(shares_str))}│")
    profit_str = f"盈利+{advice['expected_profit']:.0f}元 / 亏损-{advice['expected_loss']:.0f}元（含手续费）"
    print(f"  │  {profit_str}{' '*(w-2-len(profit_str))}│")
    print(f"  ├{sep}┤")
    print(f"  │  信号理由：{' '*(w-7)}│")
    for r in advice["reasons"]:
        txt = f"    · {r}"
        print(f"  │  {txt:<{w-2}}│")
    if advice["warnings"]:
        print(f"  │  ⚠️  注意：{' '*(w-8)}│")
        for wn in advice["warnings"]:
            txt = f"    · {wn}"
            print(f"  │  {txt:<{w-2}}│")
    print(f"  └{sep}┘")


# ────────────── 主扫描逻辑 ──────────────

def run_daily_scan(
    sectors: list[str],
    data_dir: str = "",
    scan_date: str = None,
    top_n: int = 3,
) -> list[dict]:
    """
    扫描指定板块，输出今日买入/卖出建议

    Parameters
    ----------
    sectors   : 关注的板块列表（来自LLM或手动指定）
    data_dir  : 真实行情数据目录（空则使用仿真数据）
    scan_date : 扫描日期（默认今天）
    top_n     : 最多给出多少个买入建议
    """
    if scan_date is None:
        scan_date = datetime.today().strftime("%Y-%m-%d")

    end_date = scan_date
    start_date = (pd.Timestamp(scan_date) - pd.offsets.BDay(30)).strftime("%Y-%m-%d")

    # 收集目标股票
    target_stocks = {}
    for sector in sectors:
        if sector not in SECTOR_STOCK_POOL:
            logger.warning(f"未知板块：{sector}，跳过")
            continue
        for code in SECTOR_STOCK_POOL[sector]:
            target_stocks[code] = sector

    if not target_stocks:
        print("没有有效的股票，请检查板块名称")
        return []

    logger.info(f"扫描 {len(target_stocks)} 只股票，日期：{scan_date}")
    logger.info(f"板块：{', '.join(sectors)}")

    # 股票名称映射（从配置中构建）
    stock_names = {}
    for sector, stocks in SECTOR_STOCK_POOL.items():
        for code in stocks:
            # 简单映射（真实场景从行情数据中读取）
            stock_names[code] = code

    # 真实名称（已知的）
    known_names = {
        "600519": "贵州茅台", "000858": "五粮液", "601318": "中国平安",
        "600036": "招商银行", "000333": "美的集团", "002415": "海康威视",
        "600276": "恒瑞医药", "601166": "兴业银行", "000568": "泸州老窖",
        "601888": "中国中免", "002594": "比亚迪", "600900": "长江电力",
        "601012": "隆基绿能", "000725": "京东方A", "600009": "上海机场",
        "002714": "牧原股份", "600030": "中信证券", "601668": "中国建筑",
        "000001": "平安银行", "600031": "三一重工", "600703": "三安光电",
        "000063": "中兴通讯", "601138": "工业富联", "601688": "华泰证券",
        "601601": "中国太保", "002304": "洋河股份", "600809": "山西汾酒",
        "601877": "正泰电器", "600941": "中国移动", "600196": "复星医药",
        "000538": "云南白药", "600085": "同仁堂", "601088": "中国神华",
        "600019": "宝钢股份", "600887": "伊利股份", "000895": "双汇发展",
    }
    stock_names.update(known_names)

    strategy = MultiFactorStrategy(STRATEGY_CONFIG)
    buy_candidates = []

    for code, sector in target_stocks.items():
        try:
            df = load_or_simulate(code, data_dir, start_date, end_date)
            if df.empty or len(df) < 60:
                logger.debug(f"[{code}] 数据不足，跳过")
                continue

            sig_df = strategy.generate_signals(df)
            if sig_df.empty:
                continue

            # 检查最新一个交易日是否有买入信号
            latest = sig_df.iloc[-1]
            latest_idx = len(sig_df) - 1

            if latest.get("signal", 0) == 1:
                signal_info = analyze_signal_strength(latest, sig_df, latest_idx)
                name = stock_names.get(code, code)
                advice = generate_order_advice(
                    code, name, latest["close"], signal_info, ACCOUNT_CONFIG
                )
                advice["sector"] = sector
                advice["signal_date"] = sig_df.index[-1].strftime("%Y-%m-%d")
                buy_candidates.append(advice)
                logger.info(f"  [{code}] {name} 触发买入信号，强度{signal_info['score']}/5")

        except Exception as e:
            logger.debug(f"[{code}] 处理失败: {e}")

    # 按信号强度排序
    buy_candidates.sort(key=lambda x: x["signal_strength"], reverse=True)

    return buy_candidates[:top_n]


def main():
    parser = argparse.ArgumentParser(
        description="每日A股信号扫描器 — 输出操作建议（人工执行）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：
  # 指定关注板块
  python run_daily_scanner.py --sectors AI与科技 金融

  # 使用真实数据（需提前下载好CSV/parquet到 data/cache/）
  python run_daily_scanner.py --sectors 消费白酒 --data-dir data/cache

  # 指定扫描日期（回测历史某天）
  python run_daily_scanner.py --sectors 新能源 --date 2025-01-15

  # 显示所有可用板块
  python run_daily_scanner.py --list-sectors
        """
    )
    parser.add_argument("--sectors", nargs="+", default=[], help="关注的板块（空格分隔）")
    parser.add_argument("--data-dir", default="data/cache", help="行情数据目录")
    parser.add_argument("--date", default=None, help="扫描日期（默认今天）")
    parser.add_argument("--top", type=int, default=3, help="最多显示几个买入建议")
    parser.add_argument("--list-sectors", action="store_true", help="显示所有可用板块")
    parser.add_argument("--news", type=str, default=None, help="政策/新闻文本（自动调用LLM选板块）")
    parser.add_argument("--api-key", type=str, default=None, help="Anthropic API Key")
    args = parser.parse_args()

    if args.list_sectors:
        print("\n可用板块：")
        for s, codes in SECTOR_STOCK_POOL.items():
            print(f"  【{s}】: {', '.join(codes)}")
        return

    scan_date = args.date or datetime.today().strftime("%Y-%m-%d")
    sectors = args.sectors

    # LLM自动选板块
    if args.news and not sectors:
        from selector.sector_llm import analyze_sectors_with_llm, print_sector_analysis
        try:
            result = analyze_sectors_with_llm(args.news, api_key=args.api_key)
            print_sector_analysis(result, args.news)
            sectors = result.get("top_sectors", [])
        except Exception as e:
            logger.warning(f"LLM分析失败: {e}，请手动指定 --sectors")
            sectors = []

    if not sectors:
        print("\n请指定 --sectors 或 --news 来选择板块")
        print("可用板块：", list(SECTOR_STOCK_POOL.keys()))
        return

    # 扫描信号
    print("\n" + "="*65)
    print(f"  A股每日信号扫描报告")
    print(f"  扫描日期：{scan_date}")
    print(f"  关注板块：{', '.join(sectors)}")
    print(f"  账户资金：{ACCOUNT_CONFIG['total_capital']/10000:.0f}万元  仓位：{ACCOUNT_CONFIG['position_size']*100:.0f}%/笔")
    print("="*65)

    candidates = run_daily_scan(sectors, args.data_dir, scan_date, args.top)

    if candidates:
        print(f"\n  发现 {len(candidates)} 个买入信号：")
        for i, advice in enumerate(candidates, 1):
            print_order_advice(advice, i)

        print(f"\n  操作提示：")
        print(f"  · 信号为次日开盘前参考，建议在竞价阶段或开盘后确认走势再买入")
        print(f"  · 买入后务必同时在券商APP设置止损预警（{STRATEGY_CONFIG['stop_loss']*100:.0f}%）")
        print(f"  · 止盈{STRATEGY_CONFIG['take_profit']*100:.0f}%到达后可全部卖出，不建议持仓超过{STRATEGY_CONFIG['max_hold_days']}个交易日")
        print(f"  · 最大同时持仓 {ACCOUNT_CONFIG['max_positions']} 只，今日若已满仓请忽略新信号")
    else:
        print(f"\n  今日无买入信号（{', '.join(sectors)}板块暂无满足条件的标的）")
        print(f"  建议：可关注明日是否有回调到位的信号")

    print("\n" + "="*65)
    print("  ⚠️  本报告仅供参考，不构成投资建议。操作需自行判断，注意风险。")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()
