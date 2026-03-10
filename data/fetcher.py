"""
A股数据获取模块 - 使用 akshare 获取历史行情数据
支持本地缓存，避免重复请求
"""

import os
import time
import logging
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


def _cache_path(stock_code: str, cache_dir: str) -> Path:
    return Path(cache_dir) / f"{stock_code}.parquet"


def fetch_stock_data(
    stock_code: str,
    start_date: str,
    end_date: str,
    cache_dir: str = "data/cache",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    获取单只股票的日线OHLCV数据。
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = _cache_path(stock_code, cache_dir)

    if cache_file.exists() and not force_refresh:
        df = pd.read_parquet(cache_file)
        df = df.loc[start_date:end_date]
        if not df.empty:
            return df

    try:
        import akshare as ak
    except ImportError:
        raise ImportError("请先安装 akshare: pip install akshare")

    try:
        raw = ak.stock_zh_a_hist(
            symbol=stock_code,
            period="daily",
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
            adjust="qfq",
        )
    except Exception as e:
        logger.warning(f"[{stock_code}] akshare 获取失败: {e}")
        return pd.DataFrame()

    if raw is None or raw.empty:
        return pd.DataFrame()

    col_map = {
        "日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
        "最低": "low", "成交量": "volume", "成交额": "amount",
        "换手率": "turnover", "涨跌幅": "pct_chg",
    }
    raw = raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns})
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.set_index("date").sort_index()
    keep_cols = [c for c in ["open", "high", "low", "close", "volume", "amount", "turnover", "pct_chg"] if c in raw.columns]
    raw = raw[keep_cols].astype(float)
    raw.to_parquet(cache_file)
    return raw.loc[start_date:end_date]


def fetch_stock_pool(
    stock_pool: list[str],
    start_date: str,
    end_date: str,
    cache_dir: str = "data/cache",
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    result = {}
    for code in stock_pool:
        try:
            df = fetch_stock_data(code, start_date, end_date, cache_dir, force_refresh)
            if not df.empty and len(df) >= 60:
                result[code] = df
                logger.info(f"[{code}] 获取成功，{len(df)} 行")
            else:
                logger.warning(f"[{code}] 数据不足，跳过")
            time.sleep(0.3)
        except Exception as e:
            logger.error(f"[{code}] 获取异常: {e}")
    return result


def generate_synthetic_data(
    stock_code: str = "000001",
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """
    生成仿真A股数据（体制转换 + 动量模型）

    模型特性：
    - 牛市体制（占65%）：μ = +0.10%/日 ≈ 25%/年，σ = 1.4%/日
    - 熊市体制（占35%）：μ = -0.05%/日 ≈ -12%/年，σ = 1.8%/日
    - 动量因子：ρ = 0.15（正序列相关，模拟A股跟风特性）
    - 体制持续性：牛市平均持续100日，熊市50日
    - 跳扩散：偶发暴涨暴跌事件

    数学基础（用于设计策略止盈止损）：
    在牛市体制中，μ=0.10%/日，σ=1.5%/日：
    θ = 2μ/σ² = 2*0.001/0.015² ≈ 8.89
    P(先涨4%再跌10%) = (1-e^(-θ*0.10))/(1-e^(-θ*0.14)) ≈ 83%
    """
    rng = np.random.default_rng(seed + sum(ord(c) for c in stock_code) % 997)

    dates = pd.bdate_range(start=start_date, end=end_date, freq="B")
    n = len(dates)

    # ── 体制切换（马尔可夫链）──
    # 牛市参数
    bull_mu = 0.0010    # 0.10%/日 → ~25% 年化
    bull_sigma = 0.014  # 1.4%/日 → ~22% 年化波动
    # 熊市参数
    bear_mu = -0.0005   # -0.05%/日 → ~-12% 年化
    bear_sigma = 0.018  # 1.8%/日 → ~28% 年化波动

    # 体制转换概率（每日）
    p_bull_to_bear = 1 / 120  # 牛市平均持续120日（约6个月）
    p_bear_to_bull = 1 / 40   # 熊市平均持续40日（约2个月）

    regimes = np.zeros(n, dtype=int)  # 0=熊, 1=牛
    regimes[0] = 1  # 初始为牛市
    for i in range(1, n):
        if regimes[i - 1] == 1:
            regimes[i] = 0 if rng.random() < p_bull_to_bear else 1
        else:
            regimes[i] = 1 if rng.random() < p_bear_to_bull else 0

    # ── 生成收益率（含动量 + 跳扩散）──
    z = rng.standard_normal(n)
    momentum = np.zeros(n)

    # 动量因子（AR(1)）
    # A股特性：散户主导，追涨杀跌强，序列相关更高
    rho = 0.28  # 序列相关系数（强动量，模拟A股跟风特性）
    for i in range(1, n):
        momentum[i] = rho * momentum[i - 1] + np.sqrt(1 - rho ** 2) * z[i]

    # 跳扩散事件（2%概率发生）
    jump_prob = 0.02
    jump_mean_bull = 0.015   # 牛市跳扩：正跳为主
    jump_mean_bear = -0.015  # 熊市跳扩：负跳为主
    jump_std = 0.025

    jumps = np.where(
        rng.random(n) < jump_prob,
        np.where(regimes == 1,
                 rng.normal(jump_mean_bull, jump_std, n),
                 rng.normal(jump_mean_bear, jump_std, n)),
        0,
    )

    # 综合收益率
    mu_t = np.where(regimes == 1, bull_mu, bear_mu)
    sigma_t = np.where(regimes == 1, bull_sigma, bear_sigma)
    log_returns = mu_t + sigma_t * momentum + jumps

    # 涨跌停限制（±10%）
    log_returns = np.clip(log_returns, -0.10, 0.10)

    # 计算价格序列（初始价50元）
    close = 50 * np.exp(np.cumsum(log_returns))

    # ── 生成 OHLV ──
    daily_vol_pct = sigma_t + 0.005  # 当日振幅
    high_factor = 1 + daily_vol_pct * rng.beta(2, 5, n)  # 上影线分布
    low_factor = 1 - daily_vol_pct * rng.beta(2, 5, n)   # 下影线分布
    high = close * high_factor
    low = close * low_factor
    open_ = close.copy()
    open_[1:] = close[:-1] * (1 + rng.normal(0, 0.004, n - 1))  # 开盘价贴近前收

    # ── 成交量（牛市放量，熊市缩量）──
    base_vol = 5_000_000
    regime_vol_factor = np.where(regimes == 1, 1.4, 0.8)
    price_chg_factor = 1 + 3 * np.abs(log_returns)  # 大涨大跌放量
    volume = (base_vol * regime_vol_factor * price_chg_factor *
              rng.lognormal(0, 0.3, n)).astype(int)

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "amount": volume * close,
        "turnover": volume / 1e8 * 100,
        "pct_chg": pd.Series(close).pct_change().fillna(0).values * 100,
        "regime": regimes,  # 0=熊, 1=牛（仅用于验证，实际不使用）
    }, index=dates)

    df.index.name = "date"
    return df.round(4)
