"""
A股数据获取模块

优先级：
  1. 本地缓存（data/cache/{code}.parquet）— 由 download_baostock.py 生成
  2. akshare 在线拉取（有网络时）
  3. 仿真数据兜底（无网络 / 无缓存）
"""

import os
import time
import logging
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


CACHE_DIR = Path(__file__).resolve().parent / "cache"


def _cache_path(stock_code: str, cache_dir: str = None) -> Path:
    base = Path(cache_dir) if cache_dir else CACHE_DIR
    return base / f"{stock_code}.parquet"


def load_cache(
    stock_code: str,
    start_date: str = None,
    end_date: str = None,
    cache_dir: str = None,
) -> pd.DataFrame:
    """
    从本地缓存读取行情数据（由 data/download_baostock.py 下载生成）。
    找不到缓存时返回空 DataFrame。
    """
    p = _cache_path(stock_code, cache_dir)
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(p)
        if df.empty:
            return df
        if start_date:
            df = df.loc[start_date:]
        if end_date:
            df = df.loc[:end_date]
        return df
    except Exception as e:
        logger.warning(f"[{stock_code}] 读取缓存失败: {e}")
        return pd.DataFrame()


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


_SYNTH_CACHE: dict = {}   # 全局缓存，避免重复生成同一股票数据

def generate_synthetic_data(
    stock_code: str = "000001",
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
    seed: int = 42,
    _base_start: str = "2015-01-01",   # 固定基础起点，保证任意end_date下2020-2025数据一致
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
    # ── 缓存检查（固定基础区间，保证任意end_date下历史段数据一致）──
    cache_key = (stock_code, seed, _base_start)
    if cache_key not in _SYNTH_CACHE:
        _SYNTH_CACHE[cache_key] = _generate_synthetic_full(
            stock_code, _base_start, seed
        )
    df_full = _SYNTH_CACHE[cache_key]
    # 从全量数据切片
    result = df_full.loc[start_date:end_date]
    if not result.empty:
        return result.copy()
    # fallback：start_date 早于 base_start 时，直接生成（不缓存）
    return _generate_synthetic_full(stock_code, start_date, seed, end_date=end_date)


def _generate_synthetic_full(
    stock_code: str,
    base_start: str,
    seed: int,
    end_date: str = "2030-12-31",
) -> pd.DataFrame:
    """
    实际生成仿真数据的内部函数。
    始终从 base_start 开始，确保同一 seed 在不同调用间数据路径一致。
    """
    rng = np.random.default_rng(seed + sum(ord(c) for c in stock_code) % 997)

    dates = pd.bdate_range(start=base_start, end=end_date, freq="B")
    n = len(dates)

    # ── 体制切换（马尔可夫链）──
    bull_mu, bull_sigma = 0.0010, 0.014
    bear_mu, bear_sigma = -0.0005, 0.018
    p_bull_to_bear = 1 / 120
    p_bear_to_bull = 1 / 40

    regimes = np.zeros(n, dtype=int)
    regimes[0] = 1
    for i in range(1, n):
        if regimes[i - 1] == 1:
            regimes[i] = 0 if rng.random() < p_bull_to_bear else 1
        else:
            regimes[i] = 1 if rng.random() < p_bear_to_bull else 0

    # ── 收益率（动量 + 跳扩散）──
    z = rng.standard_normal(n)
    momentum = np.zeros(n)
    rho = 0.28
    for i in range(1, n):
        momentum[i] = rho * momentum[i - 1] + np.sqrt(1 - rho ** 2) * z[i]

    jumps = np.where(
        rng.random(n) < 0.02,
        np.where(regimes == 1,
                 rng.normal(0.015, 0.025, n),
                 rng.normal(-0.015, 0.025, n)),
        0,
    )

    mu_t = np.where(regimes == 1, bull_mu, bear_mu)
    sigma_t = np.where(regimes == 1, bull_sigma, bear_sigma)
    log_returns = np.clip(mu_t + sigma_t * momentum + jumps, -0.10, 0.10)
    close = 50 * np.exp(np.cumsum(log_returns))

    # ── OHLV ──
    daily_vol_pct = sigma_t + 0.005
    high = close * (1 + daily_vol_pct * rng.beta(2, 5, n))
    low  = close * (1 - daily_vol_pct * rng.beta(2, 5, n))
    open_ = close.copy()
    open_[1:] = close[:-1] * (1 + rng.normal(0, 0.004, n - 1))

    regime_vol_factor = np.where(regimes == 1, 1.4, 0.8)
    volume = (5_000_000 * regime_vol_factor * (1 + 3 * np.abs(log_returns)) *
              rng.lognormal(0, 0.3, n)).astype(int)

    df = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume,
        "amount": volume * close,
        "turnover": volume / 1e8 * 100,
        "pct_chg": pd.Series(close).pct_change().fillna(0).values * 100,
        "regime": regimes,
    }, index=dates)
    df.index.name = "date"
    return df.round(4)
