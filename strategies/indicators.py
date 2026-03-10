"""
技术指标计算模块
使用纯 pandas/numpy 实现，无需 ta-lib 依赖
"""

import numpy as np
import pandas as pd


# ──────────────────────────── 均线 ────────────────────────────

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


# ──────────────────────────── MACD ────────────────────────────

def macd(close: pd.Series, fast=12, slow=26, signal=9):
    """返回 (macd_line, signal_line, histogram)"""
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ──────────────────────────── RSI ────────────────────────────

def rsi(close: pd.Series, period=14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


# ──────────────────────────── KDJ ────────────────────────────

def kdj(high: pd.Series, low: pd.Series, close: pd.Series, n=9, m1=3, m2=3):
    """返回 K, D, J"""
    lowest_low = low.rolling(n).min()
    highest_high = high.rolling(n).max()
    rsv = (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan) * 100
    rsv = rsv.fillna(50)
    k = rsv.ewm(com=m1 - 1, adjust=False).mean()
    d = k.ewm(com=m2 - 1, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


# ──────────────────────────── 布林带 ────────────────────────────

def bollinger_bands(close: pd.Series, period=20, std_dev=2.0):
    """返回 (upper, mid, lower, bandwidth, %b)"""
    mid = sma(close, period)
    std = close.rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    bandwidth = (upper - lower) / mid
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return upper, mid, lower, bandwidth, pct_b


# ──────────────────────────── ATR ────────────────────────────

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period=14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


# ──────────────────────────── 成交量指标 ────────────────────────────

def volume_ratio(volume: pd.Series, period=20) -> pd.Series:
    """量比 = 当日成交量 / N日平均成交量"""
    return volume / volume.rolling(period).mean()


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On Balance Volume"""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


# ──────────────────────────── 趋势强度 ────────────────────────────

def adx(high: pd.Series, low: pd.Series, close: pd.Series, period=14):
    """ADX 趋势强度指标，返回 (ADX, +DI, -DI)"""
    tr_val = atr(high, low, close, period)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = pd.Series(plus_dm, index=close.index).ewm(com=period - 1, adjust=False).mean()
    minus_dm = pd.Series(minus_dm, index=close.index).ewm(com=period - 1, adjust=False).mean()
    plus_di = 100 * plus_dm / tr_val.replace(0, np.nan)
    minus_di = 100 * minus_dm / tr_val.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = dx.ewm(com=period - 1, adjust=False).mean()
    return adx_val, plus_di, minus_di


# ──────────────────────────── 综合指标计算 ────────────────────────────

def compute_all_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    给 OHLCV DataFrame 添加所有技术指标列
    原始列：open, high, low, close, volume
    """
    d = df.copy()

    # 均线
    d["ma_fast"] = sma(d["close"], cfg["ma_fast"])
    d["ma_mid"] = sma(d["close"], cfg["ma_mid"])
    d["ma_slow"] = sma(d["close"], cfg["ma_slow"])
    d["ma_trend"] = sma(d["close"], cfg["ma_trend"])

    # MACD
    d["macd"], d["macd_signal"], d["macd_hist"] = macd(
        d["close"], cfg["macd_fast"], cfg["macd_slow"], cfg["macd_signal"]
    )

    # RSI
    d["rsi"] = rsi(d["close"], cfg["rsi_period"])

    # KDJ
    d["kdj_k"], d["kdj_d"], d["kdj_j"] = kdj(d["high"], d["low"], d["close"])

    # 布林带
    d["boll_upper"], d["boll_mid"], d["boll_lower"], d["boll_bw"], d["boll_pb"] = (
        bollinger_bands(d["close"], cfg["boll_period"], cfg["boll_std"])
    )

    # ATR
    d["atr"] = atr(d["high"], d["low"], d["close"], cfg["atr_period"])

    # 量能
    d["vol_ma"] = sma(d["volume"], cfg["volume_ma"])
    d["vol_ratio"] = volume_ratio(d["volume"], cfg["volume_ma"])
    d["obv"] = obv(d["close"], d["volume"])

    # ADX
    d["adx"], d["plus_di"], d["minus_di"] = adx(d["high"], d["low"], d["close"])

    # 衍生特征
    d["close_pct"] = d["close"].pct_change()
    d["high_pct"] = (d["high"] - d["close"].shift(1)) / d["close"].shift(1)
    d["low_pct"] = (d["low"] - d["close"].shift(1)) / d["close"].shift(1)

    # 均线多头排列评分 (0~4)
    d["ma_bull_score"] = (
        (d["close"] > d["ma_fast"]).astype(int) +
        (d["ma_fast"] > d["ma_mid"]).astype(int) +
        (d["ma_mid"] > d["ma_slow"]).astype(int) +
        (d["ma_slow"] > d["ma_trend"]).astype(int)
    )

    # 价格位置（相对52周高低点）
    d["high_52w"] = d["close"].rolling(252).max()
    d["low_52w"] = d["close"].rolling(252).min()
    d["position_52w"] = (d["close"] - d["low_52w"]) / (d["high_52w"] - d["low_52w"]).replace(0, np.nan)

    return d
