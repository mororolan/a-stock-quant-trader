"""
回调反弹高胜率策略 (Pullback Rebound Strategy)

════════════════════════════════════════════════════════════
策略数学基础：

对于随机游走（无方向性），止盈a / (止盈a + 止损b) = 胜率基准
  TP=2%, SL=8%: 基准胜率 = 8/(8+2) = 80%

在上升趋势体制（μ=+0.1%/日，σ=1.5%/日），θ=2μ/σ²=8.89：
  P(先涨2%再跌8%) = (1-e^{-θ×8%}) / (1-e^{-θ×10%}) ≈ 88%

关键：在趋势中的"低点"（支撑位附近）入场 → 获得方向性优势
回调信号让我们在趋势中更低的位置入场 → 实际胜率 > 基准80%

信号设计：买低不追高（避免均线全排好才买的"高点追入"陷阱）
════════════════════════════════════════════════════════════

买入条件（4个核心条件均需满足）：

  A. 大趋势确认（长期）
     - 收盘价 > MA60（中期牛市结构）
     - MA60 向上倾斜（MA60 > MA60.shift(10)）

  B. 回调确认（RSI之前跌到50以下，证明已经历回调）
     - 过去5日内 RSI 曾低于 52

  C. 反弹触发（今日出现反转信号）
     - RSI 当前 > 45 且 RSI 上穿（RSI > RSI.shift(1) + 1）
     - 或 MACD hist 从负转正（macd_hist > 0 且 macd_hist.shift(1) < 0）

  D. 量能确认（放量反弹，非缩量反弹）
     - 今日成交量 > 20日均量

卖出条件（任一触发）：
  - RSI 超买（> 75）
  - 价格跌破 MA20 × 0.985（支撑破位）
  - MACD 死叉（macd < signal）且价格低于 MA20
  - 止盈/止损由回测引擎处理（TP=2%, SL=8%）
"""

import numpy as np
import pandas as pd
from .indicators import compute_all_indicators


class MultiFactorStrategy:

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def generate_signals(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """生成买卖信号"""
        df = compute_all_indicators(df_raw, self.cfg)
        df = df.dropna().copy()

        if len(df) < 60:
            return df

        # ════════════ 条件 A：大趋势确认 ════════════
        trend_up = df["close"] > df["ma_slow"]           # 价格 > MA60（中期牛市）
        ma60_slope = df["ma_slow"] > df["ma_slow"].shift(10)  # MA60 向上倾斜

        # ════════════ 条件 B：回调确认 ════════════
        # 过去5日内 RSI 曾低于 52（证明有过回调，不是刚从超买下来）
        rsi_had_dip = df["rsi"].rolling(5).min() < 52

        # ════════════ 条件 C：反弹触发信号 ════════════
        # C1: RSI 恢复性上升（从低位反弹）
        rsi_recovery = (df["rsi"] > 45) & (df["rsi"] > df["rsi"].shift(1) + 0.5)

        # C2: MACD histogram 转正（从负到正：动能转向）
        macd_turn = (df["macd_hist"] > 0) & (df["macd_hist"].shift(1) < 0)

        # C3: 价格比3日前高（短期动量转正）
        price_momentum = df["close"] > df["close"].shift(3)

        # 至少满足 C 中的 2 个条件
        rebound_signals = (
            rsi_recovery.astype(int) +
            macd_turn.astype(int) +
            price_momentum.astype(int)
        )
        rebound_ok = rebound_signals >= 2

        # ════════════ 条件 D：量能确认 ════════════
        volume_confirm = df["vol_ratio"] >= 0.9  # 成交量不低于均量的90%

        # ════════════ 条件 E：20日价格动量为正（近期确实在涨） ════════════
        momentum_20d = df["close"] > df["close"].shift(20)  # 20日内价格上涨

        # ════════════ 额外过滤：不在过热位置买入 ════════════
        not_overbought_entry = df["rsi"] < 65       # RSI未超买
        not_near_boll_upper = df["boll_pb"] < 0.80  # 不在布林上轨附近

        # ════════════ 综合买入信号 ════════════
        buy_candidate = (
            trend_up &
            ma60_slope &
            rsi_had_dip &
            rebound_ok &
            volume_confirm &
            not_overbought_entry &
            not_near_boll_upper
        )

        # ════════════ 卖出条件 ════════════
        exit_rsi = df["rsi"] > 75                          # RSI超买
        exit_support = df["close"] < df["ma_mid"] * 0.985  # 跌破MA20支撑
        exit_macd = (df["macd"] < df["macd_signal"]) & (df["close"] < df["ma_mid"])

        sell_candidate = exit_rsi | exit_support | exit_macd

        # ════════════ 信号标记 ════════════
        # 策略只标记"潜在买入机会"，不维护持仓状态机
        # 引擎层负责实际的进出仓（同一时间只持一仓，TP/SL触发后才允许再次入场）
        # 这样确保：每次TP/SL出场后，策略可以重新捕捉新机会
        df["buy_candidate"] = buy_candidate.astype(int)
        df["sell_candidate"] = sell_candidate.astype(int)

        # signal=1 标记所有满足条件的潜在入场日（引擎负责过滤重复信号）
        df["signal"] = buy_candidate.astype(int)

        return df

    def get_signal_stats(self, df: pd.DataFrame) -> dict:
        return {
            "total_buy_signals": int((df["signal"] == 1).sum()),
            "total_sell_signals": int((df["signal"] == -1).sum()),
        }
