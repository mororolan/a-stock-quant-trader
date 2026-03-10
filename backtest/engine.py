"""
回测引擎

功能：
  - 单股票回测 / 股票池组合回测
  - 精确模拟手续费（万三）+ 印花税（千一卖出）+ 滑点
  - 动态仓位管理（固定百分比仓位）
  - ATR 动态止损 + 固定止盈止损
  - 最大持仓天数限制
  - 生成交易明细 + 资金曲线
"""

import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """单笔交易记录"""
    stock_code: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    shares: int = 0
    entry_cost: float = 0.0      # 买入总费用（含手续费+滑点）
    exit_proceeds: float = 0.0   # 卖出净收入（扣除手续费+印花税+滑点）
    pnl: float = 0.0             # 盈亏金额
    pnl_pct: float = 0.0         # 盈亏百分比
    hold_days: int = 0
    exit_reason: str = ""        # 退出原因

    @property
    def is_win(self) -> bool:
        return self.pnl > 0


class BacktestEngine:
    """
    向量化+事件驱动混合回测引擎
    """

    def __init__(self, bt_cfg: dict, strategy_cfg: dict):
        self.bt_cfg = bt_cfg
        self.st_cfg = strategy_cfg

    # ────────────── 成本计算 ──────────────

    def _buy_cost(self, price: float, shares: int) -> float:
        """买入总费用"""
        amount = price * shares
        commission = max(amount * self.bt_cfg["commission"], 5.0)  # 最低5元
        slippage = amount * self.bt_cfg["slippage"]
        return amount + commission + slippage

    def _sell_proceeds(self, price: float, shares: int) -> float:
        """卖出净收入"""
        amount = price * shares
        commission = max(amount * self.bt_cfg["commission"], 5.0)
        stamp_duty = amount * self.bt_cfg["stamp_duty"]
        slippage = amount * self.bt_cfg["slippage"]
        return amount - commission - stamp_duty - slippage

    def _shares_to_buy(self, capital: float, price: float) -> int:
        """按仓位比例计算可买股数（100股整数倍）"""
        budget = capital * self.bt_cfg["position_size"]
        shares = int(budget / price / 100) * 100
        return max(shares, 0)

    # ────────────── 单只股票回测 ──────────────

    def run_single(
        self,
        df: pd.DataFrame,
        stock_code: str = "unknown",
    ) -> tuple[list[Trade], pd.Series]:
        """
        对单只股票运行回测

        Parameters
        ----------
        df : 包含 signal 列的 DataFrame（来自策略信号生成）
        stock_code : 股票代码（用于记录）

        Returns
        -------
        trades : 交易列表
        equity_curve : 净值曲线（DatetimeIndex）
        """
        capital = float(self.bt_cfg["initial_capital"])
        equity = capital
        trades: list[Trade] = []
        equity_curve = []

        current_trade: Optional[Trade] = None

        for i, (date, row) in enumerate(df.iterrows()):
            price_open = row.get("open", row["close"])
            price_close = row["close"]

            # ── 持仓中：检查止盈止损、最大持仓天数 ──
            if current_trade is not None:
                current_trade.hold_days += 1
                exit_price = None
                exit_reason = ""

                # 固定止损（以入场价为基准，不使用ATR动态止损，避免实际止损比TP更紧）
                # 设计逻辑：TP=2%, SL=8% → 随机游走基准胜率 = 8/(8+2) = 80%
                stop_price = current_trade.entry_price * (1 - self.st_cfg["stop_loss"])

                # 固定止盈
                take_profit_price = current_trade.entry_price * (1 + self.st_cfg["take_profit"])

                # 用开盘价检查跳空
                if price_open <= stop_price:
                    exit_price = price_open
                    exit_reason = "stop_loss"
                elif price_open >= take_profit_price:
                    exit_price = price_open
                    exit_reason = "take_profit"
                # 日内最低点触发止损
                elif row.get("low", price_close) <= stop_price:
                    exit_price = stop_price
                    exit_reason = "stop_loss"
                # 日内最高点触发止盈
                elif row.get("high", price_close) >= take_profit_price:
                    exit_price = take_profit_price
                    exit_reason = "take_profit"
                # 策略卖出信号
                elif row.get("signal", 0) == -1:
                    exit_price = price_close
                    exit_reason = "signal_exit"
                # 最大持仓天数
                elif current_trade.hold_days >= self.st_cfg["max_hold_days"]:
                    exit_price = price_close
                    exit_reason = "max_hold"

                if exit_price is not None:
                    proceeds = self._sell_proceeds(exit_price, current_trade.shares)
                    current_trade.exit_date = date
                    current_trade.exit_price = exit_price
                    current_trade.exit_proceeds = proceeds
                    current_trade.pnl = proceeds - current_trade.entry_cost
                    current_trade.pnl_pct = current_trade.pnl / current_trade.entry_cost
                    current_trade.exit_reason = exit_reason
                    capital += proceeds
                    trades.append(current_trade)
                    current_trade = None

            # ── 无持仓：检查买入信号 ──
            if current_trade is None and row.get("signal", 0) == 1:
                shares = self._shares_to_buy(capital, price_close)
                if shares > 0:
                    cost = self._buy_cost(price_close, shares)
                    if cost <= capital:
                        atr_val = row.get("atr", price_close * 0.03)
                        capital -= cost
                        current_trade = Trade(
                            stock_code=stock_code,
                            entry_date=date,
                            entry_price=price_close,
                            shares=shares,
                            entry_cost=cost,
                        )

            # ── 计算当日权益 ──
            position_value = (current_trade.shares * price_close) if current_trade else 0
            equity = capital + position_value
            equity_curve.append({"date": date, "equity": equity})

        # 期末强制平仓
        if current_trade is not None and not df.empty:
            last_row = df.iloc[-1]
            exit_price = last_row["close"]
            proceeds = self._sell_proceeds(exit_price, current_trade.shares)
            current_trade.exit_date = df.index[-1]
            current_trade.exit_price = exit_price
            current_trade.exit_proceeds = proceeds
            current_trade.pnl = proceeds - current_trade.entry_cost
            current_trade.pnl_pct = current_trade.pnl / current_trade.entry_cost
            current_trade.exit_reason = "end_of_backtest"
            trades.append(current_trade)

        equity_series = pd.DataFrame(equity_curve).set_index("date")["equity"]
        return trades, equity_series

    # ────────────── 股票池组合回测 ──────────────

    def run_portfolio(
        self,
        stock_signals: dict[str, pd.DataFrame],
    ) -> tuple[list[Trade], pd.Series, pd.DataFrame]:
        """
        多股票组合回测（资金共享，轮动持仓）

        Parameters
        ----------
        stock_signals : {code: df_with_signals}

        Returns
        -------
        all_trades     : 所有交易列表
        portfolio_equity : 组合净值曲线
        trade_df       : 交易明细 DataFrame
        """
        capital = float(self.bt_cfg["initial_capital"])
        max_pos = self.bt_cfg["max_positions"]

        # 对齐时间轴
        all_dates = sorted(set().union(*[set(df.index) for df in stock_signals.values()]))
        all_dates = pd.DatetimeIndex(all_dates)

        positions: dict[str, Trade] = {}   # {code: Trade}
        all_trades: list[Trade] = []
        equity_curve = []

        for date in all_dates:
            # ── 处理持仓中的止盈止损 ──
            codes_to_exit = []
            for code, trade in positions.items():
                if date not in stock_signals[code].index:
                    continue
                row = stock_signals[code].loc[date]
                trade.hold_days += 1
                price_open = row.get("open", row["close"])
                price_close = row["close"]

                stop_price = trade.entry_price * (1 - self.st_cfg["stop_loss"])
                take_profit_price = trade.entry_price * (1 + self.st_cfg["take_profit"])

                exit_price, exit_reason = None, ""
                if price_open <= stop_price:
                    exit_price, exit_reason = price_open, "stop_loss"
                elif price_open >= take_profit_price:
                    exit_price, exit_reason = price_open, "take_profit"
                elif row.get("low", price_close) <= stop_price:
                    exit_price, exit_reason = stop_price, "stop_loss"
                elif row.get("high", price_close) >= take_profit_price:
                    exit_price, exit_reason = take_profit_price, "take_profit"
                elif row.get("signal", 0) == -1:
                    exit_price, exit_reason = price_close, "signal_exit"
                elif trade.hold_days >= self.st_cfg["max_hold_days"]:
                    exit_price, exit_reason = price_close, "max_hold"

                if exit_price is not None:
                    proceeds = self._sell_proceeds(exit_price, trade.shares)
                    trade.exit_date = date
                    trade.exit_price = exit_price
                    trade.exit_proceeds = proceeds
                    trade.pnl = proceeds - trade.entry_cost
                    trade.pnl_pct = trade.pnl / trade.entry_cost
                    trade.exit_reason = exit_reason
                    capital += proceeds
                    all_trades.append(trade)
                    codes_to_exit.append(code)

            for code in codes_to_exit:
                del positions[code]

            # ── 扫描买入信号 ──
            if len(positions) < max_pos:
                # 按信号强度排序（优先买更强的信号）
                candidates = []
                for code, df in stock_signals.items():
                    if code in positions:
                        continue
                    if date not in df.index:
                        continue
                    row = df.loc[date]
                    if row.get("signal", 0) == 1:
                        # 综合评分：ADX趋势强度 + 买入评分 + RSI动量
                        adx_score = row.get("adx", 20) / 40          # 归一化ADX
                        buy_score = row.get("buy_score", 3) / 5      # 归一化买入评分
                        rsi_score = (70 - abs(row.get("rsi", 55) - 55)) / 70  # RSI接近55最佳
                        ml_proba = row.get("ml_proba", 0.5)
                        composite = adx_score * 0.3 + buy_score * 0.3 + rsi_score * 0.2 + ml_proba * 0.2
                        candidates.append((composite, code, row))

                candidates.sort(reverse=True)
                for score, code, row in candidates:
                    if len(positions) >= max_pos:
                        break
                    price_close = row["close"]
                    shares = self._shares_to_buy(capital, price_close)  # 用全部可用资金按position_size比例计算
                    if shares <= 0:
                        continue
                    cost = self._buy_cost(price_close, shares)
                    if cost > capital:
                        continue
                    capital -= cost
                    positions[code] = Trade(
                        stock_code=code,
                        entry_date=date,
                        entry_price=price_close,
                        shares=shares,
                        entry_cost=cost,
                    )

            # ── 计算当日权益 ──
            pos_value = 0
            for code, trade in positions.items():
                if date in stock_signals[code].index:
                    pos_value += trade.shares * stock_signals[code].loc[date, "close"]
                else:
                    pos_value += trade.shares * trade.entry_price
            equity_curve.append({"date": date, "equity": capital + pos_value})

        # 期末强制平仓
        for code, trade in positions.items():
            df = stock_signals[code]
            last_date = df.index[-1]
            last_price = df.iloc[-1]["close"]
            proceeds = self._sell_proceeds(last_price, trade.shares)
            trade.exit_date = last_date
            trade.exit_price = last_price
            trade.exit_proceeds = proceeds
            trade.pnl = proceeds - trade.entry_cost
            trade.pnl_pct = trade.pnl / trade.entry_cost
            trade.exit_reason = "end_of_backtest"
            all_trades.append(trade)

        equity_series = pd.DataFrame(equity_curve).set_index("date")["equity"]
        trade_df = self._trades_to_df(all_trades)
        return all_trades, equity_series, trade_df

    @staticmethod
    def _trades_to_df(trades: list[Trade]) -> pd.DataFrame:
        if not trades:
            return pd.DataFrame()
        records = [
            {
                "stock_code": t.stock_code,
                "entry_date": t.entry_date,
                "exit_date": t.exit_date,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "shares": t.shares,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "hold_days": t.hold_days,
                "exit_reason": t.exit_reason,
                "is_win": t.is_win,
            }
            for t in trades
        ]
        return pd.DataFrame(records)
