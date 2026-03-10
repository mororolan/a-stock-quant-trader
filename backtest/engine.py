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
    trailing_high: float = 0.0   # 追踪止损用：持仓期间最高收盘价

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
        # 固定单笔预算 = 初始资金 × position_size（不随盈亏浮动，保证每笔入场金额一致）
        self.fixed_position_budget = bt_cfg["initial_capital"] * bt_cfg["position_size"]

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
        """
        按固定仓位计算可买股数（A股最小交易单位：1手=100股）

        优先级：
        1. 标准仓位(initial_capital × position_size)内尽量多买，向下取整到100股
        2. 标准仓位不够一手 → 动用当前可用现金凑满1手(100股)
        3. 连1手都买不起(price×100 > capital) → 返回0，跳过该信号
        """
        budget = min(capital, self.fixed_position_budget)
        shares = int(budget / price / 100) * 100
        if shares == 0 and capital >= price * 100:
            # 高价股：标准仓不够一手，但账上资金能凑满，买最小单位
            return 100
        return shares

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

                # 更新追踪最高价
                if price_close > current_trade.trailing_high:
                    current_trade.trailing_high = price_close

                # 固定止损（以入场价为基准）
                stop_price = current_trade.entry_price * (1 - self.st_cfg["stop_loss"])

                # 追踪止损：盈利达到 trail_activation 后激活，从最高价回撤 trail_pct 离场
                trail_activation = self.st_cfg.get("trail_activation", 0.03)
                trail_pct = self.st_cfg.get("trail_pct", 0.05)
                trail_active = current_trade.trailing_high >= current_trade.entry_price * (1 + trail_activation)
                trail_stop = current_trade.trailing_high * (1 - trail_pct) if trail_active else None

                price_low = row.get("low", price_close)
                price_high = row.get("high", price_close)

                # 用开盘价检查跳空止损
                if price_open <= stop_price:
                    exit_price = price_open
                    exit_reason = "stop_loss"
                # 开盘即触发追踪止损
                elif trail_stop is not None and price_open <= trail_stop:
                    exit_price = price_open
                    exit_reason = "trail_stop"
                # 日内最低点触发固定止损
                elif price_low <= stop_price:
                    exit_price = stop_price
                    exit_reason = "stop_loss"
                # 日内触发追踪止损
                elif trail_stop is not None and price_low <= trail_stop:
                    exit_price = trail_stop
                    exit_reason = "trail_stop"
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
                        capital -= cost
                        current_trade = Trade(
                            stock_code=stock_code,
                            entry_date=date,
                            entry_price=price_close,
                            shares=shares,
                            entry_cost=cost,
                            trailing_high=price_close,
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

                # 更新追踪最高价
                if price_close > trade.trailing_high:
                    trade.trailing_high = price_close

                stop_price = trade.entry_price * (1 - self.st_cfg["stop_loss"])

                trail_activation = self.st_cfg.get("trail_activation", 0.03)
                trail_pct = self.st_cfg.get("trail_pct", 0.05)
                trail_active = trade.trailing_high >= trade.entry_price * (1 + trail_activation)
                trail_stop = trade.trailing_high * (1 - trail_pct) if trail_active else None

                price_low = row.get("low", price_close)

                exit_price, exit_reason = None, ""
                if price_open <= stop_price:
                    exit_price, exit_reason = price_open, "stop_loss"
                elif trail_stop is not None and price_open <= trail_stop:
                    exit_price, exit_reason = price_open, "trail_stop"
                elif price_low <= stop_price:
                    exit_price, exit_reason = stop_price, "stop_loss"
                elif trail_stop is not None and price_low <= trail_stop:
                    exit_price, exit_reason = trail_stop, "trail_stop"
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
                        # 综合评分：优先选同日多信号中最强的
                        # adx         - 趋势强度（列名实际存在）
                        # ma_bull_score - MA多头排列得分（列名实际存在，取代原buy_score）
                        # rsi         - RSI动量（接近55最优，过高过低均差）
                        # ml_proba    - ML过滤概率（ML模式下存在；纯技术模式=0.5）
                        adx_score  = row.get("adx", 20) / 40
                        bull_score = row.get("ma_bull_score", 2) / 4   # ma_bull_score 0-4
                        rsi_score  = (70 - abs(row.get("rsi", 55) - 55)) / 70
                        ml_proba   = row.get("ml_proba", 0.5)
                        composite  = adx_score * 0.3 + bull_score * 0.3 + rsi_score * 0.2 + ml_proba * 0.2
                        candidates.append((composite, code, row))

                candidates.sort(reverse=True)
                for score, code, row in candidates:
                    if len(positions) >= max_pos:
                        break
                    price_close = row["close"]
                    shares = self._shares_to_buy(capital, price_close)  # 固定仓位：min(可用现金, initial_capital×position_size)
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
                        trailing_high=price_close,
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
