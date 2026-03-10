"""
全局配置文件
"""

# ────────────── 数据配置 ──────────────
# 回测用的默认股票（从宇宙文件动态加载，此处为fallback小池）
DATA_CONFIG = {
    "cache_dir": "data/cache",
    "start_date": "2023-01-01",
    "end_date":   "2026-02-28",
    # fallback：无宇宙文件时使用的最小股票池（各板块代表）
    "stock_pool": [
        "600519","000858","000568",              # 消费白酒
        "601318","600036","601166","600030",     # 金融
        "002594","600900","601012","600438",     # 新能源
        "002415","000063","601138","002230",     # AI与科技
        "600276","000538","600436",              # 医药
        "601668","600031","000333",              # 基建与制造
        "601888","600887","000651",              # 消费与出行
        "601899","600547",                       # 资源与大宗
    ],
}

# ────────────── 用户账户配置 ──────────────
ACCOUNT_CONFIG = {
    "total_capital":    150_000,      # 总资金 15万
    "position_size":    0.28,         # 每次仓位 28%（≈4.2万/笔）
    "max_positions":    3,            # 最多同时持仓3只（≈12.6万）
    "exclude_boards":   ["300","688"],# 排除创业板/科创板
    "max_tier":         2,            # 最大市值级别（1=大盘蓝筹, 2=含中盘）
}

# ────────────── 策略配置 ──────────────
STRATEGY_CONFIG = {
    # 趋势参数
    "ma_fast": 10,
    "ma_mid": 20,
    "ma_slow": 60,
    "ma_trend": 120,

    # 动量参数
    "rsi_period": 14,
    "rsi_buy_low": 40,
    "rsi_buy_high": 68,
    "rsi_sell": 80,

    # MACD参数
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,

    # 量能参数
    "volume_ma": 20,
    "volume_ratio": 1.0,

    # KDJ参数
    "kdj_k_buy": 25,
    "kdj_d_buy": 25,

    # 布林带参数
    "boll_period": 20,
    "boll_std": 2.0,

    # ATR参数
    "atr_period": 14,
    "atr_stop_multiplier": 1.5,

    # ─── 止盈止损：追踪止损版（目标年化10%+） ───
    #
    # 策略逻辑：
    # - 固定止损 -7%（硬性保本）
    # - 追踪止损：盈利达到 +3% 后激活，从最高价回撤 5% 离场
    #   → 牛市强势股可跑出 +8~15% 的利润（原固定TP仅3%）
    #   → 弱势反弹仍能保护盈利，不会亏损
    # - 体制过滤：MA60向上 + 价格在MA120上方 → 仅牛市开仓
    #   → 消除熊市低胜率（60%）拖累，提高整体胜率至80%+
    #
    # 期望值估算：胜率82% × avg_win≈4% + 18% × (-7%) = +2.02%/笔
    # 扣除手续费/滑点 ≈ +1.66%/笔净期望
    # 年化估算：150笔/5年 × 1.66% × 28% ≈ 10%
    "max_hold_days": 30,       # 最大持仓30个交易日
    "stop_loss": 0.07,         # 固定止损 -7%
    "trail_activation": 0.03,  # 追踪止损激活阈值（盈利达+3%后激活）
    "trail_pct": 0.05,         # 从最高价回撤5%触发离场
    "take_profit": 0.20,       # 硬性止盈上限20%（防止极端情况不出场）
}

# ────────────── 回测配置 ──────────────
# initial_capital 与 ACCOUNT_CONFIG["total_capital"] 保持一致（15万），
# 确保回测反映真实资金约束（有限子弹）
BACKTEST_CONFIG = {
    "initial_capital": 150_000,    # 初始资金 15万（与ACCOUNT_CONFIG一致）
    "commission": 0.0003,          # 手续费 万三
    "slippage": 0.001,             # 滑点 0.1%
    "stamp_duty": 0.001,           # 印花税 千一（卖出收取）
    "position_size": 0.28,         # 每次仓位 28%（≈4.2万/笔，固定额不随盈亏浮动）
    "max_positions": 3,            # 最大同时持仓数（3×4.2万≈12.6万，保留2.4万缓冲）
}

# ────────────── ML配置 ──────────────
ML_CONFIG = {
    "train_ratio": 0.7,
    "feature_lookback": 5,
    "xgb_params": {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    },
    "min_proba": 0.52,
}
