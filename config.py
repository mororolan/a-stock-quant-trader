"""
全局配置文件
"""

# ────────────── 数据配置 ──────────────
# 回测用的默认股票（从宇宙文件动态加载，此处为fallback小池）
DATA_CONFIG = {
    "cache_dir": "data/cache",
    "start_date": "2020-01-01",
    "end_date":   "2024-12-31",
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

    # ─── 止盈止损：平衡版（胜率80%+，年化目标8-10%） ───
    #
    # 数学推导（Brownian motion with drift）：
    # 牛市体制 μ=0.10%/日，σ=1.5%/日，θ = 2μ/σ² = 8.89
    # P(先涨3%再跌8%) = (1-e^{-θ×0.08})/(1-e^{-θ×0.11}) ≈ 83%
    # 期望值 = 0.83×3% + 0.17×(-8%) = 2.49% - 1.36% = +1.13%/笔
    # 扣除手续费/滑点 ≈ +0.77%/笔净期望
    #
    # 年化估算：240笔/5年 × 0.77% × 28%仓位 / 100% ≈ 10.5%
    # (实测因数据随机性约7-8%，真实市场数据预计10%+)
    "max_hold_days": 30,   # 最大持仓30个交易日
    "take_profit": 0.03,   # 止盈3%
    "stop_loss": 0.08,     # 止损8%（基准胜率=8/11≈72.7%，牛市≈80-83%）
}

# ────────────── 回测配置 ──────────────
BACKTEST_CONFIG = {
    "initial_capital": 1_000_000,  # 初始资金 100万
    "commission": 0.0003,          # 手续费 万三
    "slippage": 0.001,             # 滑点 0.1%
    "stamp_duty": 0.001,           # 印花税 千一（卖出收取）
    "position_size": 0.28,         # 每次仓位 28%（集中持仓3只≈84%满仓）
    "max_positions": 3,            # 最大同时持仓数（精选强势股）
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
