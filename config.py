"""
全局配置文件
"""

# ────────────── 数据配置 ──────────────
DATA_CONFIG = {
    "cache_dir": "data/cache",
    "start_date": "2020-01-01",
    "end_date": "2024-12-31",
    # 用于回测的股票池（沪深300成分股部分 + 权重股）
    "stock_pool": [
        "600519",  # 贵州茅台
        "000858",  # 五粮液
        "601318",  # 中国平安
        "600036",  # 招商银行
        "000333",  # 美的集团
        "002415",  # 海康威视
        "600276",  # 恒瑞医药
        "601166",  # 兴业银行
        "000568",  # 泸州老窖
        "601888",  # 中国中免
        "002594",  # 比亚迪
        "600900",  # 长江电力
        "601012",  # 隆基绿能
        "000725",  # 京东方A
        "600009",  # 上海机场
        "002714",  # 牧原股份
        "600030",  # 中信证券
        "601668",  # 中国建筑
        "000001",  # 平安银行
        "600031",  # 三一重工
    ],
}

# ────────────── 策略配置 ──────────────
STRATEGY_CONFIG = {
    # 趋势参数
    "ma_fast": 10,       # 短期均线
    "ma_mid": 20,        # 中期均线
    "ma_slow": 60,       # 长期均线
    "ma_trend": 120,     # 趋势均线

    # 动量参数
    "rsi_period": 14,
    "rsi_buy_low": 40,   # RSI买入下限
    "rsi_buy_high": 68,  # RSI买入上限
    "rsi_sell": 80,      # RSI超买卖出

    # MACD参数
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,

    # 量能参数
    "volume_ma": 20,
    "volume_ratio": 1.0,  # 成交量不低于均值

    # KDJ参数
    "kdj_k_buy": 25,
    "kdj_d_buy": 25,

    # 布林带参数
    "boll_period": 20,
    "boll_std": 2.0,

    # ATR止损参数
    "atr_period": 14,
    "atr_stop_multiplier": 1.5,

    # ─── 高胜率设计核心参数 ───
    # 数学推导（Brownian motion with drift）：
    # 趋势期 μ=0.10%/日，σ=1.5%/日，θ=2μ/σ²=8.89
    # P(先涨2%再跌8%) = (1-e^{-θ×0.08})/(1-e^{-θ×0.10}) ≈ 88%
    # 期望值 = 0.88×2% + 0.12×(-8%) = +0.80% （正期望）
    "max_hold_days": 30,   # 最大持仓30个交易日（给止盈足够时间触发）
    "take_profit": 0.02,   # 止盈2%（快速锁利，大幅提高胜率）
    "stop_loss": 0.08,     # 止损8%（宽止损避免噪声）
}

# ────────────── 回测配置 ──────────────
BACKTEST_CONFIG = {
    "initial_capital": 1_000_000,  # 初始资金 100万
    "commission": 0.0003,          # 手续费 万三
    "slippage": 0.001,             # 滑点 0.1%
    "stamp_duty": 0.001,           # 印花税 千一（卖出收取）
    "position_size": 0.18,         # 每次仓位 18%
    "max_positions": 5,            # 最大同时持仓数
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
    "min_proba": 0.52,  # ML信号最低置信度
}
