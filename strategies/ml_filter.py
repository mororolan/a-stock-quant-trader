"""
XGBoost 机器学习信号过滤器

思路：
  - 以多因子策略产生的候选信号为基础
  - 提取丰富特征，训练二分类模型预测「信号后N日收益 > 阈值」
  - 只在模型置信度 >= min_proba 时实际进场
  - 显著过滤低质量信号，进一步提升胜率

特征工程：
  - 技术指标截面值
  - 动量特征（5/10/20日收益率）
  - 波动率特征
  - 量能特征
  - 趋势强度特征
"""

import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, roc_auc_score

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    # 价格动量
    "ret_5d", "ret_10d", "ret_20d",
    # RSI 状态
    "rsi", "rsi_diff",
    # MACD 状态
    "macd_hist", "macd_hist_chg",
    # KDJ
    "kdj_k", "kdj_d", "kdj_j",
    # 布林带位置
    "boll_pb", "boll_bw",
    # 量能
    "vol_ratio", "vol_ratio_5d",
    # 趋势强度
    "adx", "plus_di", "minus_di",
    # 均线多头评分
    "ma_bull_score",
    # 52周位置
    "position_52w",
    # ATR标准化波动率
    "atr_pct",
    # OBV变化率
    "obv_roc",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """从指标 DataFrame 构建 ML 特征"""
    f = df.copy()

    # 动量特征
    f["ret_5d"] = f["close"].pct_change(5)
    f["ret_10d"] = f["close"].pct_change(10)
    f["ret_20d"] = f["close"].pct_change(20)

    # RSI变化
    f["rsi_diff"] = f["rsi"].diff(3)

    # MACD hist 变化
    f["macd_hist_chg"] = f["macd_hist"].diff(2)

    # 量能5日均比
    f["vol_ratio_5d"] = f["vol_ratio"].rolling(5).mean()

    # ATR 百分比
    f["atr_pct"] = f["atr"] / f["close"]

    # OBV 变化率
    f["obv_roc"] = f["obv"].pct_change(10)

    return f


def build_labels(df: pd.DataFrame, forward_days: int = 5, threshold: float = 0.03) -> pd.Series:
    """
    标签：未来 forward_days 日收益率 > threshold 则为 1，否则 0
    threshold=3% 对应较高盈利标准
    """
    future_ret = df["close"].shift(-forward_days) / df["close"] - 1
    return (future_ret > threshold).astype(int)


class MLSignalFilter:
    """
    基于 XGBoost 的信号过滤器
    在多因子策略信号的基础上进行二次过滤
    """

    def __init__(self, ml_cfg: dict):
        self.cfg = ml_cfg
        self.model = None
        self.scaler = RobustScaler()
        self.is_trained = False
        self.feature_importance_ = None

    def _get_model(self):
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError("请安装 xgboost: pip install xgboost")
        return XGBClassifier(**self.cfg["xgb_params"], eval_metric="logloss")

    def train(
        self,
        all_stock_data: dict[str, pd.DataFrame],
        forward_days: int = 5,
        threshold: float = 0.03,
    ) -> dict:
        """
        用所有股票历史数据训练 ML 过滤器

        Parameters
        ----------
        all_stock_data : dict {code: DataFrame（含技术指标）}
        forward_days   : 未来N日为标签窗口
        threshold      : 盈利阈值

        Returns
        -------
        训练结果字典（包含 OOF 评估指标）
        """
        X_list, y_list = [], []

        for code, df in all_stock_data.items():
            try:
                feat_df = build_features(df)
                labels = build_labels(feat_df, forward_days, threshold)
                # 去掉最后 forward_days 行（无标签）
                feat_df = feat_df.iloc[:-forward_days]
                labels = labels.iloc[:-forward_days]

                avail = [c for c in FEATURE_COLS if c in feat_df.columns]
                X = feat_df[avail].copy()
                y = labels.copy()

                # 去除 NaN
                mask = X.notna().all(axis=1) & y.notna()
                X_list.append(X[mask])
                y_list.append(y[mask])
            except Exception as e:
                logger.warning(f"[{code}] 特征构建失败: {e}")

        if not X_list:
            raise ValueError("没有有效的训练数据")

        X_all = pd.concat(X_list, ignore_index=True)
        y_all = pd.concat(y_list, ignore_index=True)

        logger.info(f"训练样本: {len(X_all)}，正例比例: {y_all.mean():.2%}")

        # 时序交叉验证
        tscv = TimeSeriesSplit(n_splits=5)
        oof_preds = np.zeros(len(X_all))
        oof_probas = np.zeros(len(X_all))

        avail_cols = [c for c in FEATURE_COLS if c in X_all.columns]
        X_arr = X_all[avail_cols].values
        y_arr = y_all.values

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_arr)):
            X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
            y_tr, y_val = y_arr[train_idx], y_arr[val_idx]

            X_tr_scaled = self.scaler.fit_transform(X_tr)
            X_val_scaled = self.scaler.transform(X_val)

            model = self._get_model()
            model.fit(
                X_tr_scaled, y_tr,
                eval_set=[(X_val_scaled, y_val)],
                verbose=False,
            )
            oof_probas[val_idx] = model.predict_proba(X_val_scaled)[:, 1]
            oof_preds[val_idx] = (oof_probas[val_idx] >= self.cfg["min_proba"]).astype(int)

        # 训练最终模型（全量）
        X_scaled = self.scaler.fit_transform(X_arr)
        self.model = self._get_model()
        self.model.fit(X_scaled, y_arr, verbose=False)
        self.is_trained = True
        self.feature_cols_ = avail_cols

        # 特征重要性
        self.feature_importance_ = pd.Series(
            self.model.feature_importances_,
            index=avail_cols,
        ).sort_values(ascending=False)

        # OOF 评估
        valid_mask = oof_probas > 0
        oof_report = classification_report(
            y_arr[valid_mask],
            oof_preds[valid_mask],
            output_dict=True,
        )
        try:
            auc = roc_auc_score(y_arr, oof_probas)
        except Exception:
            auc = float("nan")

        result = {
            "n_samples": len(X_all),
            "positive_rate": float(y_all.mean()),
            "oof_accuracy": float(oof_report.get("accuracy", 0)),
            "oof_precision_1": float(oof_report.get("1", {}).get("precision", 0)),
            "oof_recall_1": float(oof_report.get("1", {}).get("recall", 0)),
            "oof_f1_1": float(oof_report.get("1", {}).get("f1-score", 0)),
            "oof_auc": float(auc),
        }
        logger.info(f"ML训练完成: ACC={result['oof_accuracy']:.3f}, AUC={result['oof_auc']:.3f}, "
                    f"Precision@1={result['oof_precision_1']:.3f}")
        return result

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """对单只股票DataFrame预测每行为正例的概率"""
        if not self.is_trained:
            raise RuntimeError("模型未训练，请先调用 train()")
        feat_df = build_features(df)
        avail = [c for c in self.feature_cols_ if c in feat_df.columns]
        X = feat_df[avail].reindex(columns=self.feature_cols_, fill_value=0).values
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def filter_signals(self, df_with_signal: pd.DataFrame) -> pd.DataFrame:
        """
        过滤 signal==1 的买入信号，保留置信度 >= min_proba 的信号

        Returns
        -------
        新 DataFrame，signal 列中置信度不足的买入信号被清零
        """
        if not self.is_trained:
            return df_with_signal

        probas = self.predict_proba(df_with_signal)
        df_out = df_with_signal.copy()
        df_out["ml_proba"] = probas
        # 置信度不足的买入信号过滤掉
        low_confidence = (df_out["signal"] == 1) & (df_out["ml_proba"] < self.cfg["min_proba"])
        df_out.loc[low_confidence, "signal"] = 0
        return df_out
