"""
动态选股器

流程：
  LLM输出主题关键词  →  从宇宙库按行业/关键词筛候选  →  技术预筛（趋势/流动性）  →  返回候选列表
  策略再对候选列表跑精确信号
"""

import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

UNIVERSE_PATH = Path(__file__).resolve().parent.parent / "data" / "universe.json"

# 主题关键词 → 匹配的 sector / industry 词
# LLM输出的板块名会先对这里映射，找不到时做模糊匹配
THEME_KEYWORD_MAP = {
    "AI与科技":     ["AI","科技","算力","半导体","芯片","通信","服务器","软件","云","视觉","AI视觉","金融IT","物联网","5G"],
    "金融":         ["金融","银行","券商","保险","证券","基金"],
    "消费白酒":     ["白酒","酿酒","消费","食品饮料"],
    "新能源":       ["新能源","光伏","风电","水电","锂电","动力电池","新能源汽车","储能","光伏逆变器"],
    "医药":         ["医药","创新药","中成药","生物制品","医疗器械","医疗服务","CRO","血液制品","疫苗"],
    "基建与制造":   ["基建","建筑","工程机械","钢铁","煤炭","房地产","轨道交通","航空","建材","制造"],
    "消费与出行":   ["消费","出行","免税","零售","家电","食品","乳制品","啤酒","美妆","航空","机场","猪","农业"],
    "核能与电力":   ["核电","电力","水电","火电","核能","能源"],
    "资源与大宗":   ["黄金","铜","铝","有色金属","煤炭","钼","矿业","大宗"],
    "传媒与互联网": ["游戏","传媒","互联网","广电","卫星","媒体"],
}


def load_universe(path: str = None) -> list[dict]:
    """加载股票宇宙，返回股票列表"""
    p = Path(path) if path else UNIVERSE_PATH
    if not p.exists():
        logger.warning(f"宇宙文件不存在: {p}，返回空列表")
        return []
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    stocks = data.get("stocks", [])
    # 过滤掉不合法条目（开发时的skip占位）
    stocks = [s for s in stocks if len(s.get("code", "")) == 6 and s.get("name", "") not in ("skip", "")]
    logger.info(f"宇宙加载完成：{len(stocks)} 只主板股票")
    return stocks


def filter_by_theme(stocks: list[dict], themes: list[str]) -> list[dict]:
    """
    按主题/关键词从宇宙里筛股票

    匹配规则（按优先级）：
    1. sector 精确匹配
    2. 主题关键词在 industry 字段里出现
    3. 主题名本身作为关键词模糊匹配 industry
    """
    if not themes:
        return stocks

    # 收集所有关键词
    all_keywords = set()
    for theme in themes:
        # 精确sector名
        all_keywords.add(theme)
        # 主题映射的关键词
        for kw in THEME_KEYWORD_MAP.get(theme, []):
            all_keywords.add(kw.lower())
        # 主题名自身拆字
        all_keywords.add(theme.lower())

    matched = []
    for s in stocks:
        sector = s.get("sector", "").lower()
        industry = s.get("industry", "").lower()
        name = s.get("name", "").lower()
        combined = sector + " " + industry + " " + name

        hit = any(kw.lower() in combined for kw in all_keywords)
        # 也允许 sector 精确命中
        if not hit:
            hit = any(theme == s.get("sector") for theme in themes)
        if hit:
            matched.append(s)

    logger.info(f"主题过滤：{themes} → {len(matched)} 只候选")
    return matched


def apply_basic_filters(
    stocks: list[dict],
    exclude_boards: list[str] = None,
    max_tier: int = 2,
) -> list[dict]:
    """
    基础过滤：
    - 排除指定板块（创业板300xxx已在宇宙里排除，此处为额外保险）
    - 按市值级别筛（tier=1大盘/2中盘/3小盘，默认最多到2）
    """
    exclude_boards = exclude_boards or []
    result = []
    for s in stocks:
        code = s["code"]
        # 板块排除（前三位数字）
        if any(code.startswith(pfx) for pfx in exclude_boards):
            continue
        if s.get("tier", 3) > max_tier:
            continue
        result.append(s)
    return result


def apply_technical_prefilter(
    stocks: list[dict],
    data_loader,
    scan_date: str,
    min_vol_ratio: float = 0.5,
) -> list[dict]:
    """
    技术预过滤（快速粗筛，减少后续精确信号计算量）：
    - 过去20日日均成交额 > 5000万（流动性基准）
    - 价格 > MA60（中期趋势向上，策略的基础要求）
    - 过去5日无连续跌停（异常排除）

    data_loader: 接受 (code, scan_date) 返回 DataFrame 的函数
    """
    passed = []
    for s in stocks:
        code = s["code"]
        try:
            df = data_loader(code, scan_date)
            if df is None or len(df) < 65:
                continue

            recent = df.iloc[-65:]
            last = df.iloc[-1]

            # 流动性：20日均成交额（amount列，单位元）> 5000万
            avg_amount = recent["amount"].tail(20).mean()
            if avg_amount < 5e7:
                logger.debug(f"[{code}] 流动性不足 {avg_amount/1e8:.2f}亿，跳过")
                continue

            # 趋势：收盘价 > 60日均线
            ma60 = recent["close"].rolling(60).mean().iloc[-1]
            price = last["close"]
            if price <= ma60 * 0.97:  # 留3%容错，避免刚跌破MA60的立即排除
                logger.debug(f"[{code}] 价格{price:.2f} < MA60{ma60:.2f}，跳过")
                continue

            # 异常排除：最近5日内不能有≥3次跌停（-9.9%以上）
            pct = recent["pct_chg"].tail(5)
            if (pct <= -9.9).sum() >= 3:
                logger.debug(f"[{code}] 疑似连续跌停，跳过")
                continue

            passed.append(s)
        except Exception as e:
            logger.debug(f"[{code}] 技术预过滤异常: {e}")
    logger.info(f"技术预过滤：{len(stocks)} → {len(passed)} 只通过")
    return passed


def screen(
    themes: list[str],
    data_loader,
    scan_date: str,
    universe_path: str = None,
    max_tier: int = 2,
    exclude_boards: list[str] = None,
    skip_tech_prefilter: bool = False,
) -> list[dict]:
    """
    完整选股流程入口

    Parameters
    ----------
    themes         : LLM或用户指定的主题列表，如 ["AI与科技", "新能源"]
    data_loader    : 函数 (code, scan_date) -> DataFrame，用于技术预过滤
    scan_date      : 扫描基准日期
    max_tier       : 最大市值级别（1=大盘蓝筹，2=含中盘）
    exclude_boards : 排除板块前缀（默认["300","688"]）
    skip_tech_prefilter : 跳过技术预过滤（数据不全时用）

    Returns
    -------
    list of dict，每个dict包含 code/name/sector/industry/tier
    """
    exclude_boards = exclude_boards or ["300", "688"]

    # 1. 加载宇宙
    universe = load_universe(universe_path)
    if not universe:
        return []

    # 2. 主题过滤
    candidates = filter_by_theme(universe, themes)
    if not candidates:
        logger.warning(f"主题 {themes} 未匹配到任何股票，将返回全市场候选")
        candidates = universe

    # 3. 基础过滤（板块、市值级别）
    candidates = apply_basic_filters(candidates, exclude_boards, max_tier)

    # 4. 技术预过滤（流动性 + 趋势方向）
    if not skip_tech_prefilter and data_loader is not None:
        candidates = apply_technical_prefilter(candidates, data_loader, scan_date)

    # 去重（同一code可能被多个关键词命中）
    seen = set()
    unique = []
    for s in candidates:
        if s["code"] not in seen:
            seen.add(s["code"])
            unique.append(s)

    logger.info(f"最终候选：{len(unique)} 只股票")
    return unique


def print_candidates(candidates: list[dict]) -> None:
    """打印候选列表"""
    print(f"\n  候选股票（{len(candidates)} 只）：")
    print(f"  {'代码':<8} {'名称':<10} {'板块':<12} {'细分行业':<16} {'级别'}")
    print("  " + "-"*55)
    for s in sorted(candidates, key=lambda x: (x.get("sector",""), x.get("tier",9))):
        tier_str = {1:"大盘",2:"中盘",3:"小盘"}.get(s.get("tier",0), "-")
        print(f"  {s['code']:<8} {s['name']:<10} {s.get('sector',''):<12} {s.get('industry',''):<16} {tier_str}")
    print()
