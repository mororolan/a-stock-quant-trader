"""
从 baostock 刷新股票宇宙（本地有网络时运行）

输出：data/universe.json
  - 全量A股主板股票（排除创业板300xxx / 科创板688xxx / ST / 退市）
  - 字段：code, name, sector(CSRC行业), industry, board, tier

用法：
  python data/download_universe.py              # 全量刷新
  python data/download_universe.py --verify     # 只打印当前宇宙统计
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ROOT = Path(__file__).resolve().parent.parent
UNIVERSE_PATH = ROOT / "data" / "universe.json"

# baostock CSRC行业代码 → 我们的sector分类
# 参考：http://baostock.com/baostock/index.php/A股行业分类
INDUSTRY_TO_SECTOR = {
    # 食品饮料
    "食品饮料":         "消费白酒",
    "白酒":             "消费白酒",
    # 金融
    "银行":             "金融",
    "证券":             "金融",
    "保险":             "金融",
    "多元金融":         "金融",
    # 新能源 / 电力
    "电力设备":         "新能源",
    "新能源":           "新能源",
    "电力":             "核能与电力",
    "公用事业":         "核能与电力",
    # 科技
    "电子":             "AI与科技",
    "计算机":           "AI与科技",
    "通信":             "AI与科技",
    "半导体":           "AI与科技",
    "信息技术":         "AI与科技",
    # 医药
    "医药生物":         "医药",
    "医疗保健":         "医药",
    # 基建制造
    "建筑装饰":         "基建与制造",
    "建材":             "基建与制造",
    "钢铁":             "基建与制造",
    "机械设备":         "基建与制造",
    "汽车":             "新能源",
    "国防军工":         "基建与制造",
    "交通运输":         "基建与制造",
    "房地产":           "基建与制造",
    # 消费
    "商业贸易":         "消费与出行",
    "休闲服务":         "消费与出行",
    "纺织服装":         "消费与出行",
    "家用电器":         "消费与出行",
    "农林牧渔":         "消费与出行",
    "轻工制造":         "消费与出行",
    # 资源
    "有色金属":         "资源与大宗",
    "采掘":             "资源与大宗",
    "化工":             "资源与大宗",
    # 传媒
    "传媒":             "传媒与互联网",
    "互联网":           "传媒与互联网",
}


def guess_tier(market_cap_rank: int, total: int) -> int:
    """根据排名估算市值级别（无真实市值时用排名代替）"""
    ratio = market_cap_rank / total
    if ratio <= 0.15:
        return 1  # 大盘（前15%）
    elif ratio <= 0.50:
        return 2  # 中盘
    return 3       # 小盘


def download_universe() -> list[dict]:
    """从baostock下载全量A股主板股票信息"""
    try:
        import baostock as bs
    except ImportError:
        logger.error("请先安装 baostock: pip install baostock")
        sys.exit(1)

    lg = bs.login()
    if lg.error_code != "0":
        logger.error(f"baostock登录失败: {lg.error_msg}")
        sys.exit(1)

    logger.info("开始下载股票列表...")

    # 获取所有A股列表
    rs_list = bs.query_stock_basic(code_name="")
    stocks_raw = []
    while rs_list.next():
        row = rs_list.get_row_data()
        stocks_raw.append(dict(zip(rs_list.fields, row)))

    logger.info(f"原始股票数：{len(stocks_raw)}")

    # 获取行业分类
    rs_ind = bs.query_stock_industry()
    industry_map = {}  # code -> industry
    while rs_ind.next():
        row = rs_ind.get_row_data()
        d = dict(zip(rs_ind.fields, row))
        code = d.get("code", "").replace("sh.", "").replace("sz.", "")
        industry_map[code] = d.get("industry", "")

    bs.logout()

    # 过滤 + 整理
    result = []
    excluded = 0
    for i, s in enumerate(stocks_raw):
        code_full = s.get("code", "")   # e.g. "sh.600519"
        code = code_full.replace("sh.", "").replace("sz.", "")
        name = s.get("code_name", "")
        status = s.get("status", "")    # 1=上市, 2=退市, 3=暂停
        type_ = s.get("type", "")       # 1=股票, 2=指数, ...
        out_date = s.get("outDate", "")

        # 排除：非股票、退市、ST
        if type_ != "1":
            continue
        if status != "1":
            continue
        if out_date and out_date != "":
            continue
        if "ST" in name or "*" in name:
            continue

        # 排除创业板(300xxx)和科创板(688xxx)
        if code.startswith("300") or code.startswith("688"):
            excluded += 1
            continue

        # 板块
        board = "SH" if code_full.startswith("sh.") else "SZ"

        # 行业
        industry = industry_map.get(code, "")

        # sector映射
        sector = "其他"
        for key, sec in INDUSTRY_TO_SECTOR.items():
            if key in industry:
                sector = sec
                break

        tier = guess_tier(i, len(stocks_raw))

        result.append({
            "code":     code,
            "name":     name,
            "sector":   sector,
            "industry": industry,
            "board":    board,
            "tier":     tier,
        })

    logger.info(f"过滤后：{len(result)} 只主板股票（排除创业板/科创板 {excluded} 只）")
    return result


def save_universe(stocks: list[dict], path: Path = UNIVERSE_PATH) -> None:
    data = {
        "_meta": {
            "desc": "A股主板股票宇宙（沪深主板，排除创业板300xxx/科创板688xxx）",
            "updated": datetime.today().strftime("%Y-%m-%d"),
            "total": len(stocks),
            "source": "baostock自动生成，可用 data/download_universe.py 刷新",
            "fields": "code/name/sector/industry/board(SH/SZ)/tier(1大盘2中盘3小盘)",
        },
        "stocks": stocks,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"宇宙已保存：{path}（{len(stocks)} 只股票）")


def print_stats(stocks: list[dict]) -> None:
    from collections import Counter
    print(f"\n  当前宇宙统计（共 {len(stocks)} 只主板股票）")
    print("  " + "─"*40)
    sector_cnt = Counter(s.get("sector", "其他") for s in stocks)
    for sec, cnt in sorted(sector_cnt.items(), key=lambda x: -x[1]):
        bar = "█" * (cnt // 3)
        print(f"  {sec:<14} {cnt:>4} 只  {bar}")
    tier_cnt = Counter(s.get("tier", 0) for s in stocks)
    print(f"\n  市值级别：大盘={tier_cnt.get(1,0)} / 中盘={tier_cnt.get(2,0)} / 小盘={tier_cnt.get(3,0)}")
    print()


def main():
    parser = argparse.ArgumentParser(description="从baostock刷新股票宇宙")
    parser.add_argument("--verify", action="store_true", help="只打印当前宇宙统计，不下载")
    args = parser.parse_args()

    if args.verify:
        if not UNIVERSE_PATH.exists():
            print(f"宇宙文件不存在: {UNIVERSE_PATH}")
            print("请先运行: python data/download_universe.py")
            return
        with open(UNIVERSE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        stocks = [s for s in data.get("stocks", []) if len(s.get("code","")) == 6]
        print_stats(stocks)
        return

    stocks = download_universe()
    save_universe(stocks)
    print_stats(stocks)


if __name__ == "__main__":
    main()
