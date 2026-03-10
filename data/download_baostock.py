"""
Baostock 真实行情数据下载器

功能：
  - 从 baostock 下载沪深A股历史日线数据（前复权）
  - 保存为 Parquet 格式到 data/cache/ 目录
  - 支持断点续传：已有文件自动跳过，除非 --force 强制刷新

使用方式（在本地有网络的机器上运行）：
  # 安装依赖
  pip install baostock pandas pyarrow

  # 下载配置的所有股票（2020至今）
  python data/download_baostock.py

  # 下载指定时间段
  python data/download_baostock.py --start 2024-01-01 --end 2026-03-10

  # 强制刷新所有数据
  python data/download_baostock.py --force

  # 只下载指定板块
  python data/download_baostock.py --sectors AI与科技 金融

  # 下载后回测和每日扫描自动使用真实数据
  python run_daily_scanner.py --sectors AI与科技

缓存文件格式：
  data/cache/{股票代码}.parquet
  例如：data/cache/600519.parquet

数据字段（与仿真数据格式一致）：
  date (index), open, high, low, close, volume, amount, turnover, pct_chg
"""

import sys
import logging
import argparse
import time
from pathlib import Path
from datetime import datetime

import pandas as pd

# 确保从项目根目录运行
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("downloader")

CACHE_DIR = ROOT / "data" / "cache"


def bs_code(stock_code: str) -> str:
    """将纯数字股票代码转换为 baostock 格式（sh.600519 / sz.000001）"""
    if stock_code.startswith("6"):
        return f"sh.{stock_code}"
    else:
        return f"sz.{stock_code}"


def download_single(
    bs,
    stock_code: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """下载单只股票的前复权日线数据"""
    fields = "date,open,high,low,close,volume,amount,turn,pctChg"

    rs = bs.query_history_k_data_plus(
        bs_code(stock_code),
        fields,
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="2",   # 前复权
    )

    if rs.error_code != "0":
        logger.warning(f"[{stock_code}] 查询失败: {rs.error_msg}")
        return pd.DataFrame()

    data = []
    while rs.next():
        data.append(rs.get_row_data())

    if not data:
        logger.warning(f"[{stock_code}] 无数据返回")
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=rs.fields)

    # 类型转换
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # 重命名与筛选
    df = df.rename(columns={
        "turn":    "turnover",
        "pctChg":  "pct_chg",
    })
    keep = ["open", "high", "low", "close", "volume", "amount", "turnover", "pct_chg"]
    df = df[[c for c in keep if c in df.columns]]

    # 转数值（baostock返回字符串），空字符串变NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 删除全NaN行（停牌等）
    df = df.dropna(subset=["close", "volume"])
    df = df[df["close"] > 0]

    return df


def download_all(
    stock_codes: list[str],
    start_date: str,
    end_date: str,
    force: bool = False,
    delay: float = 0.15,
) -> dict:
    """
    批量下载股票数据，保存到 data/cache/

    Parameters
    ----------
    stock_codes : 股票代码列表
    start_date  : 开始日期 YYYY-MM-DD
    end_date    : 结束日期 YYYY-MM-DD
    force       : 是否强制刷新（忽略已有缓存）
    delay       : 每次请求间隔（秒），避免频控
    """
    try:
        import baostock as bs
    except ImportError:
        logger.error("请先安装 baostock: pip install baostock")
        sys.exit(1)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # 登录 baostock
    lg = bs.login()
    if lg.error_code != "0":
        logger.error(f"baostock 登录失败: {lg.error_msg}")
        sys.exit(1)
    logger.info(f"baostock 登录成功，开始下载 {len(stock_codes)} 只股票")
    logger.info(f"时间范围：{start_date} → {end_date}")

    results = {"success": [], "skip": [], "fail": []}

    try:
        for i, code in enumerate(stock_codes, 1):
            cache_file = CACHE_DIR / f"{code}.parquet"

            # 断点续传：已有文件检查日期覆盖
            if cache_file.exists() and not force:
                existing = pd.read_parquet(cache_file)
                if not existing.empty:
                    existing_end = existing.index.max().strftime("%Y-%m-%d")
                    if existing_end >= end_date:
                        logger.info(f"[{i:2d}/{len(stock_codes)}] {code} 已缓存至 {existing_end}，跳过")
                        results["skip"].append(code)
                        continue
                    else:
                        # 只补充新数据
                        new_start = (existing.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                        logger.info(f"[{i:2d}/{len(stock_codes)}] {code} 补充 {new_start} → {end_date}")
                        df_new = download_single(bs, code, new_start, end_date)
                        if not df_new.empty:
                            df = pd.concat([existing, df_new]).sort_index()
                            df = df[~df.index.duplicated(keep="last")]
                            df.to_parquet(cache_file)
                            logger.info(f"  ✅ 共 {len(df)} 行（+{len(df_new)} 新行）")
                            results["success"].append(code)
                        else:
                            results["skip"].append(code)
                        time.sleep(delay)
                        continue

            # 全量下载
            logger.info(f"[{i:2d}/{len(stock_codes)}] 下载 {code}...")
            df = download_single(bs, code, start_date, end_date)

            if df.empty:
                logger.warning(f"  ❌ {code} 无数据")
                results["fail"].append(code)
            else:
                df.to_parquet(cache_file)
                logger.info(f"  ✅ {code} 保存 {len(df)} 行 → {cache_file.name}")
                results["success"].append(code)

            time.sleep(delay)

    finally:
        bs.logout()
        logger.info("baostock 已退出登录")

    # 汇总报告
    print("\n" + "="*50)
    print(f"  下载完成！")
    print(f"  成功：{len(results['success'])} 只")
    print(f"  跳过：{len(results['skip'])} 只（已是最新缓存）")
    print(f"  失败：{len(results['fail'])} 只")
    if results["fail"]:
        print(f"  失败列表：{', '.join(results['fail'])}")
    print(f"  缓存目录：{CACHE_DIR}")
    print("="*50 + "\n")

    return results


def verify_cache(stock_codes: list[str]) -> None:
    """验证缓存文件的完整性"""
    print("\n缓存验证报告：")
    print(f"  {'代码':<8} {'行数':>6} {'开始日期':>12} {'结束日期':>12} {'状态'}")
    print("  " + "-"*50)

    for code in stock_codes:
        cache_file = CACHE_DIR / f"{code}.parquet"
        if not cache_file.exists():
            print(f"  {code:<8} {'---':>6} {'---':>12} {'---':>12} ❌ 未下载")
            continue
        try:
            df = pd.read_parquet(cache_file)
            start = df.index.min().strftime("%Y-%m-%d") if not df.empty else "---"
            end = df.index.max().strftime("%Y-%m-%d") if not df.empty else "---"
            status = "✅" if len(df) >= 60 else "⚠️  数据不足"
            print(f"  {code:<8} {len(df):>6} {start:>12} {end:>12} {status}")
        except Exception as e:
            print(f"  {code:<8} {'ERR':>6} {'---':>12} {'---':>12} ❌ {e}")
    print()


def main():
    from config import SECTOR_STOCK_POOL, DATA_CONFIG

    parser = argparse.ArgumentParser(
        description="Baostock真实行情数据下载器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--start", default="2020-01-01",
        help="开始日期 (默认: 2020-01-01)"
    )
    parser.add_argument(
        "--end", default=datetime.today().strftime("%Y-%m-%d"),
        help=f"结束日期 (默认: 今天)"
    )
    parser.add_argument(
        "--sectors", nargs="+", default=[],
        help="只下载指定板块（默认下载全部）"
    )
    parser.add_argument(
        "--codes", nargs="+", default=[],
        help="只下载指定股票代码"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="强制刷新（忽略已有缓存）"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="只验证缓存，不下载"
    )
    parser.add_argument(
        "--delay", type=float, default=0.15,
        help="请求间隔秒数 (默认: 0.15s)"
    )
    args = parser.parse_args()

    # 确定下载列表
    if args.codes:
        stock_codes = args.codes
    elif args.sectors:
        stock_codes = list(dict.fromkeys(
            code
            for sector in args.sectors
            for code in SECTOR_STOCK_POOL.get(sector, [])
        ))
        if not stock_codes:
            print(f"未找到板块，可用板块：{list(SECTOR_STOCK_POOL.keys())}")
            sys.exit(1)
    else:
        stock_codes = DATA_CONFIG["stock_pool"]

    if args.verify:
        verify_cache(stock_codes)
        return

    print(f"\n准备下载 {len(stock_codes)} 只股票：{', '.join(stock_codes)}")
    download_all(stock_codes, args.start, args.end, force=args.force, delay=args.delay)

    # 下载完毕后顺便验证
    verify_cache(stock_codes)


if __name__ == "__main__":
    main()
