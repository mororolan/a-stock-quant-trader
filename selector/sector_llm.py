"""
LLM驱动的板块选择器

工作流程：
  1. 用户把今日政策/新闻文本粘贴进来
  2. 调用 Claude API 分析受益板块
  3. 返回推荐板块列表 + 理由

使用方式：
  python -m selector.sector_llm --news "央行降准50bp，释放长期流动性..."
  或者
  python -m selector.sector_llm --interactive  # 交互输入
"""

import os
import json
import argparse
import logging
from typing import Optional

logger = logging.getLogger(__name__)


SECTOR_NAMES = [
    "AI与科技",
    "金融",
    "消费白酒",
    "新能源",
    "医药",
    "基建与制造",
    "消费与出行",
]

ANALYSIS_PROMPT = """你是一位专业的A股投资分析师，擅长从政策面和消息面分析受益板块。

用户提供了以下今日政策/新闻内容：

---
{news_text}
---

请分析上述内容，判断短期内（1-4周）哪些A股板块最可能受益。

可选板块列表：
{sectors}

请以JSON格式返回分析结果，格式如下：
{{
  "top_sectors": ["板块1", "板块2"],     // 最看好的1-3个板块（按优先级排序）
  "avoid_sectors": ["板块X"],            // 建议回避的板块（如有）
  "market_sentiment": "偏多/中性/偏空",  // 整体市场情绪判断
  "key_drivers": [                       // 核心驱动因素（3条以内）
    "驱动1",
    "驱动2"
  ],
  "risk_warning": "主要风险提示",
  "reasoning": {{                        // 每个推荐板块的简短理由
    "板块1": "理由...",
    "板块2": "理由..."
  }}
}}

注意：
- 只输出JSON，不要输出任何其他文字
- 如果新闻对市场没有明显方向性，返回 market_sentiment: "中性" 且 top_sectors 为空列表
- 板块名称必须完全匹配可选板块列表中的名称
"""


def analyze_sectors_with_llm(
    news_text: str,
    api_key: Optional[str] = None,
    model: str = "claude-sonnet-4-6",
) -> dict:
    """
    调用 Claude API 分析政策/新闻，返回推荐板块

    Parameters
    ----------
    news_text : 用户输入的政策/新闻文本
    api_key   : Anthropic API Key（默认读取环境变量 ANTHROPIC_API_KEY）
    model     : 使用的模型

    Returns
    -------
    dict with keys: top_sectors, avoid_sectors, market_sentiment,
                    key_drivers, risk_warning, reasoning
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError("请安装 anthropic SDK: pip install anthropic")

    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "未找到 Anthropic API Key。\n"
            "请设置环境变量：export ANTHROPIC_API_KEY='your-key'\n"
            "或者通过参数传入：--api-key sk-ant-..."
        )

    client = anthropic.Anthropic(api_key=api_key)

    prompt = ANALYSIS_PROMPT.format(
        news_text=news_text.strip(),
        sectors="\n".join(f"  - {s}" for s in SECTOR_NAMES),
    )

    message = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()

    # 提取JSON（有时模型会包裹在```json中）
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    result = json.loads(raw)

    # 验证板块名称合法性
    valid_sectors = set(SECTOR_NAMES)
    result["top_sectors"] = [s for s in result.get("top_sectors", []) if s in valid_sectors]
    result["avoid_sectors"] = [s for s in result.get("avoid_sectors", []) if s in valid_sectors]

    return result


def print_sector_analysis(result: dict, news_text: str) -> None:
    """格式化打印板块分析结果"""
    print("\n" + "="*60)
    print("  LLM板块分析结果")
    print("="*60)
    print(f"  市场情绪：{result.get('market_sentiment', '未知')}")
    print()

    top = result.get("top_sectors", [])
    avoid = result.get("avoid_sectors", [])

    if top:
        print("  📈 推荐关注板块（按优先级）：")
        for i, sector in enumerate(top, 1):
            reason = result.get("reasoning", {}).get(sector, "")
            print(f"    {i}. 【{sector}】 {reason}")
    else:
        print("  ⚪ 暂无明确方向性板块推荐")

    if avoid:
        print(f"\n  📉 建议回避：{', '.join(avoid)}")

    drivers = result.get("key_drivers", [])
    if drivers:
        print("\n  🔑 核心驱动：")
        for d in drivers:
            print(f"    · {d}")

    risk = result.get("risk_warning", "")
    if risk:
        print(f"\n  ⚠️  风险提示：{risk}")

    print("="*60)


def fallback_manual_selection() -> dict:
    """
    无法连接API时的手动板块选择模式
    供用户直接指定要关注的板块
    """
    print("\n" + "="*60)
    print("  手动板块选择（API不可用时使用）")
    print("="*60)
    print("  可选板块：")
    for i, s in enumerate(SECTOR_NAMES, 1):
        print(f"    {i}. {s}")
    print()

    while True:
        try:
            choices = input("  请输入板块编号（多个用逗号分隔，如 1,3）：").strip()
            indices = [int(x.strip()) - 1 for x in choices.split(",")]
            selected = [SECTOR_NAMES[i] for i in indices if 0 <= i < len(SECTOR_NAMES)]
            if selected:
                return {
                    "top_sectors": selected,
                    "avoid_sectors": [],
                    "market_sentiment": "手动选择",
                    "key_drivers": ["用户手动指定"],
                    "risk_warning": "",
                    "reasoning": {s: "用户手动选择" for s in selected},
                }
        except (ValueError, IndexError):
            pass
        print("  输入有误，请重试")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="LLM板块分析器")
    parser.add_argument("--news", type=str, help="政策/新闻文本")
    parser.add_argument("--interactive", action="store_true", help="交互式输入")
    parser.add_argument("--api-key", type=str, help="Anthropic API Key")
    parser.add_argument("--manual", action="store_true", help="手动选择板块（不调用API）")
    args = parser.parse_args()

    if args.manual:
        result = fallback_manual_selection()
        print_sector_analysis(result, "手动选择")
    else:
        if args.interactive or not args.news:
            print("请输入今日政策/新闻（输入完成后按两次Enter）：")
            lines = []
            while True:
                line = input()
                if line == "" and lines and lines[-1] == "":
                    break
                lines.append(line)
            news_text = "\n".join(lines[:-1])
        else:
            news_text = args.news

        try:
            result = analyze_sectors_with_llm(news_text, api_key=args.api_key)
            print_sector_analysis(result, news_text)
        except Exception as e:
            print(f"\n[ERROR] LLM分析失败：{e}")
            print("切换到手动选择模式...")
            result = fallback_manual_selection()
            print_sector_analysis(result, news_text)
