"""Generate Chinese versions of HTML evaluation forms."""

import re
from pathlib import Path


# Translation mapping
TRANSLATIONS = {
    # Page elements
    "Human Evaluation Form - Page": "人工评估表 - 第",
    "Evaluator ID:": "评估员编号：",
    "Date:": "日期：",

    # Sample info
    "Sample": "样本",
    "Model:": "模型：",
    "Language:": "提示语言：",
    "True Ethnic Group:": "真实民族：",
    "Classification:": "分类结果：",
    "Predicted:": "预测：",
    "Correct:": "正确：",

    # Ethnic groups
    "Miao": "苗族",
    "Dong": "侗族",
    "Yi": "彝族",
    "Li": "黎族",
    "Tibetan": "藏族",
    "Zhuang": "壮族",

    # Section headers
    "Generated Description:": "模型生成的描述：",
    "Evaluation Ratings:": "评估评分：",

    # Rating dimensions
    "1. Cultural Accuracy:": "1. 文化准确性：",
    "2. Visual Completeness:": "2. 视觉完整性：",
    "3. Terminology:": "3. 术语恰当性：",
    "4. Factual Correctness:": "4. 事实正确性：",
    "5. Overall Quality:": "5. 整体质量：",

    # Aspects
    "Aspects covered:": "涵盖方面：",
    "Style/Color": "风格/色彩",
    "Patterns": "图案",
    "Materials": "材料",
    "Accessories": "配饰",
    "Usage": "场合",

    # Usefulness
    "Useful for documentation?": "可用于文化档案记录？",
    "Yes, as-is": "可直接使用",
    "Yes, with minor edits": "稍作修改后可用",
    "Needs major revision": "需要大幅修改",
    "Not useful": "不可用",

    # Comments
    "Additional Comments:": "补充意见：",
    "Comments on cultural accuracy...": "关于文化准确性的评语...",
    "Note any factual errors...": "请注明发现的事实错误...",
    "What did the model do well? What did it miss?": "模型做得好的方面是什么？遗漏或出错的方面是什么？",

    # Image
    "Costume image": "服饰图像",
}


def translate_html(html_content: str) -> str:
    """Translate HTML content from English to Chinese."""
    result = html_content

    # Change language attribute
    result = result.replace('lang="en"', 'lang="zh"')

    # Apply translations (longer phrases first to avoid partial matches)
    sorted_translations = sorted(TRANSLATIONS.items(), key=lambda x: -len(x[0]))
    for eng, chn in sorted_translations:
        result = result.replace(eng, chn)

    return result


def generate_chinese_forms(input_dir: str, output_dir: str) -> None:
    """Generate Chinese versions of all HTML forms.

    Args:
        input_dir: Directory containing English HTML forms.
        output_dir: Directory to save Chinese HTML forms.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for html_file in sorted(input_path.glob("*.html")):
        with open(html_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Translate
        chinese_content = translate_html(content)

        # Save with _zh suffix
        output_file = output_path / html_file.name.replace(".html", "_zh.html")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(chinese_content)

        print(f"Generated: {output_file.name}")

    print(f"\nGenerated {len(list(input_path.glob('*.html')))} Chinese form files")


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        input_dir = sys.argv[1]
        output_dir = sys.argv[2]
    else:
        input_dir = "results/human_eval/samples/forms"
        output_dir = "results/human_eval_package/zh/forms"

    generate_chinese_forms(input_dir, output_dir)
