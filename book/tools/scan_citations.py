#!/usr/bin/env python3
"""
scan_citations.py —— 扫描 book/manuscript/ 全部正文（排除代码块/行内代码），
提取散文中的 Author (Year) 引用，去重输出为生成 references.bib 做准备。

输出格式：每行一条 "作者 | 年份 | 出现次数 | 首见文件#行号"
"""
import os
import re
import sys
from collections import defaultdict

# UTF-8 stdout
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "manuscript")
if not os.path.exists(BASE):
    BASE = os.path.join(os.getcwd(), "book", "manuscript")

# 年份模式（4位，19xx/20xx）
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

# 引用模式，两类形态：
#  A) 散文式：Karras et al. (2022) / Anderson (1982) / Rudin-Osher-Fatemi (1992)
#  B) 括号内前置式：（Pereyra, 2019）/ (Roberts et al., 1997) / （Ho et al., 2020）
# 年份必须完整 4 位；作者串允许：连字符、&/and/逗号分隔多人、单字母缩写、et al（带不带点）
_NAME = r"(?:[A-ZÀ-Ÿ][a-zÀ-ÿ\-]+|[a-zà-ÿ]{1,3}(?=[\s])|[A-Z]\.?(?=[\s&,)]))"
_AUTHOR_TAIL = r"(?:\s*(?:,|&|and)\s*" + _NAME + r"|\s+and\s+" + _NAME + r")*"
_ETAL = r"(?:\s+et\.?\s*al\.?)?"
CITATION_RE = re.compile(
    r"(?P<author>"
    r"(?:" + _NAME + r"(?:\s+" + _NAME + r"){0,3}" + _AUTHOR_TAIL + _ETAL + r"|"
    r"" + _NAME + _AUTHOR_TAIL + _ETAL + r")"
    r")"
    r"\s*[\(\uff08]\s*(?P<year>(?:19|20)\d{2})(?:\s*[,，]\s*(?:19|20)\d{2})?\s*[\)\uff09]"
)
# B 类：括号/全角括号内，作者在前、逗号隔年份（逗号可省，如 （Anderson 1982）/（Ho et al. 2020））
CITATION_INPAREN_RE = re.compile(
    r"[\(\uff08]\s*(?P<author>" + _NAME + r"(?:\s+" + _NAME + r"){0,3}" + _AUTHOR_TAIL + _ETAL + r")"
    r"\s*[,，]?\s*(?P<year>(?:19|20)\d{2})\s*[\)\uff09]"
)

# 更宽松的备用模式：只要年份前有 1-4 个首字母大写的单词（避免句首普通词误报）
LOOSE_CITATION_RE = re.compile(
    r"(?P<author>(?:[A-ZÀ-Ÿ][a-zÀ-ÿ\-]+(?:\s+|\s+(?:and|&)\s+|\s+et\.?\s+al\.?\s*)){1,4})"
    r"[\(\[]\s*(?P<year>(?:19|20)\d{2})\s*[\)\]]"
)


def strip_code_blocks(lines):
    """剔除 ``` ... ``` 和 ~~~ ... ~~~ 代码块，返回正文行列表 [(line_no, text, in_code_block)]"""
    code_fence = re.compile(r"^\s*(`{3,}|~{3,})")
    in_block = False
    fence_char = None
    result = []
    for raw in lines:
        line = raw.rstrip("\n")
        m = code_fence.match(line)
        if m:
            if not in_block:
                in_block = True
                fence_char = m.group(1)[0]
            elif m.group(1)[0] == fence_char:
                in_block = False
            # 整体标记为代码块行
            result.append((raw, "", in_block))
            continue
        # 行内代码 `...` 不剔除整行，只做标记
        has_inline = "`" in line and not line.strip().startswith("```")
        result.append((raw, line.strip(), in_block))
    return result


def strip_inline_code(text):
    """剔除反引号行内代码后返回"""
    return re.sub(r"`[^`]+`", "", text)


def main():
    findings = defaultdict(lambda: {"count": 0, "first_seen": None, "examples": []})
    
    md_files = []
    for root, _dirs, files in os.walk(BASE):
        for f in files:
            if f.endswith(".md"):
                md_files.append(os.path.join(root, f))
    
    md_files.sort()
    
    for fpath in md_files:
        rel = os.path.relpath(fpath, BASE)
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                raw_lines = f.readlines()
        except Exception:
            continue
        
        processed = strip_code_blocks(raw_lines)
        for idx, (raw, text_inblock, in_block) in enumerate(processed, 1):
            if in_block or not text_inblock:
                continue
            # 剔行内代码
            text = strip_inline_code(text_inblock)
            # 剔除数学公式（$...$ / $$..$$），避免公式内 "N (2020)" 类误报
            text = re.sub(r"\$[^$]*\$", "", text)
            # 找引用：两类模式合并扫描
            matches = list(CITATION_RE.finditer(text)) + list(CITATION_INPAREN_RE.finditer(text))
            for m in matches:
                author_raw = m.group("author").strip()
                year = m.group("year")
                # 清洗作者名
                author_raw = author_raw.rstrip(".,; ")  # 去尾部句点/逗号（正则回溯残留）
                author = re.sub(r"\b[A-Z]\.(?=\s)", "", author_raw).strip()  # 仅删孤立首字母缩写（"D. Anderson" → "Anderson"），保留 Adler & Adler 这类叠姓
                author = re.sub(r"\s+", " ", author).strip()
                if len(author) < 3:
                    continue
                # 剔除 "et al" 开头（说明前文被截断）
                if author.lower().startswith("et al"):
                    continue
                # 剔除明显不是引用的标题词 / 句首词（仅用大写形式匹配，避免 re.I 误杀 Anderson/Austin 等真姓氏）
                stop_words = r"^(In|Section|Lemma|Theorem|Corollary|Proof|Figure|Table|Chapter|Algorithm|Equation|Appendix|For|The|An|We|This|It|Our|Such|Other|Moreover|However|Thus|Hence|Also|Via|See|On|By|At|Of|To|From|With|Under|Over|Between|Through|During|Before|After|Above|Below|Into|Upon|There|Here|Which|Who|What|When|Where|Why|How|Not|No|But|If|Then|So|Its|Their|These|Those|All|Any|Each|Every|Both|Few|More|Most|Some|Than|Too|Very|Just|Because|As|Until|While|About|Against|Up|Down|Out|Off|Again|Further|Once)\b"
                if re.match(stop_words, author):
                    continue
                # 剔除已知误报（挑战赛 / 工具名）
                if re.search(r"\b(Challenge|Notes|Colab|Appendix|Section)\b", author, re.I):
                    continue
                key = (author, year)
                findings[key]["count"] += 1
                if findings[key]["first_seen"] is None:
                    findings[key]["first_seen"] = f"{rel}#{idx}"
                    # 记录上下文（截断）
                    ctx_start = max(0, m.start() - 40)
                    ctx_end = min(len(text), m.end() + 40)
                    ctx = text[ctx_start:ctx_end].replace("\n", " ")
                    findings[key]["examples"] = [ctx]
                elif len(findings[key]["examples"]) < 3:
                    ctx_start = max(0, m.start() - 40)
                    ctx_end = min(len(text), m.end() + 40)
                    ctx = text[ctx_start:ctx_end].replace("\n", " ")
                    findings[key]["examples"].append(ctx)
    
    # 输出
    # 按年份 + 作者排序
    sorted_keys = sorted(findings.keys(), key=lambda k: (k[1], k[0]), reverse=False)
    
    print("=" * 80)
    print(f"扫描范围: {os.path.relpath(BASE, os.getcwd())}")
    print(f"md 文件数: {len(md_files)}")
    print(f"去重后引用条目数: {len(sorted_keys)}")
    print("=" * 80)
    print()
    
    # 详细表
    print(f"{'#':<4} {'作者':<45} {'年份':<6} {'次数':<5} 首见位置")
    print("-" * 110)
    total_occ = 0
    for i, (author, year) in enumerate(sorted_keys, 1):
        info = findings[(author, year)]
        print(f"{i:<4} {author:<45} {year:<6} {info['count']:<5} {info['first_seen']}")
        total_occ += info["count"]
    
    print()
    print(f"总出现次数: {total_occ}")
    
    # 附录：上下文样例（每个引用首条）
    print()
    print("=" * 80)
    print("附录：上下文样例（每引用首条 ≤3 个）")
    print("=" * 80)
    for i, (author, year) in enumerate(sorted_keys, 1):
        info = findings[(author, year)]
        print(f"\n[{i}] {author} ({year}) ×{info['count']} — 首见 {info['first_seen']}")
        for ex in info["examples"]:
            print(f"    ... {ex} ...")
    
    # 保存精简版到 meta/citations_raw.txt
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "meta", "citations_raw.csv")
    with open(out_path, "w", encoding="utf-8") as out:
        out.write("index,author,year,count,first_seen\n")
        for i, (author, year) in enumerate(sorted_keys, 1):
            info = findings[(author, year)]
            out.write(f'{i},"{author}",{year},{info["count"]},{info["first_seen"]}\n')
    print()
    print(f"[*] 精简版已保存: {os.path.relpath(out_path, os.getcwd())}")


if __name__ == "__main__":
    main()
