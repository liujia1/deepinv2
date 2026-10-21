#!/usr/bin/env python3
"""
scan_citations.py —— 扫描 book/manuscript/ 全部正文（排除代码块/行内代码），
提取散文中的 Author (Year) 引用，去重输出为生成 references.bib 做准备。

输出格式：每行一条 "作者 | 年份 | 次数 | 首见文件#行号"

覆盖形态（2026-10-18 依据独立交叉复核结论升级，v3）：
  A) 散文式 Author (Year)：
     - 连字符姓氏可跨大写字母：Rudin-Osher-Fatemi (1992) / Albergo & Vanden-Eijnden (2023)
     - 中文连接符多作者：Bredies、Kunisch 和 Pock（2010）/ Chen, Ranftl 和 Pock（2014）
     - "等人"式：Wang 等人（2004）→ "Wang et al (2004)"
     - APA 全名式：Hastings, W. K. (1970) / Roberts, G. O., Gelman, A., & Gilks, W. R. (1997)
       （首字母缩写自动剥离："Roberts, G. O., Gelman, A., & Gilks, W. R." → "Roberts, Gelman & Gilks"）
     - APA 斜体文献式：Lou et al., *Discrete Diffusion Modeling...*（SEDD, ICML 2024 最佳论文）
     - arXiv 编号式：Chung et al. (2209.14687)（年份位记录 arXiv 编号，bib 时归并到对应年份条目）
     - 附录编号残留："附录12B Karras et al. (2022)" → "Karras et al (2022)"
     - 作者串不得从字母/数字中段开始（防 "D3PM、Gat et al." → "PM, Gat et al"）
  B) 括号内式 （Author, Year）/（Author Year）：
     - 作者可不在括号首位：（cINN, Ardizzone et al., 2019）/（SURE, Stein 1981）/（RAM, Terris et al. 2025）
     - 前缀须以分隔符或 CJK 字符收尾（防 "（Hörmann & Leydold, 2014）" 被拆成前缀+末作者）
     - 年份后可带文字：（Barratt & Sharma 2018 的系统批评）/（Tachella et al. 2023, JMLR）
     - 年份前可带会场名：（Tachella et al., NeurIPS 2022）/（Gutha et al., WACV 2025）
     - 同一括号内多条引用（；分隔）自动拆分：（Greenleaf & Uhlmann 1989；Frikel & Quinto 2013）
  C) 作者串清洗：剥首字母缩写、去编号残留、全角逗号标题剥离、会场名剥离、
     中文连接符→&、等人→et al、模型缩写不作为作者
  已知不覆盖（正文仅计数差异或条目已在别处捕获）：
     - 作者在括号外且间隔标题的非斜体形式：Daras et al. Survey (2410.00083)
     - 嵌套括号中的外层引用：（Li & Pereira (NeurIPS 2024) ...）仅内层可匹配
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

# ---------- 作者串构件 ----------
_SURNAME = r"[A-ZÀ-Þ][A-Za-zÀ-ÿ\-\u2013\u2014]+"  # 姓氏：连字符/连接号可跨大写（Rudin-Osher-Fatemi / Vanden-Eijnden / Rudin–Osher–Fatemi）
_INITIAL = r"[A-Z]\."                              # 单字母缩写（Hastings, W. K.）
# 模型/方法缩写与会议名：不得作为作者串（作者起始处负向断言 + 整串过滤）
ACRONYMS = {
    "GPU", "CPU", "TPU", "T4", "DDPM", "DDIM", "D3PM", "SEDD", "SSIM", "FID",
    "PSNR", "VAE", "SDE", "ODE", "GAN", "GANS", "CNN", "VIT", "DIT", "UNET",
    "CT", "MRI", "PET", "MAP", "MMSE", "MLE", "KL", "ELBO", "MCMC", "RAM",
    "LPIPS", "NCSN", "EDM", "LDM", "MOI", "SURE", "R2R", "DPS", "DOC", "FOE",
    "TGV", "ROF", "WACV", "CVPR", "ICCV", "ICLR", "NEURIPS", "JMLR", "ECCV",
    "BMVC", "AAAI", "ICML", "NIPS", "SIAM", "TPAMI", "PAMI", "TIP", "TCI",
    "ARXIV",
}
_ACRONYM_LOOKAHEAD = r"(?!(?:" + "|".join(sorted(ACRONYMS)) + r")\b)"
# 作者条目：姓 + 至多两个缩写，或独立缩写（APA 式 "Hastings, W. K." 的逗号后缩写）
_ITEM = _ACRONYM_LOOKAHEAD + r"(?:" + _SURNAME + r"(?:\s+" + _INITIAL + r"){0,2}|" + _INITIAL + r")"
# 条目间连接：逗号/&/and/和/与/以及/顿号（可组合，如 ", & "）或空白
# 注意：不含分号——分号只用于括号内前缀切分与 junk 段拆分（标题词后常跟 "；Author (Year)"）
_SEP = r"(?:(?:\s*(?:[,，]|&|and|和|与|以及|、))+\s*|\s+)"
_AUTHOR_TAIL = r"(?:" + _SEP + _ITEM + r")*"
_ETAL = r"(?:\s*(?:et\.?\s*al\.?|等人|等))?"
_AUTHOR = _ITEM + _AUTHOR_TAIL + _ETAL
# 年份：公元 4 位，或 arXiv 编号（2209.14687 / 2406.08929v2）
_YEAR = r"(?P<year>(?:19|20)\d{2}|(?:1[0-9]|2[0-9])\d{2}\.\d{4,5}(?:v\d+)?)"
# 会场/来源标识（可出现于作者与年份之间）
_VENUE = (r"(?:NeurIPS|NIPS|ICLR|ICML|CVPR|ICCV|ECCV|WACV|BMVC|AAAI|IJCAI|JMLR|"
          r"TPAMI|PAMI|TIP|TCI|SIAM|arXiv|Preprint|preprint|Journal|Trans\.|Proc\.|"
          r"Conf\.|Workshop|Symposium)")
# 作者不得从字母/数字中段开始（防 "D3PM、Gat et al." 匹配出 "PM, Gat et al"）
_NO_MIDTOKEN = r"(?<![A-Za-z0-9À-ÿ])"

# A) 散文式：Author (Year) / Author 等人（Year）/ Author, I. I. (Year)
CITATION_RE = re.compile(
    _NO_MIDTOKEN
    + r"(?P<author>" + _AUTHOR + r")"
    r"\s*[\(\uff08]\s*" + _YEAR + r"(?:\s*[,，]\s*(?:19|20)\d{2})?\s*[\)\uff09]"
)

# A') APA 斜体文献式：Author et al., *Title* (Year) / （Venue Year）
CITATION_ITALIC_RE = re.compile(
    _NO_MIDTOKEN
    + r"(?P<author>" + _AUTHOR + r")\s*,\s*\*[^*\n]{5,150}\*\s*[\(\uff08]\s*"
    r"(?:[^()（）]{0,40}?[,，]\s*)?(?:" + _VENUE + r"\s*[,，]?\s*)?"
    + _YEAR + r"[^)\uff08]{0,30}?[\)\uff09]"
)

# B) 括号内式：空前缀置首位；前缀须以分隔符或 CJK 字符收尾；允许会场名、年份后文字；
#    junk 段供同括号多引用二次扫描
CITATION_INPAREN_RE = re.compile(
    r"[\(\uff08]\s*"
    r"(?P<prefix>|[^()（）]{0,60}?[,，;；:：、/／]\s*|[^()（）]{0,30}?[\u4e00-\u9fff]\s*)"
    + _NO_MIDTOKEN
    + r"(?P<author>" + _AUTHOR + r")"
    r"\s*[,，]?\s*(?:" + _VENUE + r"\s*[,，]?\s*)?"
    + _YEAR + r"(?:\s*[,，]\s*(?:19|20)\d{2})?"
    r"(?P<junk>[^()（）]{0,60}?)"
    r"\s*[\)\uff09]"
)

# junk 段内的后续裸引用（junk 先按分号拆分）："；Frikel & Quinto 2013"
JUNK_CITATION_RE = re.compile(
    _NO_MIDTOKEN
    + r"(?P<author>" + _AUTHOR + r")"
    r"\s*[,，]?\s*(?:" + _VENUE + r"\s*[,，]?\s*)?"
    r"(?P<year>(?:19|20)\d{2}|(?:1[0-9]|2[0-9])\d{2}\.\d{4,5}(?:v\d+)?)"
)

# 明显不是引用的标题词 / 句首词（仅用大写形式匹配，避免 re.I 误杀 Anderson/Austin 等真姓氏）
stop_words = r"^(In|Section|Lemma|Theorem|Corollary|Proof|Figure|Table|Chapter|Algorithm|Equation|Appendix|For|The|An|We|This|It|Our|Such|Other|Moreover|However|Thus|Hence|Also|Via|See|On|By|At|Of|To|From|With|Under|Over|Between|Through|During|Before|After|Above|Below|Into|Upon|There|Here|Which|Who|What|When|Where|Why|How|Not|No|But|If|Then|So|Its|Their|These|Those|All|Any|Each|Every|Both|Few|More|Most|Some|Than|Too|Very|Just|Because|As|Until|While|About|Against|Up|Down|Out|Off|Again|Further|Once|Step|Part|Case|Eq|Fig|Sec|Ref|Version|Survey|Review|Tutorial|Analysis|Framework|Benchmark|Study|Report|Handbook|Proceedings|Journal|Conference)\b"


def clean_author(a):
    """作者串归一：剥首字母缩写、去编号残留、标题/会场剥离、中文连接符→&、等人→et al"""
    a = a.strip()
    a = a.replace("\u2013", "-").replace("\u2014", "-")  # en/em dash → 连字符
    had_dengren = bool(re.search(r"等人|等", a))
    a = re.sub(r"\s*等人?\s*", " ", a)
    # 全角逗号只可能来自括号内标题前缀（本书合法作者串不用全角逗号）→ 取末段
    if "，" in a:
        a = a.split("，")[-1]
    # 中文连接符先归一（后续标题/缩写剥离依赖逗号形态）
    a = re.sub(r"\s*、\s*", ", ", a)
    a = re.sub(r"\s*和\s*", " & ", a)
    a = re.sub(r"\s*与\s*", " & ", a)
    a = re.sub(r"\s*以及\s*", " & ", a)
    # 去首字母缩写串（含前导逗号）："Roberts, G. O., Gelman, A., & Gilks, W. R." → "Roberts, Gelman & Gilks"
    a = re.sub(r"(?:\s*(?:,\s*)?[A-Z]\.)+", "", a)
    # 无点残留缩写（尾点被吃或原文无点）："Frankel, S" / "Hastings K" / "Roberts, Gelman & Gilks R"
    a = re.sub(r"(?:^|(?<=[\s,]))\s*[A-Z](?=[\s,]|$)", "", a)
    a = re.sub(r"\s+", " ", a)
    a = re.sub(r",\s*,", ",", a)
    a = re.sub(r",\s*&", " &", a)
    a = a.strip(".,; ")
    # 行首模型缩写 + 逗号（RAM, Terris et al → Terris et al；DiT, Peebles & Xie → Peebles & Xie）
    m = re.match(r"^([A-Za-z0-9\-]+),\s+(?=[A-Z])", a)
    if m and m.group(1).upper() in ACRONYMS:
        a = a[m.end():]
    # 首个逗号前是多个词且不含 &/and（标题短语）→ 取逗号后
    # （Consistency Trajectory Model, Kim et al → Kim et al；Discrete Flow Matching, Sahoo et al → Sahoo et al）
    if ", " in a:
        head, _, tail = a.partition(", ")
        if len(head.split()) >= 2 and "&" not in head and " and " not in head:
            a = tail
    # 尾部会场名剥离（Chen, Tachella & Davies, ICCV → Chen, Tachella & Davies）
    a = re.sub(r",\s*(?:" + _VENUE + r")$", "", a)
    a = re.sub(r"\s+", " ", a).strip(" ,;&")
    if had_dengren and "et al" not in a.lower():
        a = a + " et al"
    return a


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


def collect_matches(text):
    """返回 [(author_raw, year)]：散文式 + APA 斜体式 + 括号内式 + junk 段（按分号拆分）二次扫描"""
    out = []
    for m in CITATION_RE.finditer(text):
        out.append((m.group("author"), m.group("year")))
    for m in CITATION_ITALIC_RE.finditer(text):
        out.append((m.group("author"), m.group("year")))
    for m in CITATION_INPAREN_RE.finditer(text):
        out.append((m.group("author"), m.group("year")))
        junk = m.group("junk") or ""
        for frag in re.split(r"[;；]", junk):
            for jm in JUNK_CITATION_RE.finditer(frag):
                out.append((jm.group("author"), jm.group("year")))
    return out


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
            # 找引用：多模式合并扫描
            for author_raw, year in collect_matches(text):
                author = clean_author(author_raw)
                if len(author) < 3:
                    continue
                # 剔除 "et al" 开头（说明前文被截断）
                if author.lower().startswith("et al"):
                    continue
                # 剔除明显不是引用的标题词 / 句首词
                if re.match(stop_words, author):
                    continue
                # 剔除已知误报（挑战赛 / 工具名）
                if re.search(r"\b(Challenge|Notes|Colab|Appendix|Section)\b", author, re.I):
                    continue
                if author.upper() in ACRONYMS:
                    continue
                key = (author, year)
                findings[key]["count"] += 1
                if findings[key]["first_seen"] is None:
                    findings[key]["first_seen"] = f"{rel}#{idx}"
                    # 记录上下文（截断）
                    pos = text.find(author_raw)
                    ctx_start = max(0, pos - 40) if pos >= 0 else 0
                    ctx_end = min(len(text), ctx_start + 90)
                    ctx = text[ctx_start:ctx_end].replace("\n", " ")
                    findings[key]["examples"] = [ctx]
                elif len(findings[key]["examples"]) < 3:
                    pos = text.find(author_raw)
                    ctx_start = max(0, pos - 40) if pos >= 0 else 0
                    ctx_end = min(len(text), ctx_start + 90)
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
    print(f"{'#':<4} {'作者':<45} {'年份':<12} {'次数':<5} 首见位置")
    print("-" * 120)
    total_occ = 0
    for i, (author, year) in enumerate(sorted_keys, 1):
        info = findings[(author, year)]
        print(f"{i:<4} {author:<45} {year:<12} {info['count']:<5} {info['first_seen']}")
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

    # 保存精简版到 meta/citations_raw.csv
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
