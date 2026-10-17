# -*- coding: utf-8 -*-
r"""
apply_notation.py —— 按 book/meta/notation.md 执行全书符号统一（Phase 2）。

安全设计：
  1. 只在**非代码段**应用规则（``` 围栏与行内 `code` 一律跳过）；
  2. 每组规则独立编号，支持 --only 1,2,4 选择执行，默认 dry-run，--apply 写盘；
  3. #3b（x 粗体化）只在**数学环境**（$$...$$ 与 $...$）内替换，且仅高置信形态：
       - 时间步/端点下标：x_0, x_1, x_2, x_t, x_T, x_s, x_S, x_{t-1}, x_{t+1}, x_{1:T}
       - 独立 x（前后为运算符/括号/边界等）
     明确跳过：分量下标 x_i/x_j/x_k/x_d、hat/tilde 形态 \hat{x}（留人工）、
     已有 \\mathbf{x}（天然不匹配）、y/z/ε 等伙伴变量（待作者看样例后决定）。
  4. 每组替换后自动统计替换数，供逐组 commit。

用法：
  python book/tools/apply_notation.py                # 全组 dry-run
  python book/tools/apply_notation.py --only 2 --apply
  python book/tools/apply_notation.py --only 1,2,4,5,6,7,8,9,10 --apply
"""
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[2]

# ---------- 代码段保护 ----------

def split_code(text):
    """把文本切成 [(segment, is_code)]，保持顺序。围栏优先，再切行内 `。"""
    out = []
    pos = 0
    fence_re = re.compile(r"```.*?```", re.S)
    for fm in fence_re.finditer(text):
        if fm.start() > pos:
            out.extend(_split_inline(text[pos:fm.start()]))
        out.append((fm.group(0), True))
        pos = fm.end()
    if pos < len(text):
        out.extend(_split_inline(text[pos:]))
    return out


def _split_inline(seg):
    out = []
    pos = 0
    for m in re.finditer(r"`[^`\n]+`", seg):
        if m.start() > pos:
            out.append((seg[pos:m.start()], False))
        out.append((m.group(0), True))
        pos = m.end()
    if pos < len(seg):
        out.append((seg[pos:], False))
    return out


def apply_on_prose(text, fn):
    """对非代码段应用 fn，重组文本。"""
    parts = split_code(text)
    return "".join(seg if code else fn(seg) for seg, code in parts)


# ---------- 数学环境提取（供 #3b） ----------

def math_spans(seg):
    """返回 seg 中数学 span [(start, end)]，$$..$$ 优先，再行内 $..$（不跨行）。"""
    spans = []
    for m in re.finditer(r"\$\$.*?\$\$", seg, re.S):
        spans.append(m.span())
    def in_display(s):
        return any(a <= s < b for a, b in spans)
    for ln_off, line in enumerate(re.finditer(r"[^\n]*", seg)):
        base = line.start()
        for m in re.finditer(r"(?<!\$)\$(?!\$)((?:[^$\n]|\\\$)+?)(?<!\$)\$(?!\$)", line.group(0)):
            s = base + m.start()
            if not in_display(s):
                spans.append((base + m.start(), base + m.end()))
    return spans


# ---------- 替换规则组 ----------

def rule_epsilon(seg):          # #1
    return seg.replace("\\epsilon", "\\varepsilon")

def rule_bar_alpha(seg):        # #2
    seg = re.sub(r"\\bar\{\\alpha\}", r"\\bar\\alpha", seg)
    seg = seg.replace("ᾱ", r"\\bar\\alpha")
    return seg

def rule_hat_eps(seg):          # #4（#1 之后跑）
    seg = re.sub(r"\\hat\s*\\varepsilon_\s*\\theta", r"\\varepsilon_\\theta", seg)
    seg = re.sub(r"\\hat\{\\varepsilon\}_\s*\\theta", r"\\varepsilon_\\theta", seg)
    return seg

def rule_gauss(seg):            # #5
    return re.sub(r"(?<![\\A-Za-z_{])N\(", r"\\mathcal{N}(", seg)

def rule_kl(seg):               # #6
    seg = re.sub(r"(?<![A-Za-z_])\\mathrm\{KL\}", r"D_\\text{KL}", seg)
    seg = re.sub(r"(?<![A-Za-z_\\])D_KL\b", r"D_\\text{KL}", seg)
    return seg

def rule_expect(seg):           # #7（先归一 \mathbb E，再补裸 E[）
    seg = re.sub(r"\\mathbb\s+E", r"\\mathbb{E}", seg)
    seg = re.sub(r"(?<![\\A-Za-z{_])E\[", r"\\mathbb{E}[", seg)
    return seg

def rule_norm(seg):             # #8
    return seg.replace("‖", "\\|")

def rule_inline_dollar(seg):    # #9
    seg = re.sub(r"(?<!\\)\\\(", "$", seg)
    seg = re.sub(r"(?<!\\)\\\)", "$", seg)
    return seg

def rule_mathbf_nest(seg):      # #3a：\mathbf{x_0} → \mathbf{x}_0 等嵌套归一
    return re.sub(r"\\mathbf\{x_([0-9tTsS](?:[-+]1)?)\}", r"\\mathbf{x}_\1", seg)


# #3b：x 粗体化（仅数学环境内、仅高置信形态）
RE_X_SUB = re.compile(
    r"(?<![\\A-Za-z_])x_(\{?(?:[012]|t(?:[-+]1)?|T|s|S|1:T)(?:[-+]1)?\}?)"
)
RE_X_ALONE = re.compile(r"(?<![\\A-Za-z_{}])x(?![a-zA-Z_0-9])")


def _x_sub_repl(m):
    return "\\mathbf{x}_" + m.group(1)


def rule_mathbf_x(seg):         # #3b
    spans = math_spans(seg)
    if not spans:
        return seg
    chars = list(seg)
    # 从后往前替换避免位移
    for a, b in sorted(spans, reverse=True):
        frag = "".join(chars[a:b])
        frag = RE_X_SUB.sub(_x_sub_repl, frag)
        frag = RE_X_ALONE.sub(r"\\mathbf{x}", frag)
        chars[a:b] = list(frag)
    return "".join(chars)


RULESETS = {
    1: ("ε → \\varepsilon", rule_epsilon),
    2: ("ᾱ/\\bar{\\alpha} → \\bar\\alpha", rule_bar_alpha),
    3: ("mathbf 嵌套归一", rule_mathbf_nest),
    4: ("\\hat ε_θ → ε_θ", rule_hat_eps),
    5: ("裸 N( → \\mathcal{N}(", rule_gauss),
    6: ("KL 记号 → D_\\text{KL}", rule_kl),
    7: ("期望 → \\mathbb{E}", rule_expect),
    8: ("Unicode ‖ → \\|", rule_norm),
    9: ("\\( \\) → $ $", rule_inline_dollar),
    10: ("x 粗体化（高置信）", rule_mathbf_x),
}


def find_chapter_dirs(exclude=frozenset()):
    return [d for d in REPO.iterdir()
            if d.is_dir() and re.match(r"^第\d+章", d.name)
            and int(re.match(r"^第(\d+)章", d.name).group(1)) not in exclude]


def count_math_x(seg):
    """统计数学环境内高置信裸 x（供 dry-run 预览 #10 的影响面）。"""
    n = 0
    for a, b in math_spans(seg):
        frag = seg[a:b]
        n += len(RE_X_SUB.findall(frag)) + len(RE_X_ALONE.findall(frag))
    return n


def main():
    args = sys.argv[1:]
    apply = "--apply" in args
    only = None
    if "--only" in args:
        i = args.index("--only")
        if i + 1 < len(args) and not args[i + 1].startswith("--"):
            only = {int(x) for x in args[i + 1].split(",")}
    else:
        for a in args:
            if a.startswith("--only="):
                only = {int(x) for x in a.split("=")[1].split(",")}
    groups = sorted(RULESETS) if only is None else sorted(only & set(RULESETS))
    if not groups:
        print("没有匹配的规则组"); return
    exclude = set()
    for a in args:
        if a.startswith("--exclude="):
            exclude = {int(x) for x in a.split("=")[1].split(",")}

    chapters = find_chapter_dirs(exclude)
    for gno in groups:
        label, fn = RULESETS[gno]
        n_files, n_sub = 0, 0
        for ch in sorted(chapters):
            for md in sorted(ch.glob("*.md")):
                text = md.read_text(encoding="utf-8", errors="replace")
                if gno == 10 and not apply:
                    new = text  # dry-run：只统计
                    for seg, code in split_code(text):
                        if not code:
                            n_sub += count_math_x(seg)
                    if n_sub:
                        pass
                    continue
                new = apply_on_prose(text, fn)
                if new != text:
                    n_files += 1
                    n_sub += 1
                    if apply:
                        md.write_text(new, encoding="utf-8", newline="\n")
        mode = "APPLY" if apply else "DRY-RUN"
        print(f"[{mode}] 组#{gno} {label}：变更 {n_files} 个文件")

    if not apply and 10 in groups:
        print(f"[DRY-RUN] 组#10 预估替换（数学环境内高置信裸 x）：约 {n_sub} 处")


if __name__ == "__main__":
    main()
