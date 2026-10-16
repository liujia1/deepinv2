# -*- coding: utf-8 -*-
"""
audit_notation.py —— 全书符号/记法一致性审计（初版）。

目的：各章独立写作，同一数学对象可能出现多种 LaTeX 记法。
本脚本统计"互斥记法组"在各章的使用次数，找出：
  * 同一章内混用的组（章内不一致，最影响阅读）
  * 全书少数派记法（跨章不一致，成书时需统一到符号表）

审计对象：仓库根下各章写作源（含绪论）。

输出：book/meta/audit_notation_report.md
用法：python book/tools/audit_notation.py
"""
import re
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[2]
META = REPO / "book" / "meta"

# 互斥记法组：组名 -> {记法名: 正则}
# 说明：这些是"同类对象的不同写法"，成书时应统一为一种。
NOTATION_GROUPS = {
    "ε（噪声）": {
        r"\\epsilon": re.compile(r"\\epsilon\b"),
        r"\\varepsilon": re.compile(r"\\varepsilon\b"),
    },
    "ᾱ_t（累积信号保留率）": {
        r"\\bar\\alpha": re.compile(r"\\bar\\alpha"),
        r"\\bar{\\alpha}": re.compile(r"\\bar\{\\alpha\}"),
        r"\\barα（Unicode）": re.compile(r"\\barα|ᾱ"),
    },
    "x 的粗体": {
        r"\\mathbf{x}": re.compile(r"\\mathbf\{x\}"),
        r"纯 x": re.compile(r"(?<![\\a-zA-Z])x(?![a-zA-Z_])"),
    },
    "ε_θ 的 hat": {
        r"\\hat\\epsilon_\\theta": re.compile(r"\\hat\\epsilon_\\theta"),
        r"\\epsilon_\\theta（无hat）": re.compile(r"\\epsilon_\\theta"),
        r"\\hat{\\epsilon}_\\theta": re.compile(r"\\hat\{\\epsilon\}_\\theta"),
        r"\\hat\\epsilon_{\\theta}": re.compile(r"\\hat\\epsilon_\{\\theta\}"),
    },
    "高斯记号": {
        r"\\mathcal{N}": re.compile(r"\\mathcal\{?N\}?"),
        r"\\mathrm{N}": re.compile(r"\\mathrm\{N\}"),
        r"N(0,I)（裸N）": re.compile(r"(?<![\\a-zA-Z\{])N\(0"),
    },
    "KL 散度": {
        r"D_\\text{KL}": re.compile(r"D_\\text\{?KL"),
        r"D_\\mathrm{KL}": re.compile(r"D_\\mathrm\{?KL"),
        r"\\mathrm{KL}（无D前缀）": re.compile(r"(?<![A-Za-z_])\\mathrm\{KL\}"),
        r"D_KL（裸下标）": re.compile(r"D_KL"),
        r"D_{\\rm KL}": re.compile(r"D_\{\\rm KL\}"),
    },
    "期望算子": {
        r"\\mathbb{E}": re.compile(r"\\mathbb\{?E\}?"),
        r"E[（裸E）": re.compile(r"(?<![\\a-zA-Z_])E\[[^]]"),
        r"\\mathbb E（无括号）": re.compile(r"\\mathbb\s+E(?![a-zA-Z])"),
    },
    "范数记号": {
        r"\\|（双竖线）": re.compile(r"\\\|"),
        r"\\lVert": re.compile(r"\\lVert"),
        r"‖（Unicode）": re.compile(r"‖"),
    },
    "行内公式定界": {
        "$...$（美元）": re.compile(r"(?<!\$)\$[^$\n]+?\$(?!\$)"),
        r"\\( \\)（圆括号）": re.compile(r"\\\(\s*[^)]+?\s*\\\)"),
    },
    "显示公式定界": {
        r"$$（双美元）": re.compile(r"^\$\$|\$\$$", re.M),
        r"\\[ \\]（方括号）": re.compile(r"^\\\[|\\\]$", re.M),
    },
    "下标 0（x_0）": {
        r"x_0": re.compile(r"x_0"),
        r"\\mathbf{x}_0": re.compile(r"\\mathbf\{x\}_0"),
        r"\\mathbf{x_0}（粗体含下标）": re.compile(r"\\mathbf\{x_0\}"),
    },
    "σ（噪声水平）": {
        r"\\sigma": re.compile(r"\\sigma\b"),
        r"\\sigma_y": re.compile(r"\\sigma_y"),
    },
}

def find_chapter_dirs():
    dirs = []
    for d in REPO.iterdir():
        if d.is_dir() and re.match(r"^第\d+章", d.name):
            dirs.append(d)
    intro = REPO / "绪论"
    if intro.is_dir():
        dirs.append(intro)
    return sorted(dirs, key=lambda d: (0, int(re.match(r"^第(\d+)章", d.name).group(1)))
                  if re.match(r"^第\d+章", d.name) else (-1, 0))

def chapter_label(d: Path) -> str:
    m = re.match(r"^第(\d+)章", d.name)
    return f"ch{int(m.group(1)):02d}" if m else "ch00绪论"

def main():
    dirs = find_chapter_dirs()
    # counts[group][notation][chapter_label] = n
    counts = {g: {n: defaultdict(int) for n in pats} for g, pats in NOTATION_GROUPS.items()}

    for d in dirs:
        label = chapter_label(d)
        text_all = []
        for p in d.rglob("*.md"):
            text_all.append(p.read_text(encoding="utf-8", errors="replace"))
        text = "\n".join(text_all)
        for g, pats in NOTATION_GROUPS.items():
            for nname, pat in pats.items():
                counts[g][nname][label] = len(pat.findall(text))

    lines = ["# 符号/记法一致性审计报告（初版）", ""]
    lines.append("> 用法：先看『章内混用』（阅读最受伤），再定『全书统一目标』写入 `book/meta/notation.md`，")
    lines.append("> Phase 2 逐章把少数派改为主流记法。注意：正则统计有噪声，改前需人工确认上下文。")
    lines.append("")

    mixed_total = 0
    lines.append("## 一、章内混用（同一章出现 ≥2 种互斥记法）")
    lines.append("")
    for g, notations in counts.items():
        mixed_chs = []
        for lbl in sorted({c for n in notations.values() for c in n}):
            used = {n: v[lbl] for n, v in notations.items() if v[lbl] > 0}
            if len(used) >= 2:
                mixed_chs.append((lbl, used))
        if mixed_chs:
            lines.append(f"### {g}")
            lines.append("")
            for lbl, used in mixed_chs:
                mixed_total += 1
                detail = ", ".join(f"`{n}`×{c}" for n, c in sorted(used.items(), key=lambda kv: -kv[1]))
                lines.append(f"- {lbl}: {detail}")
            lines.append("")

    lines.append("## 二、全书记法分布（决定统一目标）")
    lines.append("")
    lines.append("| 记法组 | 记法 | 全书总次数 | 使用章数 |")
    lines.append("|---|---|---|---|")
    for g, notations in counts.items():
        for n, per_ch in notations.items():
            total = sum(per_ch.values())
            n_ch = len([c for c in per_ch.values() if c > 0])
            if total > 0:
                lines.append(f"| {g} | `{n}` | {total} | {n_ch} |")
    lines.append("")

    lines.append(f"**章内混用组合共 {mixed_total} 处（组×章）。**")
    lines.append("")

    META.mkdir(parents=True, exist_ok=True)
    out = META / "audit_notation_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"记法组 {len(NOTATION_GROUPS)} 组，章内混用 {mixed_total} 处（组×章），详见 {out}")

if __name__ == "__main__":
    main()
