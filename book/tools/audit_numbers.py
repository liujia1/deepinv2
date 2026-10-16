# -*- coding: utf-8 -*-
"""
audit_numbers.py —— 正文数值 ↔ results_summary.json 对账（初版）。

原理：
  * 每个实验目录有 results_summary.json（脚本写出的结构化数值）；
  * 正文 md 中引用该实验的节文件里，应出现这些数值（经四舍五入）；
  * 本脚本对每个 json 数值生成若干"截断候选串"（3/4/5 位有效数字、
    百分比、科学计数等），在相关 md 文本中搜索；
  * 输出每个实验的覆盖率 + 未匹配数值样例，供人工核查
    （未匹配 ≠ 错误：可能正文只引用了关键数字，也可能存在抄写错误）。

审计对象：仓库根下各章写作源。
输出：book/meta/audit_numbers_report.md
用法：python book/tools/audit_numbers.py
"""
import json
import re
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[2]
META = REPO / "book" / "meta"

RE_EXP_REF = re.compile(r"实验\s*(\d{1,2})\.(\d{1,2})\s*[-–]\s*(\d+)")

def find_chapters():
    out = {}
    for d in REPO.iterdir():
        m = re.match(r"^第(\d+)章", d.name) if d.is_dir() else None
        if m:
            out[int(m.group(1))] = d
    return out

def find_lab_dir(ch_no: int, ch_dir: Path):
    """找配套实验目录：目录名为中文数字（如“第一章 配套实验”）或阿拉伯数字。"""
    for cand in ch_dir.iterdir():
        if cand.is_dir() and "配套实验" in cand.name:
            return cand
    return ch_dir / f"第{ch_no}章 配套实验"

def flatten_numbers(obj, prefix=""):
    """递归展开 json，产出 [(path, float_value)]。"""
    vals = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            vals.extend(flatten_numbers(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            vals.extend(flatten_numbers(v, f"{prefix}[{i}]"))
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        vals.append((prefix, float(obj)))
    return vals

def candidates(x: float):
    """为数值生成正文中可能出现的字符串候选。"""
    out = set()
    if x == 0:
        return {"0", "0.0"}
    # 一般表示（不同小数位）
    for nd in range(1, 7):
        s = f"{x:.{nd}f}"
        out.add(s)
        if s.startswith("0."):
            out.add(s[1:])  # .xx 形式
    # 有效数字截断（科学计数法场景）
    try:
        for sig in range(2, 6):
            from decimal import Decimal
            q = float(f"{x:.{sig}g}")
            for nd in range(1, 7):
                out.add(f"{q:.{nd}f}")
    except Exception:
        pass
    # 百分比
    p = x * 100
    for nd in range(1, 3):
        out.add(f"{p:.{nd}f}%")
        out.add(f"{p:.{nd}f} %")
    # 整数
    if abs(x - round(x)) < 1e-9:
        out.add(str(int(round(x))))
    return {c for c in out if len(c.replace(".", "").replace("-", "").replace("%", "")) >= 2}

def main():
    chapters = find_chapters()
    report = []
    total_json_vals, total_matched = 0, 0
    experiments = []

    for no, d in sorted(chapters.items()):
        lab = find_lab_dir(no, d)
        if not lab.is_dir():
            continue
        # 章内全部 md 文本（正文引用实验的位置可能在任何节）
        md_texts = {}
        for p in d.rglob("*.md"):
            if "配套实验" in str(p.relative_to(d)):
                continue
            md_texts[p.name] = p.read_text(encoding="utf-8", errors="replace")

        for exp_dir in sorted(lab.iterdir()):
            if not exp_dir.is_dir():
                continue
            js = exp_dir / "results_summary.json"
            if not js.is_file():
                continue
            exp_name = exp_dir.name  # 实验X.Y-Z
            m = re.match(r"^实验(\d{1,2}\.\d{1,2}-\d+)$", exp_name)
            if not m:
                continue
            key = m.group(1)
            # 找出引用了该实验的 md
            refs = [fname for fname, t in md_texts.items()
                    if RE_EXP_REF.search(t) and RE_EXP_REF.search(t).group(0).endswith(key.split("实验")[-1])
                    or f"实验{key}" in t or f"实验 {key}" in t]
            hay = "\n".join(md_texts.get(f, "") for f in refs) or "\n".join(md_texts.values())

            vals = flatten_numbers(json.loads(js.read_text(encoding="utf-8")))
            matched, unmatched = 0, []
            for path, x in vals:
                cands = candidates(x)
                if any(c in hay for c in cands):
                    matched += 1
                else:
                    unmatched.append((path, x))
            total_json_vals += len(vals)
            total_matched += matched
            experiments.append((no, key, len(refs), len(vals), matched, unmatched))

    # ---------- 报告 ----------
    lines = ["# 数值对账审计报告（初版）", ""]
    lines.append("> 目的：检查正文叙述的数字是否忠于实验输出（results_summary.json）。")
    lines.append("> 注意：『未匹配』不等于错误——正文通常只引用关键数值；")
    lines.append("> 但若某实验覆盖率异常低，或关键指标未匹配，值得人工比对。")
    lines.append("")
    lines.append(f"- 实验数（含 results_summary.json）：{len(experiments)}")
    lines.append(f"- json 数值总数：{total_json_vals}，在正文找到匹配：{total_matched}"
                 f"（{total_matched/total_json_vals*100:.1f}%）" if total_json_vals else "- json 数值总数：0")
    lines.append("")

    # 低覆盖 / 无引用实验
    lines.append("## 一、需要关注（无正文引用 或 覆盖率 < 30%）")
    lines.append("")
    flagged = 0
    for no, key, n_refs, n_vals, matched, unmatched in experiments:
        cov = matched / n_vals if n_vals else 1
        if n_refs == 0 or cov < 0.30:
            flagged += 1
            lines.append(f"- ch{no:02d} 实验{key}：正文引用文件数 {n_refs}，json 数值 {n_vals}，"
                         f"匹配 {matched}（覆盖率 {cov*100:.0f}%）")
    if flagged == 0:
        lines.append("（无）")
    lines.append("")

    # 分章明细
    lines.append("## 二、分实验明细")
    lines.append("")
    lines.append("| 章 | 实验 | 正文引用md数 | json数值 | 匹配 | 覆盖率 |")
    lines.append("|---|---|---|---|---|---|")
    for no, key, n_refs, n_vals, matched, unmatched in experiments:
        cov = f"{matched/n_vals*100:.0f}%" if n_vals else "—"
        lines.append(f"| ch{no:02d} | {key} | {n_refs} | {n_vals} | {matched} | {cov} |")
    lines.append("")

    META.mkdir(parents=True, exist_ok=True)
    out = META / "audit_numbers_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    pct = f"{total_matched/total_json_vals*100:.1f}%" if total_json_vals else "n/a"
    print(f"实验 {len(experiments)} 个，json 数值 {total_json_vals}，匹配 {total_matched}"
          f"（{pct}），关注项 {flagged}")
    print(f"报告: {out}")

if __name__ == "__main__":
    main()
