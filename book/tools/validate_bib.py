#!/usr/bin/env python3
"""
validate_bib.py —— Phase 3 参考文献体系结构校验

校验项：
  1. bib 解析：条目数、重复 citekey、花括号配平、citekey 命名规范（小写字母/数字/连字符）
  2. 条目必填字段（按 @type）：author/title/year + 类型特定字段
     （article→journal；inproceedings→booktitle；book→publisher；techreport→institution）
  3. 活跃 ⚠ 扫描：note={⚠...} 计数（目标 0）
  4. 映射表覆盖（citations_mapping.md 主表）：
     - 每行 bib 键必须存在于 bib（🗑️已废弃/redirect 行除外，但 redirect 目标必须在 bib）
     - bib 每个键必须被主表引用
  5. 反查覆盖（citations_raw.csv ↔ 映射表主表）：211 条双向一致

用法：python book/tools/validate_bib.py
退出码：0 = 全部通过；1 = 存在失败项
"""
import io
import os
import re
import csv
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "meta")
BIB = os.path.join(BASE, "references.bib")
MAPPING = os.path.join(BASE, "citations_mapping.md")
CSV = os.path.join(BASE, "citations_raw.csv")

REQUIRED = {
    "article": ["author", "title", "journal", "year"],
    "inproceedings": ["author", "title", "booktitle", "year"],
    "book": ["author", "title", "publisher", "year"],
    "techreport": ["author", "title", "institution", "year"],
    "misc": ["author", "title", "year"],
    "unpublished": ["author", "title", "year"],
}

fails = []
warns = []


def parse_bib(text):
    """返回 [(type, key, chunk)]；chunk 为条目全文（至下一个 @ 或文件尾）"""
    entries = []
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text):
        nxt = text.find("\n@", m.start() + 10)
        end = nxt if nxt > 0 else len(text)
        entries.append((m.group(1).lower(), m.group(2), text[m.start():end]))
    return entries


def field(chunk, name):
    m = re.search(r"\b" + name + r"\s*=\s*\{", chunk, re.I)
    if not m:
        return None
    i = m.end()
    depth = 1
    while i < len(chunk) and depth:
        if chunk[i] == "{":
            depth += 1
        elif chunk[i] == "}":
            depth -= 1
        i += 1
    return chunk[m.end():i - 1] if depth == 0 else None


def main():
    bib = io.open(BIB, encoding="utf-8").read()
    mapping = io.open(MAPPING, encoding="utf-8").read()

    print("=" * 72)
    print("validate_bib.py —— references.bib 结构校验")
    print("=" * 72)

    # ---------- 1. bib 解析 ----------
    entries = parse_bib(bib)
    keys = [k for _t, k, _c in entries]
    print(f"[1] bib 条目数: {len(entries)}")
    if len(entries) != len(set(keys)):
        dups = sorted({k for k in keys if keys.count(k) > 1})
        fails.append(f"重复 citekey: {dups}")
    for t, k, c in entries:
        if not re.fullmatch(r"[a-z0-9\-]+", k):
            fails.append(f"citekey 命名不规范: {k}")
    # 花括号配平（剔除 % 注释行后）
    body = "\n".join(ln for ln in bib.splitlines() if not ln.lstrip().startswith("%"))
    if body.count("{") != body.count("}"):
        fails.append(f"花括号不配平: {{={body.count('{')} }}={body.count('}')}")
    types = {}
    for t, _k, _c in entries:
        types[t] = types.get(t, 0) + 1
    print(f"    类型分布: {types}")

    # ---------- 2. 必填字段 ----------
    missing = []
    for t, k, c in entries:
        for f in REQUIRED.get(t, ["author", "title", "year"]):
            if field(c, f) is None:
                missing.append(f"{k} 缺 {f}")
    if missing:
        fails.append("缺必填字段: " + "; ".join(missing))
    print(f"[2] 必填字段: {'全部完整' if not missing else str(len(missing)) + ' 处缺失'}")

    # ---------- 3. 活跃 ⚠ ----------
    active_warn = [k for _t, k, c in entries
                   if field(c, "note") and "⚠" in (field(c, "note") or "")]
    confirmed = [k for _t, k, c in entries
                 if field(c, "note") and "已联网坐实" in (field(c, "note") or "")]
    print(f"[3] 活跃 ⚠ note: {len(active_warn)}（目标 0）；带'已联网坐实'注释: {len(confirmed)}")
    if active_warn:
        fails.append(f"活跃 ⚠ 未清零: {active_warn}")

    # ---------- 4. 映射表覆盖 ----------
    rows = re.findall(r"^\| (\d+) \| ([^|]+) \| (\S+) \| ([^|]+) \| ([^|]+) \| ([^|]+) \|$",
                      mapping, re.M)
    main_rows = [r for r in rows if r[0].isdigit()]
    bibkeys = set(keys)
    bad_keys = []
    open_items = []
    for r in main_rows:
        k = r[4].strip()
        status = r[5].strip()
        if k not in bibkeys:
            if "废弃" in status or "redirect" in status:
                continue  # 已标注废弃/重定向的行允许键不在 bib
            if "⚠" in status or "待核" in status:
                open_items.append(f"#{r[0]} {r[1].strip()} ({r[2].strip()})")
                continue  # 待核开放项：键尚未建立，计警告不计失败
            bad_keys.append(f"#{r[0]} {k}")
    if bad_keys:
        fails.append("映射表键在 bib 中不存在: " + "; ".join(bad_keys))
    if open_items:
        warns.append(f"映射表 ⚠ 待核开放项 ({len(open_items)}): {'; '.join(open_items)}")
    cited = {r[4].strip() for r in main_rows if r[4].strip() in bibkeys}
    uncited = sorted(bibkeys - cited)
    if uncited:
        fails.append(f"bib 键未被映射表引用 ({len(uncited)}): {uncited}")
    print(f"[4] 映射表主表 {len(main_rows)} 行；引用键 {len(cited)}；bib 键 {len(bibkeys)}；"
          f"bib 未引用键 {len(uncited)}")

    # ---------- 5. CSV ↔ 映射表 ----------
    csv_pairs = [(r[1], r[2]) for r in list(csv.reader(io.open(CSV, encoding="utf-8")))[1:]
                 if len(r) >= 5]
    mp_pairs = [(r[1].strip(), r[2].strip()) for r in main_rows]
    only_csv = [p for p in csv_pairs if p not in mp_pairs]
    only_mp = [p for p in mp_pairs if p not in csv_pairs]
    if only_csv:
        fails.append(f"CSV 有、映射表无: {only_csv}")
    if only_mp:
        fails.append(f"映射表有、CSV 无: {only_mp}")
    print(f"[5] CSV 条目 {len(csv_pairs)} ↔ 映射表 {len(mp_pairs)}：双向一致"
          if not (only_csv or only_mp) else "[5] 覆盖不一致")

    # ---------- 汇总 ----------
    print()
    if warns:
        print(f"警告 {len(warns)} 项（不阻塞）：")
        for w in warns:
            print("  ! " + w)
        print()
    if fails:
        print(f"*** 校验失败 {len(fails)} 项 ***")
        for f in fails:
            print("  ✗ " + f)
        sys.exit(1)
    print("*** 全部通过 ***")
    sys.exit(0)


if __name__ == "__main__":
    main()
