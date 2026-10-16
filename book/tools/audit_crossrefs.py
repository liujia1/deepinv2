# -*- coding: utf-8 -*-
"""
audit_crossrefs.py —— 全书交叉引用审计。

审计对象：仓库根下各章写作源（非 book/manuscript 同步产物）。

检查四类引用的"存在性"：
  1. 章引用   ：`第7章`          → 该章目录是否存在
  2. 节引用   ：`7.3 节` / `第7章7.3节` → 该章下是否存在 `7.3 *.md`
  3. 附录引用 ：`附录12A`        → 该章下是否存在 `附录12A *.md`
  4. 实验引用 ：`实验13.4-2`     → 配套实验下是否存在 `实验13.4-2/13.4-2.py`
  5. 图片引用 ：`![...](<path>)` → 相对 md 所在目录该图片文件是否存在

另外统计三级节引用（如 16.1.4），仅作信息列出（不判定缺失，
因为三级小节在正文内以标题形式存在，验证较复杂，留待人工抽查）。

输出：book/meta/audit_crossrefs_report.md
用法：python book/tools/audit_crossrefs.py
"""
import re
from pathlib import Path
from collections import defaultdict
from urllib.parse import unquote

REPO = Path(__file__).resolve().parents[2]
META = REPO / "book" / "meta"

# ---------- 收集章节结构 ----------

def find_chapters():
    chapters = {}
    for d in REPO.iterdir():
        m = re.match(r"^第(\d+)章", d.name) if d.is_dir() else None
        if m:
            chapters[int(m.group(1))] = d
    return chapters

def chapter_index(chapters):
    """每章的：节号→文件名、附录号→文件名、实验号→路径、md文件列表。"""
    idx = {}
    for no, d in chapters.items():
        secs, aps, mds = {}, {}, []
        for p in d.iterdir():
            if p.suffix == ".md":
                mds.append(p)
                ms = re.match(r"^(\d{1,2})\.(\d{1,2})[\s　]", p.name)
                if ms:
                    secs[(int(ms.group(1)), int(ms.group(2)))] = p
                ma = re.match(r"^附录(\d{1,2})([A-Z])", p.name)
                if ma:
                    aps[(int(ma.group(1)), ma.group(2))] = p
        exps = {}
        for p in d.rglob("*.py"):
            # 兼容两种命名：规范 `X.Y-Z.py` 与带前缀 `实验X.Y-Z.py`（后者记为非规范）
            me = re.match(r"^(?:实验)?(\d{1,2})\.(\d{1,2})-(\d+)\.py$", p.name)
            if me:
                exps[(int(me.group(1)), int(me.group(2)), int(me.group(3)))] = p
        idx[no] = {"secs": secs, "aps": aps, "exps": exps, "mds": mds, "dir": d}
    return idx

# ---------- 抽取引用 ----------

RE_SEC = re.compile(r"(\d{1,2})\.(\d{1,2})\s*节")
# 连接段不允许句读/加号/“章”字，避免把“第 9 章细讲。还记得 8.3 节”这类两个独立引用误连
RE_SEC_WITH_CH = re.compile(r"第\s*(\d{1,2})\s*章[^\n。：；！？+＋章]{0,8}?(\d{1,2})\.(\d{1,2})\s*节")
RE_APP = re.compile(r"附录\s*(\d{1,2})\s*([A-Z])")
RE_EXP = re.compile(r"实验\s*(\d{1,2})\.(\d{1,2})\s*[-–]\s*(\d+)")
RE_IMG = re.compile(r"!\[[^\]]*\]\(<([^>]+)>\)")   # 尖括号式：![...](<path>)
RE_IMG_PLAIN = re.compile(r"!\[[^\]]*\]\((?!\s*<)([^)\n]+)\)")  # 普通式：![...](path)
RE_SUBSEC = re.compile(r"(\d{1,2}\.\d{1,2}\.\d{1,2})")

def main():
    chapters = find_chapters()
    idx = chapter_index(chapters)

    problems = defaultdict(list)   # chapter -> list of (file, line_no, message)
    stats = {"sec_refs": 0, "app_refs": 0, "exp_refs": 0, "img_refs": 0,
             "subsec_refs": 0, "sec_ok": 0, "app_ok": 0, "exp_ok": 0, "img_ok": 0,
             "img_urlencoded": 0}

    def add_problem(ch_no, md, ln, msg):
        rel = md.relative_to(REPO)
        problems[ch_no].append(f"- `{rel}` L{ln}: {msg}")

    for no, info in sorted(idx.items()):
        for md in info["mds"]:
            text = md.read_text(encoding="utf-8", errors="replace")
            for ln, line in enumerate(text.splitlines(), 1):
                # 节引用：优先匹配"第X章X.Y节"形式
                matched_spans = []
                for m in RE_SEC_WITH_CH.finditer(line):
                    ch, s1, s2 = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    matched_spans.append(m.span())
                    stats["sec_refs"] += 1
                    if ch not in idx:
                        add_problem(no, md, ln, f"引用了不存在的章：第{ch}章（{m.group(0)}）")
                    elif (s1, s2) not in idx[ch]["secs"]:
                        add_problem(no, md, ln, f"引用了第{ch}章不存在的节：{ch}.{s2}（{m.group(0)}）")
                    else:
                        stats["sec_ok"] += 1
                # 裸 "X.Y 节"：按当前章解释（若当前章有该节则 OK，否则再尝试跨章解释不判定）
                for m in RE_SEC.finditer(line):
                    if any(a <= m.start() < b for a, b in matched_spans):
                        continue  # 已被"第X章X.Y节"覆盖
                    s1, s2 = int(m.group(1)), int(m.group(2))
                    if s1 != no:
                        continue  # 裸节号属于他章的写法（少见），交由 WITH_CH 形式覆盖
                    stats["sec_refs"] += 1
                    if (s1, s2) not in idx[no]["secs"]:
                        add_problem(no, md, ln, f"本章节引用缺失：{s1}.{s2}（{m.group(0)}）")
                    else:
                        stats["sec_ok"] += 1
                # 附录引用
                for m in RE_APP.finditer(line):
                    ch, ap = int(m.group(1)), m.group(2)
                    stats["app_refs"] += 1
                    if ch not in idx:
                        add_problem(no, md, ln, f"引用了不存在的章：第{ch}章（附录{ch}{ap}）")
                    elif (ch, ap) not in idx[ch]["aps"]:
                        add_problem(no, md, ln, f"引用了第{ch}章不存在的附录：附录{ch}{ap}")
                    else:
                        stats["app_ok"] += 1
                # 实验引用
                for m in RE_EXP.finditer(line):
                    e1, e2, e3 = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    stats["exp_refs"] += 1
                    if e1 not in idx:
                        add_problem(no, md, ln, f"引用了不存在的章：第{e1}章（实验{m.group(0)}）")
                    elif (e1, e2, e3) not in idx[e1]["exps"]:
                        add_problem(no, md, ln, f"引用了不存在的实验：实验{e1}.{e2}-{e3}（配套实验目录或脚本缺失）")
                    else:
                        stats["exp_ok"] += 1
            # 图片引用（两种形式：尖括号式与普通式，全书需统一）
            for ln, line in enumerate(text.splitlines(), 1):
                for m in list(RE_IMG.finditer(line)) + list(RE_IMG_PLAIN.finditer(line)):
                    rel_path = m.group(1).strip()
                    stats["img_refs"] += 1
                    target = (md.parent / rel_path)
                    dec_target = (md.parent / unquote(rel_path))
                    if target.is_file():
                        stats["img_ok"] += 1
                    elif dec_target.is_file():
                        # URL 编码形式（%20 等）：文件存在但路径未解码，计入格式问题
                        stats["img_ok"] += 1
                        stats["img_urlencoded"] += 1
                    else:
                        add_problem(no, md, ln, f"图片缺失：`{rel_path}`")
            # 三级节引用统计（仅信息）
            stats["subsec_refs"] += len(RE_SUBSEC.findall(text))

    # ---------- 报告 ----------
    lines = ["# 交叉引用审计报告", ""]
    lines.append(f"- 审计范围：绪论之外的 18 章写作源，共 {sum(len(i['mds']) for i in idx.values())} 个 md 文件")
    lines.append("")
    lines.append("## 总体统计")
    lines.append("")
    lines.append("| 引用类型 | 总数 | 通过 | 缺失 | 通过率 |")
    lines.append("|---|---|---|---|---|")
    for k, label in [("sec_refs", "节引用（X.Y节）"), ("app_refs", "附录引用"),
                     ("exp_refs", "实验引用"), ("img_refs", "图片嵌入")]:
        total = stats[k]; ok = stats[k.replace("refs", "ok")]
        rate = f"{ok/total*100:.1f}%" if total else "—"
        lines.append(f"| {label} | {total} | {ok} | {total-ok} | {rate} |")
    lines.append("")
    lines.append(f"- 三级节引用（如 16.1.4）出现 {stats['subsec_refs']} 处（未自动验证，建议人工抽查）")
    lines.append("")
    # 图片引用形式统计（成书前需统一为一种）
    n_angle = n_plain = 0
    for no, info in sorted(idx.items()):
        for md in info["mds"]:
            t = md.read_text(encoding="utf-8", errors="replace")
            n_angle += len(RE_IMG.findall(t))
            n_plain += len(RE_IMG_PLAIN.findall(t))
    lines.append(f"- 图片引用形式：尖括号式 `![](<...>)` {n_angle} 处，普通式 `![](...)` {n_plain} 处，其中 URL 编码路径（`%20`）{stats['img_urlencoded']} 处 —— **三种格式问题并存，Phase 5 排版前需统一并去编码**")
    lines.append("")

    n_prob = sum(len(v) for v in problems.values())
    lines.append(f"## 问题清单（共 {n_prob} 处）")
    lines.append("")
    if n_prob == 0:
        lines.append("全部通过，未发现缺失引用。")
    else:
        for ch in sorted(problems):
            lines.append(f"### 第{ch}章（{len(problems[ch])} 处）")
            lines.append("")
            lines.extend(problems[ch])
            lines.append("")

    META.mkdir(parents=True, exist_ok=True)
    out = META / "audit_crossrefs_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"节引用 {stats['sec_refs']}（通过 {stats['sec_ok']}） | "
          f"附录 {stats['app_refs']}（{stats['app_ok']}） | "
          f"实验 {stats['exp_refs']}（{stats['exp_ok']}） | "
          f"图片 {stats['img_refs']}（{stats['img_ok']}） | 问题 {n_prob}")
    print(f"报告: {out}")

if __name__ == "__main__":
    main()
