# -*- coding: utf-8 -*-
"""
normalize_img_refs.py —— 全书图片引用格式统一（Phase 2 内容整备）。

目标：把三种并存的图片引用形式统一为**尖括号式** `![alt](<path>)`：
  1. 尖括号式 `![alt](<p%20th>)` → 路径 URL 解码：`![alt](<p th>)`
  2. 普通式   `![alt](p%20th)`  → 解码 + 加尖括号：`![alt](<p th>)`
  3. 普通式   `![alt](p th)`    → 加尖括号：`![alt](<p th>)`

为什么选尖括号式：CommonMark 标准语法、支持含空格路径、无需 URL 编码、
人类可读；全书当前 141 处已是该形式（占比最高），改动面最小。

安全策略：
  - 只改写"解码后文件确实存在"的引用；解码后文件不存在的保留原样并报告
    （交由 audit_crossrefs.py 继续暴露问题）。
  - http(s) 外链图片不动。
  - HTML <img> 标签不动（仅 2 个文件 8 处，Phase 5 排版时再决定去留）。
  - 默认 dry-run，加 --apply 才写盘。

用法：
  python book/tools/normalize_img_refs.py            # 预览
  python book/tools/normalize_img_refs.py --apply    # 执行
"""
import re
import sys
from pathlib import Path
from urllib.parse import unquote

REPO = Path(__file__).resolve().parents[2]

RE_IMG_ANGLE = re.compile(r"!\[([^\]]*)\]\(<([^>\n]+)>\)")
RE_IMG_PLAIN = re.compile(r"!\[([^\]]*)\]\((?!\s*<)([^)\n]+)\)")


def find_chapter_dirs():
    out = []
    for d in REPO.iterdir():
        if d.is_dir() and re.match(r"^第\d+章", d.name):
            out.append(d)
    return out


def normalize_file(md: Path, apply: bool):
    """返回 (统计dict, 变更行列表)。"""
    text = md.read_text(encoding="utf-8", errors="replace")
    orig = text
    stat = {"angle_decoded": 0, "plain_to_angle": 0, "skipped_missing": 0,
            "skipped_remote": 0, "unchanged": 0}
    changes = []

    def resolve_ok(path_str: str) -> bool:
        """解码后的相对路径能否在 md 所在目录下找到文件。"""
        p = unquote(path_str).strip()
        if p.startswith(("http://", "https://")):
            return False
        return (md.parent / p).is_file()

    def repl_angle(m):
        alt, path = m.group(1), m.group(2)
        if path.strip().startswith(("http://", "https://")):
            stat["skipped_remote"] += 1
            return m.group(0)
        dec = unquote(path)
        if dec != path:
            if resolve_ok(path):
                stat["angle_decoded"] += 1
                return f"![{alt}](<{dec}>)"
            stat["skipped_missing"] += 1
            changes.append(f"  L?: 尖括号式解码后文件缺失，保留原样：{path}")
            return m.group(0)
        stat["unchanged"] += 1
        return m.group(0)

    def repl_plain(m):
        alt, path = m.group(1), m.group(2).strip()
        if path.startswith(("http://", "https://")):
            stat["skipped_remote"] += 1
            return m.group(0)
        if not resolve_ok(path):
            stat["skipped_missing"] += 1
            changes.append(f"  L?: 普通式目标文件缺失，保留原样：{path}")
            return m.group(0)
        dec = unquote(path)
        stat["plain_to_angle"] += 1
        return f"![{alt}](<{dec}>)"

    text = RE_IMG_ANGLE.sub(repl_angle, text)
    text = RE_IMG_PLAIN.sub(repl_plain, text)

    if text != orig and apply:
        md.write_text(text, encoding="utf-8", newline="\n")
    return stat, changes, (text != orig)


def main():
    apply = "--apply" in sys.argv
    total = {"angle_decoded": 0, "plain_to_angle": 0, "skipped_missing": 0,
             "skipped_remote": 0, "unchanged": 0}
    n_files_changed = 0
    all_changes = []
    for ch in sorted(find_chapter_dirs()):
        for md in sorted(ch.glob("*.md")):
            stat, changes, changed = normalize_file(md, apply)
            for k in total:
                total[k] += stat[k]
            if changed:
                n_files_changed += 1
                all_changes.append(f"[变更] {md.relative_to(REPO)}")
                all_changes.extend(changes)
    mode = "APPLY（已写盘）" if apply else "DRY-RUN（预览，加 --apply 执行）"
    print(f"模式：{mode}")
    print(f"变更文件数：{n_files_changed}")
    print(f"普通式 → 尖括号式：{total['plain_to_angle']} 处")
    print(f"尖括号式 URL 解码：{total['angle_decoded']} 处")
    print(f"已是规范形式（未动）：{total['unchanged']} 处")
    print(f"跳过（外链）：{total['skipped_remote']} 处")
    print(f"跳过（解码后文件缺失，保留原样待查）：{total['skipped_missing']} 处")
    if total["skipped_missing"]:
        print("\n需人工核实的引用：")
        for line in all_changes:
            if "缺失" in line or "保留原样" in line:
                print(line)


if __name__ == "__main__":
    main()
