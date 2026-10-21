# -*- coding: utf-8 -*-
"""
sync_manuscript.py —— 把写作源同步为"成书树"（book/）。

设计原则：
  * 写作源永远是仓库根下的 `第N章 .../` 目录，作者继续在那里写作；
  * `book/manuscript/` 是同步产物（可随时删除重建），供后续排版构建使用；
  * 同步时保持每章目录内部相对结构不变，因此正文中的图片相对引用
    （`./第N章 配套实验/实验X.Y-Z/xxx.png`）在成书树中依然有效，无需重写；
  * 只同步书稿需要的文件（md / py / png / jpg / json / txt），
    不同步数据集、checkpoint、缓存等大文件。

用法：
    python book/tools/sync_manuscript.py          # 同步全部
    python book/tools/sync_manuscript.py --dry    # 只打印计划，不写文件
    python book/tools/sync_manuscript.py --ch 12  # 只同步第12章

输出：
    book/manuscript/ch00/         绪论
    book/manuscript/ch01..ch18/   各章正文（md + 章根图片 + 配套实验）
    book/meta/sync_report.md      同步统计报告
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]  # d:/works/deepinv
BOOK = REPO / "book"
MANUSCRIPT = BOOK / "manuscript"
META = BOOK / "meta"

# 同步到成书树的扩展名（白名单）
INCLUDE_EXT = {".md", ".py", ".png", ".jpg", ".jpeg", ".json", ".txt", ".svg"}
# 这些文件名即使扩展名在白名单内也不同步（实验缓存等）
EXCLUDE_NAMES = {"experiment_cache.pkl", "experiment_cache_S30.pkl",
                 "experiment_checkpoint.pkl", "experiment_checkpoint_v2.pkl"}
# 这些子目录整体跳过
EXCLUDE_DIR_PARTS = {"data", "__pycache__", "Pretrained_models", "old_assets",
                     "old_linear", "stale_bak", "orig_backup", "old_figs_backup",
                     "rundir", "opt_out", "smoke_out", "smoke_out2",
                     "pycache_1261", "rerun"}
# 注：`.chinese/` 不能排除——各实验脚本以 `sys.path.insert(0, '.chinese')`
# 导入其中的 chinese_font.py（字体 .ttf 不在白名单，不会同步）。


def find_chapters():
    """返回 [(chapter_no, chapter_dir, lab_dir)]，按章号排序。"""
    chapters = []
    for d in REPO.iterdir():
        if not d.is_dir():
            continue
        m = re.match(r"^第(\d+)章", d.name)
        if m:
            lab = d / f"第{m.group(1)}章 配套实验"
            if not lab.is_dir():
                # 兜底：找名字含"配套实验"/"配套实现"的子目录（第6章用的是后者）
                cand = [c for c in d.iterdir()
                        if c.is_dir() and ("配套实验" in c.name or "配套实现" in c.name)]
                lab = cand[0] if cand else None
            chapters.append((int(m.group(1)), d, lab))
    chapters.sort()
    return chapters


def iter_files(root: Path):
    """遍历 root 下白名单文件，跳过排除目录/文件。"""
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in INCLUDE_EXT:
            continue
        if p.name in EXCLUDE_NAMES:
            continue
        rel_parts = p.relative_to(root).parts
        if any(part in EXCLUDE_DIR_PARTS for part in rel_parts[:-1]):
            continue
        yield p


def sync_chapter(no: int, src: Path, lab: Path | None, dry: bool):
    dst = MANUSCRIPT / f"ch{no:02d}"
    copied, skipped_ext = 0, {}
    if not dry:
        dst.mkdir(parents=True, exist_ok=True)

    # 1) 章根 md 与图片（不含配套实验目录）
    for p in src.iterdir():
        if p.is_file() and p.suffix.lower() in {".md", ".png", ".jpg", ".jpeg", ".svg"}:
            if not dry:
                shutil.copy2(p, dst / p.name)
            copied += 1

    # 2) 配套实验（保持 `第N章 配套实验/实验X.Y-Z/` 相对结构）
    if lab is not None and lab.is_dir():
        for p in iter_files(lab):
            rel = p.relative_to(lab)
            # results_summary.json 以外的 json 一般是中间产物，仍保留（白名单已控制）
            if not dry:
                target = dst / lab.name / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, target)
            copied += 1

    return copied


def sync_intro(dry: bool):
    src = REPO / "绪论"
    dst = MANUSCRIPT / "ch00"
    if not dry:
        dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for p in src.iterdir():
        if p.is_file() and p.suffix.lower() in {".md", ".png", ".jpg", ".jpeg"}:
            if not dry:
                shutil.copy2(p, dst / p.name)
            copied += 1
    return copied


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="只打印计划，不写文件")
    ap.add_argument("--ch", type=int, default=None, help="只同步指定章号")
    args = ap.parse_args()

    chapters = find_chapters()
    if args.ch is not None:
        chapters = [c for c in chapters if c[0] == args.ch]
        if not chapters:
            print(f"未找到第 {args.ch} 章")
            sys.exit(1)

    total = 0
    report = ["# 成书树同步报告", "",
              f"- 模式：{'DRY RUN（未写入）' if args.dry else '实际同步'}",
              f"- 章数：{len(chapters)}（绪论另计）", ""]

    n = sync_intro(args.dry)
    total += n
    report.append(f"| 章节 | 同步文件数 |")
    report.append(f"|---|---|")
    report.append(f"| ch00 绪论 | {n} |")

    for no, src, lab in chapters:
        n = sync_chapter(no, src, lab, args.dry)
        total += n
        report.append(f"| ch{no:02d} {src.name} | {n} |")

    report.append("")
    report.append(f"**合计同步文件：{total}**")
    report.append("")
    report.append("> 说明：成书树 `book/manuscript/` 为同步产物，可删除后用本脚本重建；")
    report.append("> 写作源仍是仓库根下的各章目录。数据集、checkpoint、审计备份均不同步。")

    if not args.dry:
        META.mkdir(parents=True, exist_ok=True)
        (META / "sync_report.md").write_text("\n".join(report), encoding="utf-8")

    print(f"[{'DRY' if args.dry else 'OK'}] 共 {len(chapters)} 章 + 绪论，同步 {total} 个文件")
    if not args.dry:
        print(f"报告已写入: {META / 'sync_report.md'}")


if __name__ == "__main__":
    main()
