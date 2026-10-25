# -*- coding: utf-8 -*-
"""B1 试排：把写作源拼成 docx / html，并收集 pandoc 警告。

用法:
  python build_book.py --ch 4          # 单章试排
  python build_book.py --all           # 全书（ch00 + 18 章）
  python build_book.py --all --fmt docx,html
"""
import io, re, sys, subprocess
from pathlib import Path
import pypandoc

REPO = Path(r'd:\works\deepinv')
BUILD = REPO / 'book' / 'build'
PANDOC = pypandoc.get_pandoc_path()
CSL = REPO / 'book' / 'tools' / 'csl' / 'chinese-author-date.csl'
BIB = REPO / 'book' / 'meta' / 'references.bib'
# 去掉叙述式引用括号前的多余空格；必须排在 --citeproc 之后
LUA = REPO / 'book' / 'tools' / 'lua' / 'fix_cite_space.lua'

CH00 = ['前言.md', '绪论.md', '符号总表.md', '术语索引.md', '实验环境.md', '后记.md']

def natkey(p):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', p.name)]

def chapter_dirs():
    ds = [d for d in REPO.iterdir() if d.is_dir() and re.match(r'^第\d+章', d.name)]
    return sorted(ds, key=lambda d: int(re.match(r'^第(\d+)章', d.name).group(1)))

def collect(ch=None):
    files = []
    if ch is None:
        files += [REPO / '绪论' / n for n in CH00 if (REPO / '绪论' / n).exists()]
        for d in chapter_dirs():
            files += sorted(d.glob('*.md'), key=natkey)
    else:
        for d in chapter_dirs():
            if d.name.startswith('第%s章' % ch):
                files += sorted(d.glob('*.md'), key=natkey)
    return [f for f in files if f.exists()]

def main():
    BUILD.mkdir(parents=True, exist_ok=True)
    ch = None
    if '--ch' in sys.argv:
        ch = sys.argv[sys.argv.index('--ch') + 1]
    fmts = ['docx']
    if '--fmt' in sys.argv:
        fmts = sys.argv[sys.argv.index('--fmt') + 1].split(',')
    files = collect(ch)
    # 图片路径是相对章目录写的（./第X章 配套实验/...），故搜索路径须含全部章目录与绪论
    search = [str(d) for d in chapter_dirs()] + [str(REPO / '绪论')]
    tag = ('ch%s' % ch) if ch else 'all'
    log = []
    for fmt in fmts:
        out = BUILD / ('试排_%s.%s' % (tag, fmt))
        cmd = [PANDOC] + [str(f) for f in files] + [
            '-o', str(out), '--standalone', '--citeproc',
            '--csl', str(CSL), '--bibliography', str(BIB),
            '--lua-filter', str(LUA),
            '--resource-path', ';'.join(search),
            '--toc', '--toc-depth=2',
            '-f', 'markdown+tex_math_dollars+raw_tex+pipe_tables+link_attributes-yaml_metadata_block',
        ]
        p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        err = (p.stderr or '')
        math_warn = re.findall(r'Could not convert TeX math (.*?), rendering as TeX', err)
        res_warn = re.findall(r'Could not fetch resource ([^\s]+)', err)
        log += ['### %s  (%s)' % (out.name, 'OK' if p.returncode == 0 else 'FAIL rc=%d' % p.returncode),
                '输入 %d 个文件 ｜ 公式警告 %d ｜ 资源缺失 %d' % (len(files), len(math_warn), len(res_warn))]
        log += ['  MATH: ' + m[:90] for m in math_warn[:60]]
        log += ['  RES : ' + m[:90] for m in set(res_warn)]
        if p.returncode != 0:
            log += ['  STDERR: ' + err[:1500]]
    (BUILD / 'build_log.txt').write_text('\n'.join(log), encoding='utf-8')
    print('\n'.join(log[:12]))

if __name__ == '__main__':
    main()
