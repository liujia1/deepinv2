# -*- coding: utf-8 -*-
"""B1 试排：把写作源拼成 docx / html，并收集 pandoc 警告。

结构处理（2026-10-25 新增）：
  - 每章开头注入 "# 第N章 <章名>" 作为一级标题，章内在投喂前把原有标题整体下沉一级，
    于是 docx 得到 章(H1) → 节(H2) → 小节(H3) 的正确层级；
  - 章前分页由 book/tools/lua/docx_layout.lua 负责；
  - 转好的 md 落在 build/_stage/，源文件不被改写。

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
STAGE = BUILD / '_stage'
PANDOC = pypandoc.get_pandoc_path()
CSL = REPO / 'book' / 'tools' / 'csl' / 'chinese-author-date.csl'
BIB = REPO / 'book' / 'meta' / 'references.bib'
# 正文以纯文本提到的文献（无 @key 引注），由 .audit_tmp/_gennocite.py 生成，
# 仅用于让它们进入参考文献表，不产生正文引注。
# 注意：--metadata nocite=@key 传进去的是纯字符串，citeproc 不解析（实测无效）；
# 必须写成 YAML 块标量并经 --metadata-file 注入。
NOCITE = REPO / 'book' / 'meta' / 'nocite.txt'
# 去掉叙述式引用括号前的多余空格；必须排在 --citeproc 之后
LUA = REPO / 'book' / 'tools' / 'lua' / 'fix_cite_space.lua'
# docx 版式：章前分页
LUA_LAYOUT = REPO / 'book' / 'tools' / 'lua' / 'docx_layout.lua'

# 前置件按出版顺序排在正文之前；后记属后置件，排在全书正文之后
FRONT = ['前言.md', '绪论.md', '符号总表.md', '术语索引.md', '实验环境.md']
BACK = ['后记.md']
FENCE = re.compile(r'^```.*?^```', re.S | re.M)


def natkey(p):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', p.name)]


def chapter_dirs():
    ds = [d for d in REPO.iterdir() if d.is_dir() and re.match(r'^第\d+章', d.name)]
    return sorted(ds, key=lambda d: int(re.match(r'^第(\d+)章', d.name).group(1)))


def read(p):
    return io.open(p, encoding='utf-8', newline='').read()


def sink_heading(text):
    """把代码块之外的 ATX 标题整体下沉一级（# -> ##）。"""
    def repl(m):
        return '#' + m.group(0)
    out, pos = [], 0
    for m in FENCE.finditer(text):
        out.append(re.sub(r'^(#{1,5})\s', repl, text[pos:m.start()], flags=re.M))
        out.append(m.group(0))
        pos = m.end()
    out.append(re.sub(r'^(#{1,5})\s', repl, text[pos:], flags=re.M))
    return ''.join(out)


def stage(num, d, files):
    """生成暂存 md：章标题 + 各节（标题下沉一级）。"""
    STAGE.mkdir(parents=True, exist_ok=True)
    out = STAGE / ('ch%02d.md' % num)
    parts = ['# %s\n' % d.name.rstrip()]
    for f in files:
        parts.append(sink_heading(read(f)).strip('\n') + '\n')
    io.open(out, 'w', encoding='utf-8', newline='\n').write('\n'.join(parts))
    return out


def collect(ch=None):
    """返回待投喂文件；单章模式返回一个暂存文件。"""
    if ch is None:
        files = [REPO / '绪论' / n for n in FRONT if (REPO / '绪论' / n).exists()]
        for d in chapter_dirs():
            num = int(re.match(r'^第(\d+)章', d.name).group(1))
            files.append(stage(num, d, sorted(d.glob('*.md'), key=natkey)))
        files += [REPO / '绪论' / n for n in BACK if (REPO / '绪论' / n).exists()]
        return [f for f in files if f.exists()]
    for d in chapter_dirs():
        if d.name.startswith('第%s章' % ch):
            num = int(re.match(r'^第(\d+)章', d.name).group(1))
            return [stage(num, d, sorted(d.glob('*.md'), key=natkey))]
    return []


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
    extra = []
    if NOCITE.exists():
        keys = [k.strip() for k in NOCITE.read_text(encoding='utf-8').split('\n') if k.strip()]
        if keys:
            meta = BUILD / '_nocite_meta.yaml'
            meta.write_text('nocite: |\n' + ''.join('  @%s\n' % k for k in keys), encoding='utf-8')
            extra = ['--metadata-file', str(meta)]
    for fmt in fmts:
        out = BUILD / ('试排_%s.%s' % (tag, fmt))
        cmd = [PANDOC] + [str(f) for f in files] + extra + [
            '-o', str(out), '--standalone', '--citeproc',
            '--csl', str(CSL), '--bibliography', str(BIB),
            '--lua-filter', str(LUA),
            '--lua-filter', str(LUA_LAYOUT),
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
