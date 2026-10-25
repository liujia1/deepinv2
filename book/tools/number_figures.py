# -*- coding: utf-8 -*-
"""图编号统一：图 N.M　题注，正文图引同步重映射。

规则
  1. 每章按文件自然序（章内全部 md，含附录）顺序编号：图 N.1、图 N.2 …；
  2. 题注文本来源优先级：紧随其后的加粗/斜体题注段 > alt 原文去前缀；
  3. 图题统一为 `![图 N.k　题注](<路径>)`，并删除原独立题注段；
  4. 正文引用 `图 x.y` 按"旧号→新号"映射改写，并统一为 `图 N.k`（数字与"图"之间留一格）；
  5. 无法映射的旧图号写入报告，交人工处理。

用法:
  python number_figures.py --dry     # 只出报告
  python number_figures.py           # 落盘改写
"""
import io, re, sys
from pathlib import Path

REPO = Path(r'd:\works\deepinv')
REPORT = REPO / '.audit_tmp' / '_fig_number_report.md'
GEN = REPO / 'book' / 'tools' / '_generated'
DRY = '--dry' in sys.argv

IMGLINE = re.compile(r'^\s*!\[(.*)\]\((.*)\)\s*$')
CAP_BOLD = re.compile(r'^\s{0,3}(?:>\s*)?\*\*\s*图\s*(\d+(?:[.\-．]\d+)?)?\s*[:：]?\s*\*\*\s*(\S.*)$')
CAP_ITAL = re.compile(r'^\s{0,3}\*\s*图\s*(\d+(?:[.\-．]\d+)?)?\s*[:：]?\s*(\S.*?)\s*\*$')
# 正文图引：前面必须是"如/见/（/、"等引导词，避免"重建图 21.58 dB"这类误命中
REF = re.compile(r'(?<=[如见视（(、，。；：和与及示])图\s?(\d+)\s*[.\-．]\s*(\d+)([a-c])?')
OLDNUM = re.compile(r'^\s*图\s*(\d+(?:[.\-．]\d+)?)\s*[:：]?\s*')
EXPPRE = re.compile(r'^\s*实验\s*[\dA-Za-z]+(?:[.\-．][\dA-Za-z]+)*\s*[：:]?\s*')
SUBNUM = re.compile(r'^\s*图\s*\d+\s*[：:]\s*')


def natkey(p):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', p.name)]


def chapter_dirs():
    ds = [d for d in REPO.iterdir() if d.is_dir() and re.match(r'^第\d+章', d.name)]
    return sorted(ds, key=lambda d: int(re.match(r'^第(\d+)章', d.name).group(1)))


def clean_caption(alt):
    """从 alt 里剥掉旧图号 / 实验号 / 反引号，得到纯题注。"""
    c = alt.strip()
    old = None
    m = OLDNUM.match(c)
    if m and re.search(r'[.\-．]', m.group(1)):
        old = m.group(1).replace('．', '.').replace('-', '.')
        c = c[m.end():]
    else:
        m = EXPPRE.match(c)
        if m:
            c = c[m.end():]
        m = SUBNUM.match(c)
        if m:
            c = c[m.end():]
    c = c.replace('`', '')
    return old, c.strip()


def wrap(path):
    path = path.strip()
    if path.startswith('<') and path.endswith('>'):
        return path
    if ' ' in path or '（' in path:
        return '<%s>' % path
    return path


def process(ch, d, files):
    figs, refs, dangling = [], [], []
    for f in files:
        t = io.open(f, encoding='utf-8', newline='').read()
        sep = '\r\n' if '\r\n' in t else '\n'
        lines = t.split(sep)
        out, i = [], 0
        while i < len(lines):
            m = IMGLINE.match(lines[i])
            if not m:
                out.append(lines[i])
                i += 1
                continue
            alt, path = m.group(1), m.group(2)
            old, cap = clean_caption(alt)
            # 紧随其后的独立题注段
            k = i + 1
            while k < len(lines) and not lines[k].strip():
                k += 1
            if k < len(lines):
                mc = CAP_BOLD.match(lines[k]) or CAP_ITAL.match(lines[k])
                if mc and mc.group(2) and len(mc.group(2)) > 6:
                    num = mc.group(1)
                    # 只有带分隔符的“图 x.y”才是全章号；单独的“图 2”是节内临时号，一律弃用
                    if num and re.search(r'[.\-．]', num):
                        old = old or num.replace('．', '.').replace('-', '.')
                    cap = mc.group(2).replace('`', '').strip()
                    lines = lines[:k] + lines[k + 1:]
            idx = len(figs) + 1
            cap = cap or '（缺题注，待补）'
            figs.append((f.name, old, cap, path))
            out.append('![图 %s.%d　%s](%s)' % (ch, idx, cap, wrap(path)))
            i += 1
        if not DRY:
            io.open(f, 'w', encoding='utf-8', newline='').write(sep.join(out))
        # 正文图引
        body = sep.join(out)
        for mr in REF.finditer(body):
            refs.append((f.name, '%s.%s' % (mr.group(1), mr.group(2)), mr.group(3) or ''))
    mapping = {old: '%s.%d' % (ch, i + 1) for i, (_, old, _, _) in enumerate(figs) if old}
    for fn, o, suf in refs:
        if o in mapping:
            if not DRY:
                fix_ref(ch, fn, o, mapping[o], suf)
        else:
            dangling.append((fn, '图 %s%s' % (o, suf)))
    return figs, mapping, dangling


def fix_ref(ch, fname, old, new, suf):
    p = None
    for d in chapter_dirs():
        if d.name.startswith('第%s章' % ch):
            p = d / fname
    t = io.open(p, encoding='utf-8', newline='').read()
    pat = re.compile(r'图\s?%s\s*([a-c])?' % re.escape(old).replace(r'\.', r'\s*[.\-．]\s*'))
    t2 = pat.sub(lambda m: '图 %s%s' % (new, m.group(1) or ''), t)
    io.open(p, 'w', encoding='utf-8', newline='').write(t2)


def main():
    rep = ['# 图编号报告', '', '| 章 | 图数 | 旧号→新号 | 无法映射的旧引用 |', '|---|---|---|---|']
    allfigs = []
    for d in chapter_dirs():
        ch = re.match(r'^第(\d+)章', d.name).group(1)
        files = sorted(d.glob('*.md'), key=natkey)
        figs, mapping, dangling = process(ch, d, files)
        allfigs += [('第%s章' % ch, k, c, p) for k, (_, _, c, p) in enumerate(figs, 1)]
        rep.append('| %s | %d | %s | %s |' % (
            ch, len(figs),
            ' '.join('%s→%s' % (o, n) for o, n in sorted(mapping.items())) or '—',
            ' '.join('%s:%s' % x for x in dangling) or '—'))
    rep += ['', '## 图目（供生成插图目录）', '']
    for chn, k, cap, p in allfigs:
        rep.append('- %s | 图 %s | %s' % (chn, k, cap[:70]))
    REPORT.write_text('\n'.join(rep), encoding='utf-8')
    print('figs', len(allfigs), 'dry' if DRY else 'written')


if __name__ == '__main__':
    main()
