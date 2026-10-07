# -*- coding: utf-8 -*-
import io, os, re, subprocess

CH = r'd:/works/deepinv/第7章 扩散模型：SDE视角'
files = [
    '7.1 从Langevin到扩散：连续时间推广.md',
    '7.2 正向SDE：从数据到噪声的连续过程.md',
    '7.3 逆向SDE：从噪声到数据的采样过程.md',
    '7.4 概率流ODE：随机采样的确定性等价.md',
    '7.5 数值离散化：从连续方程到可执行算法.md',
    '7.6 实践：用扩散SDE实现图像生成.md',
]
img_line = re.compile(r'^[!<>].*(配套实验|diff --git|index |\+\+\+|---)')
for f in files:
    p = os.path.join(CH, f)
    d = subprocess.run(['git', '-C', r'd:/works/deepinv', 'diff', '--', p],
                       capture_output=True, text=True, encoding='utf-8').stdout
    bad = []
    for ln in d.splitlines():
        if ln[:1] in ('+', '-') and not ln.startswith(('+++', '---')):
            body = ln[1:]
            if '配套实验' not in body:  # 非图片链接行被改动 => 异常
                bad.append(ln)
    # 反斜杠损坏检测: 换行后紧跟 abla / u2212 等
    s = io.open(p, encoding='utf-8').read()
    corrupt = s.count('\nabla')  # 真实换行+abla
    print(f'{f}: 非链接改动行={len(bad)}, 可疑\\nabla损坏={corrupt}')
    for b in bad:
        print('    >>', b[:80])
