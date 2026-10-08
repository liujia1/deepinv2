# -*- coding: utf-8 -*-
"""实证 tqdm 写入侧行为（注意：tqdm 默认写 stderr，不是 stdout）：
1) 跑满的条是否一定写出 100% 帧，且末尾被换行封口（leave=True）
2) 非 tty（近似 Colab 的 OutStream.isatty()==False）下是否同样写出 100%+换行
3) tqdm.auto 在纯脚本环境是否回退为标准文本条
"""
import sys


class FakeStream:
    def __init__(self, tty):
        self.parts = []
        self._tty = tty

    def write(self, s):
        self.parts.append(s)
        return len(s)

    def flush(self):
        pass

    def isatty(self):
        return self._tty


def capture(tty, fn):
    buf = FakeStream(tty)
    old = sys.stderr
    sys.stderr = buf
    try:
        fn()
    finally:
        sys.stderr = old
    return ''.join(buf.parts)


# 实验1：tty 模拟，两条背靠背
def two_bars():
    from tqdm import tqdm
    for _ in tqdm(range(30), desc='A', mininterval=0.01):
        pass
    for _ in tqdm(range(30), desc='B', mininterval=0.01):
        pass


raw = capture(True, two_bars)
a_part, b_part = raw.split('B:', 1)
print('=== 实验1：isatty=True，两条背靠背 ===')
print('A 段是否写出 100% 帧:', '100%' in a_part)
print('A 段末尾60字符:', repr(a_part[-60:]))
print('A 末帧后被换行封口:', '\n' in a_part[-60:])
print()

# 实验2：非 tty（接近 Colab），3000 步、默认 mininterval
def long_bar():
    from tqdm import tqdm
    for _ in tqdm(range(3000), desc='C'):
        pass


raw2 = capture(False, long_bar)
frames = raw2.split('\r')
print('=== 实验2：isatty=False，3000步默认参数 ===')
print('写入帧数(按 \\r 分段):', len(frames))
print('是否写出 100% 帧:', '100%' in raw2)
print('流末尾3字符:', repr(raw2[-3:]))
print()

# 实验3：tqdm.auto 回退验证
import tqdm as _t
from tqdm.auto import tqdm as _auto
print('=== 实验3：tqdm.auto 回退 ===')
print('纯脚本下 tqdm.auto 即标准文本 tqdm:', _auto is _t.tqdm)
