# -*- coding: utf-8 -*-
import io, sys
P = r'd:\works\deepinv\第1章 逆问题与贝叶斯推断\1.3 不适定性：为什么逆问题难？.md'
src = io.open(P, encoding='utf-8').read()
REPS = [
(r'''蓝线轻微发钝、红线彻底失控——条件数越大，重建离真相越远。''',
 r'''蓝线只在尾部轻微抖动（抖的是被放大的噪声，不是"变钝"）、红线彻底失控——条件数越大，重建离真相越远。'''),
(r'''我们再次看到，正向算子 $D$ 像低通（压制小奇异值方向），逆向 $D^{-1}$ 像高通（放大小奇异值方向的噪声）。小奇异值 = 高频方向，$1/\sigma_i$ 巨大导致噪声爆炸。这和前面数值微分的例子是同一回事。''',
 r'''我们再次看到，正向算子 $D$ 像低通（压制小奇异值方向），逆向 $D^{-1}$ 像高通（放大小奇异值方向的噪声）。小奇异值 = 高频方向，$1/\sigma_i$ 巨大导致噪声爆炸。这和前面数值微分的例子是同一回事。（按上文脚注的限定：本实验的 $D$ 只保证"$\sigma_i$ 随索引衰减"，"高频"是把奇异向量取成傅里叶基时的读法——那正是实验 1.3-2 用模糊算子要坐实的对应关系。）'''),
]
for i, (old, new) in enumerate(REPS, 1):
    if src.count(old) != 1:
        print(f'[FAIL] {i}: {src.count(old)}'); sys.exit(1)
    src = src.replace(old, new)
io.open(P, 'w', encoding='utf-8', newline='').write(src)
print('OK')
