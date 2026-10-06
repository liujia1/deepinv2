# -*- coding: utf-8 -*-
"""对比附件图与实验目录内图是否同一版本, 并列出目录状态"""
import sys, io, os, hashlib, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = r'd:\works\deepinv\第12章 Score ≡ ELBO：殊途同归\第十二章 配套实验\实验12.6-1'
ATT = r'C:\Users\admin\AppData\Roaming\QoderCN\SharedClientCache\cache\images\81813445\步骤2_采样结果对比-99bca23f.png'

def h(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()

print('--- 目录状态 ---')
for f in sorted(os.listdir(BASE)):
    p = os.path.join(BASE, f)
    if os.path.isfile(p):
        print(f'{f:32s} {os.path.getsize(p):>9d}  {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(p)))}')

cur = os.path.join(BASE, '步骤2_采样结果对比.png')
print('\n附件图 sha256 :', h(ATT))
print('目录图 sha256 :', h(cur))
print('是否同一文件  :', h(ATT) == h(cur))

# 像素级对比(若尺寸一致)
try:
    import numpy as np
    from PIL import Image
    a = np.asarray(Image.open(ATT).convert('L'), dtype=np.int16)
    b = np.asarray(Image.open(cur).convert('L'), dtype=np.int16)
    if a.shape == b.shape:
        print('像素最大差异:', int(np.abs(a - b).max()))
    else:
        print('尺寸不同:', a.shape, b.shape)
except Exception as e:
    print('像素对比失败:', e)

# JSON是否也变了
js = json.load(open(os.path.join(BASE, 'results_summary.json'), encoding='utf-8'))
print('\nJSON: DSM final_loss=', js['sampling_path_DSM']['final_loss'],
      ' VLB final_loss=', js['variational_path_VLB']['final_loss'],
      ' norms=', js['dual_verification'])
