# -*- coding: utf-8 -*-
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

MD = r"d:\works\deepinv\第2章 先验：贝叶斯推断的灵魂\2.2 经典先验族.md"
with open(MD, encoding='utf-8') as f:
    lines = f.readlines()

append = ('需补两点以免误读：其一，本例被误杀的系数真值仅 −0.16、低于噪声标准差'
          '（σ≈0.38），其观测量级甚至小于 45 个纯噪声分量里的 44 个，已沉在噪声地板之下、'
          '任何阈值都无法把它和噪声区分；其二，Ridge 的"支撑恢复率 100%"是单侧指标'
          '（只数真支撑留住多少、不看误报）造成的假象——它 50 个分量全部非零（含 45 个'
          '纯噪声误报），只因从不做置零决策才"保住"了这个弱峰，并非真的恢复得更好。'
          'λ 扫描也印证：recall 要回到 100% 须把 λ 压到 ≤0.02，而那时置零数归零、'
          'LASSO 便退化得和 Ridge 一样。故"要稀疏就必然牺牲这类低于噪声地板的弱峰"是 L1 '
          '的构造性固有代价，而非调参不当。')

hit = [i for i, l in enumerate(lines) if '反直觉细节' in l]
assert len(hit) == 1, f"anchor found {len(hit)} times"
i = hit[0]
old = lines[i]
assert old.rstrip('\n').endswith('弱峰更少被淹没）。'), "line tail unexpected"
lines[i] = old.rstrip('\n') + append + '\n'
print("appended at line", i + 1)

with open(MD, 'w', encoding='utf-8') as f:
    f.writelines(lines)
print("WRITTEN")
