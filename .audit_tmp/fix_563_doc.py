# -*- coding: utf-8 -*-
import io
p = r"d:\works\deepinv\第5章 朗之万动力学与得分函数\第五章配套实验\实验5.6-3\5.6-3.py"
with io.open(p, "r", encoding="utf-8") as f:
    s = f.read()

old = (
"前置实验 / 依赖：\n"
"  - 需要先运行实验5.5-1，生成 sampling_results.npz（本实验读取其中的\n"
"    post_mean / mc_samples / x_true / y_obs）\n"
"  - 步骤2/6 的真实 MAP 与参数扫描需要 5.5-1 目录下的 sampling_tools 与预训练去噪器；\n"
"    若缺失则自动回退到旧版行为（y_obs 作基准 + 理论示意曲线），并打印警告。\n"
"\n"
"素材来源：\n"
"  - 实验5.5-1.py（前向算子/去噪器/PnP-ULA 设定）\n"
"  - Terris et al. (2022); Laumont et al. (2021)"
)
new = (
"前置条件 / 依赖（已自包含，无需先跑 5.5-1）：\n"
"  - 本实验自己完成“加载图像→模糊→加噪→PnP-ULA 采样”，直接生成 MMSE(后验均值)\n"
"    与后验样本，不再读取 5.5-1 的 sampling_results.npz。\n"
"  - 所需资源已随本目录自带：sampling_tools/、cman.png；去噪器权重缺失会自动下载。\n"
"  - 与 5.5-1 同种子、同顺序，GPU 上可复现 MMSE PSNR≈26.57 dB。\n"
"\n"
"素材来源：\n"
"  - 前向算子/去噪器/PnP-ULA 设定沿用 sampling_tools（与实验5.5-1 同源）\n"
"  - Terris et al. (2022); Laumont et al. (2021)"
)
if old in s:
    s = s.replace(old, new)
    with io.open(p, "w", encoding="utf-8") as f:
        f.write(s)
    print("DOCSTRING_UPDATED")
else:
    print("PATTERN_NOT_FOUND")
