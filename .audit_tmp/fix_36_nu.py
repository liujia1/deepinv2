# -*- coding: utf-8 -*-
import io
p = r"d:\works\deepinv\第3章 从MAP到后验探索\3.6 收敛性分析与正则化参数选择.md"
with io.open(p, "r", encoding="utf-8") as f:
    s = f.read()

reps = [
    ("$\\phi(t)=t^{\nu/2}$", "PLACEHOLDER1"),
]
# The corruption turned each "\nu" into a real newline followed by "u".
# Repair the specific fragments in the 源条件 paragraph.
fixes = [
    ("t^{\nu/2}", None),  # target for reconstruction
]

# Direct string repairs: replace newline+u fragments back to \nu
bad_frags = [
    ("$\\phi(t)=t^{\nu/2}$", "$\\phi(t)=t^{\nu/2}$"),
]

# Build the corrupted (as-in-file) and correct strings explicitly.
corrupt = (
    "参数选好后，误差随 $\\delta$ 如何衰减由源条件定（Lorenz/Korolev & Latz）："
    "若 $x^\\dagger=\\phi(A^*A)x_0$，$\\phi(t)=t^{\nu/2}$，则 Tikhonov（资格 $\\nu=1$）"
    "达 H\u00f6lder 速率 $O(\\delta^{1/2})$；迭代 Tikhonov（$\\nu=2$）达 $O(\\delta^{2/3})$；"
    "无源条件退化为对数速率（Tikhonov 饱和效应）。Landweber 在 a-priori 停止 "
    "$m^*\\propto\\delta^{-2/(\\nu+1)}$ 时阶最优（Lorenz Thm 11.6）"
    "——把 3.3 半收敛、Morozov 早停、源条件速率串成一线。"
)

if corrupt in s:
    print("correct form already present, nothing to fix")
else:
    # reconstruct the corrupted version: \nu -> newline + u
    corrupted = corrupt.replace("\\nu", "\nu")
    # careful: replace "\nu" (backslash-n-u) with newline + "u"
    if corrupted in s:
        s = s.replace(corrupted, corrupt)
        with io.open(p, "w", encoding="utf-8") as f:
            f.write(s)
        print("FIXED: restored \\nu in source-condition paragraph")
    else:
        print("ERROR: neither form found; manual inspection needed")
        # show context
        idx = s.find("收敛速率与源条件")
        print(repr(s[idx:idx+400]))
