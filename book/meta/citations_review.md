# 正文引用反查清单（Phase 3 输入）

> **2026-10-21 更新（权威源已转移）**：本清单是反查阶段的输入快照； bib 与正文引用的最终对应关系以 **`citations_mapping.md`（211 行 ↔ 196 键，1:1）** 和 **`references.bib`** 为准（经 `book/tools/validate_bib.py` 校验）。本文"人工复核注意项"中下列事项已有结论：
> - A.1 "Holden et al (2022) 与 Holden, Pereyra & Zygalakis (2022) 可能不同篇" → **同一篇**（SIAM J. Imaging Sci. 15(2):892–924, 2022，arXiv:2103.10182），已合并为 `holden2022datadriven`；
> - A.3 "Chung et al (2209.14687) 疑与 Chung et al (2023) 为同一篇" → **已确认合并**入 `chung2023dps`（ICLR 2023）；
> - B.1 "Daras et al. Survey (2410.00083) 需人工补题名" → 已坐实为 `daras2024survey`（arXiv:2410.00083）；
> - B/C 节其余非正式来源处置、A.2 同名消歧结论，见 `citations_mapping.md` 第一/三节与 `verify_checklist.md` 2026-10-21 修订日志。
> 另：正文两处**张冠李戴**待 Phase 5 修正——13.3#630 "Wu et al. (2024)" 实为 Trippe et al. (2022)；12.7#158 与 13.3#742 "Chung et al. (2025)" 实为 Daras et al. (2024) 综述。

> 工具：`book/tools/scan_citations.py`（2026-10-18 升级 v3：支持连字符姓氏跨大写、中文连接符多作者、"等人"式、APA 全名式/斜体文献式、arXiv 编号式、括号内前缀/后缀/会场名、同括号多引用拆分；作者串自动清洗：剥首字母缩写、去附录编号残留、标题/会场剥离、中文连接符归一、模型缩写过滤）。
> 扫描范围：`book/manuscript/` 全部 180 个正文 md（排除代码块、行内代码、`$...$` 公式）。
> 逐条上下文样例见 `.audit_tmp/scan_citations_out_utf8.txt`（旧版）与 `.audit_tmp/scan_v6_out.txt`（升级版）；机器可读版见 `citations_raw.csv`。
> 升级依据：独立交叉复核（year-first 扫描器 + APA/形态枚举，脚本见 `.audit_tmp/indep_citation_scan.py`、`full_enum.py`、`pattern_enum.py`、`apa_enum.py`）；旧版工具漏报 36 条、截断 23 条，升级前后双向 diff 见 `.audit_tmp/diff_final.txt`。

- 去重条目数：**211**；总出现次数：**602**
- 与旧版（198 条 / 523 次）差异：**+36 新增**（原漏报）、**−23 移除**（截断 artifact 被正确合并）、**48 条计数上调**（原少计的实例）

| 章 | 条目数（按首见位置） |
|---|---|
| ch01 | 4 |
| ch02 | 7 |
| ch03 | 3 |
| ch04 | 23 |
| ch05 | 20 |
| ch06 | 8 |
| ch07 | 14 |
| ch08 | 14 |
| ch09 | 13 |
| ch10 | 1 |
| ch11 | 2 |
| ch12 | 1 |
| ch13 | 25 |
| ch14 | 31 |
| ch15 | 11 |
| ch16 | 13 |
| ch17 | 20 |
| ch18 | 1 |

## 全部条目（按年份排序）

| # | Author | Year | 次数 | 首见位置 |
|---|---|---|---|---|
| 1 | Bloch | 1946 | 1 | `ch16\附录16B Bloch方程的完整推导.md#71` |
| 2 | Frankel | 1950 | 2 | `ch04\4.6 加速采样方法：过松弛与惯性Langevin算法.md#26` |
| 3 | Robbins | 1956 | 5 | `ch05\5.3 Tweedie等式：从去噪器到得分函数.md#206` |
| 4 | Miyasawa | 1961 | 2 | `ch05\5.3 Tweedie等式：从去噪器到得分函数.md#207` |
| 5 | Polyak | 1964 | 2 | `ch04\4.6 加速采样方法：过松弛与惯性Langevin算法.md#67` |
| 6 | Moreau | 1965 | 1 | `ch05\5.4 MAP与MMSE的结构对偶性.md#245` |
| 7 | Hastings | 1970 | 1 | `ch04\附录4A 马尔可夫链收敛性与Gibbs分布.md#128` |
| 8 | Rockafellar | 1970 | 1 | `ch08\附录8A Fenchel共轭与ELBO的优化理论根基.md#183` |
| 9 | Hudson | 1978 | 2 | `ch17\附录17A SURE的无偏性证明.md#97` |
| 10 | Adler | 1981 | 2 | `ch04\4.6 加速采样方法：过松弛与惯性Langevin算法.md#32` |
| 11 | Stein | 1981 | 3 | `ch13\13.6 闭环：回到第1章的逆问题.md#284` |
| 12 | Anderson | 1982 | 14 | `ch07\7.1 从Langevin到扩散：连续时间推广.md#78` |
| 13 | Shepp & Vardi | 1982 | 1 | `ch16\附录16D PET与发射断层成像.md#130` |
| 14 | Geman & Geman | 1984 | 1 | `ch04\附录4A 马尔可夫链收敛性与Gibbs分布.md#126` |
| 15 | Natterer | 1986 | 2 | `ch16\附录16A Radon变换与Fourier切片定理的严格证明.md#65` |
| 16 | Parisi | 1988 | 1 | `ch08\附录8B 平均场近似的闭式推导.md#111` |
| 17 | Cybenko | 1989 | 1 | `ch09\附录9C 过参数化与双重下降.md#125` |
| 18 | Greenleaf & Uhlmann | 1989 | 2 | `ch16\附录16C 波前集理论与有限角CT的数学框架.md#33` |
| 19 | Hutchinson | 1990 | 4 | `ch06\6.4 切片得分匹配（SSM）与Hutchinson迹估计.md#23` |
| 20 | Hornik | 1991 | 1 | `ch09\附录9C 过参数化与双重下降.md#125` |
| 21 | Geman & Reynolds | 1992 | 2 | `ch04\4.5 Gibbs采样：利用条件结构的采样.md#39` |
| 22 | Geyer | 1992 | 2 | `ch04\4.2 Metropolis-Hastings算法.md#393` |
| 23 | Rudin-Osher-Fatemi | 1992 | 2 | `ch02\2.2 经典先验族.md#513` |
| 24 | Williams | 1992 | 1 | `ch09\9.2 重参数化技巧.md#323` |
| 25 | Quinto | 1993 | 2 | `ch16\附录16C 波前集理论与有限角CT的数学框架.md#23` |
| 26 | Hudson & Larkin | 1994 | 1 | `ch16\附录16D PET与发射断层成像.md#130` |
| 27 | Risken | 1996 | 3 | `ch05\5.1 从MCMC到Langevin SDE.md#212` |
| 28 | Roberts et al | 1997 | 2 | `ch04\4.2 Metropolis-Hastings算法.md#82` |
| 29 | Roberts, Gelman & Gilks | 1997 | 2 | `ch04\4.2 Metropolis-Hastings算法.md#314` |
| 30 | Roberts-Gelman-Gilks | 1997 | 2 | `ch04\4.2 Metropolis-Hastings算法.md#378` |
| 31 | Otto | 2001 | 3 | `ch14\附录14C Wasserstein空间的黎曼结构.md#33` |
| 32 | Vogel | 2002 | 1 | `ch03\3.6 收敛性分析与正则化参数选择.md#160` |
| 33 | Wang et al | 2004 | 1 | `ch01\附录1A 图像质量度量.md#86` |
| 34 | Buades et al | 2005 | 1 | `ch05\5.5 PnP框架：用去噪器替换先验梯度.md#153` |
| 35 | Hyvärinen | 2005 | 10 | `ch05\5.2 得分函数：对数概率的梯度.md#18` |
| 36 | Minka | 2005 | 2 | `ch08\8.4 变分推断与正则化的统一视角.md#331` |
| 37 | Roth & Black | 2005 | 1 | `ch04\附录4A 马尔可夫链收敛性与Gibbs分布.md#134` |
| 38 | Cover & Thomas | 2006 | 1 | `ch08\附录8C KL散度的性质与变分推断的信息论视角.md#87` |
| 39 | Dabov et al | 2007 | 1 | `ch05\5.5 PnP框架：用去噪器替换先验梯度.md#152` |
| 40 | Rahimi & Recht | 2007 | 1 | `ch15\附录15B 正弦位置编码的数学性质.md#60` |
| 41 | Ramani et al | 2007 | 2 | `ch17\17.3 SURE：Stein无偏风险估计与R2R.md#79` |
| 42 | Ambrosio et al | 2008 | 2 | `ch14\附录14C Wasserstein空间的黎曼结构.md#3` |
| 43 | Ambrosio, Gigli & Savaré | 2008 | 1 | `ch14\附录14C Wasserstein空间的黎曼结构.md#62` |
| 44 | Lustig et al | 2008 | 1 | `ch16\16.3 MRI成像：从自旋到k-space.md#204` |
| 45 | Villani | 2008 | 6 | `ch14\14.1 最优传输基础.md#203` |
| 46 | Villani | 2009 | 1 | `ch05\附录5A Fokker-Planck方程与Langevin平稳分布证明.md#162` |
| 47 | Borwein & Lewis | 2010 | 1 | `ch08\附录8A Fenchel共轭与ELBO的优化理论根基.md#183` |
| 48 | Bredies, Kunisch & Pock | 2010 | 1 | `ch02\附录2B TGV：TV的改进.md#21` |
| 49 | Glorot & Bengio | 2010 | 2 | `ch15\15.5 训练最佳实践.md#187` |
| 50 | Efron | 2011 | 4 | `ch05\5.3 Tweedie等式：从去噪器到得分函数.md#208` |
| 51 | Kloeden & Platen | 2011 | 1 | `ch07\7.5 数值离散化：从连续方程到可执行算法.md#146` |
| 52 | Vincent | 2011 | 13 | `ch06\6.0 本章导读.md#49` |
| 53 | Cotter et al | 2013 | 2 | `ch04\4.2 Metropolis-Hastings算法.md#326` |
| 54 | Cotter, Roberts, Stuart & White | 2013 | 1 | `ch04\附录4A 马尔可夫链收敛性与Gibbs分布.md#120` |
| 55 | Cuturi | 2013 | 3 | `ch14\14.1 最优传输基础.md#140` |
| 56 | Frikel & Quinto | 2013 | 2 | `ch16\附录16C 波前集理论与有限角CT的数学框架.md#33` |
| 57 | Venkatakrishnan et al | 2013 | 1 | `ch05\5.5 PnP框架：用去噪器替换先验梯度.md#266` |
| 58 | Venkatakrishnan, Bouman & Wohlberg | 2013 | 1 | `ch05\5.5 PnP框架：用去噪器替换先验梯度.md#245` |
| 59 | Bakry, Gentil & Ledoux | 2014 | 1 | `ch05\附录5A Fokker-Planck方程与Langevin平稳分布证明.md#162` |
| 60 | Chen, Ranftl & Pock | 2014 | 2 | `ch02\2.4 从显式先验到隐式先验.md#107` |
| 61 | Hörmann & Leydold | 2014 | 1 | `ch04\附录4A 马尔可夫链收敛性与Gibbs分布.md#92` |
| 62 | Kingma & Welling | 2014 | 10 | `ch08\8.2 ELBO推导：证据下界.md#277` |
| 63 | Rezende et al | 2014 | 4 | `ch08\8.5 从变分推断到生成模型：路线图.md#113` |
| 64 | He et al | 2015 | 2 | `ch15\15.5 训练最佳实践.md#188` |
| 65 | Kingma & Ba | 2015 | 2 | `ch15\15.5 训练最佳实践.md#19` |
| 66 | Pereyra | 2015 | 2 | `ch04\4.4 MYULA：近端ULA.md#42` |
| 67 | Rezende & Mohamed | 2015 | 2 | `ch09\9.2 重参数化技巧.md#311` |
| 68 | Ronneberger | 2015 | 1 | `ch06\6.6 去噪器作为得分估计器：实践与架构.md#32` |
| 69 | Ronneberger et al | 2015 | 4 | `ch14\14.5 前沿组合：SD3-Flux = Rectified Flow + DiT.md#198` |
| 70 | Santambrogio | 2015 | 3 | `ch14\14.1 最优传输基础.md#203` |
| 71 | Bowman et al | 2016 | 1 | `ch09\9.3 ELBO训练与KL正则化.md#321` |
| 72 | Cohen & Welling | 2016 | 1 | `ch17\17.6 测量一致性与等变架构.md#63` |
| 73 | Hoffman & Johnson | 2016 | 2 | `ch09\9.3 ELBO训练与KL正则化.md#317` |
| 74 | Salimans et al | 2016 | 1 | `ch01\附录1A 图像质量度量.md#188` |
| 75 | Sreehari et al | 2016 | 1 | `ch05\5.5 PnP框架：用去噪器替换先验梯度.md#246` |
| 76 | Sønderby et al | 2016 | 1 | `ch10\10.1 从VAE到层级VAE.md#255` |
| 77 | Alemi et al | 2017 | 1 | `ch08\附录8C KL散度的性质与变分推断的信息论视角.md#87` |
| 78 | Blei et al | 2017 | 4 | `ch08\8.1 为什么需要变分推断？.md#238` |
| 79 | Bora et al | 2017 | 1 | `ch02\2.4 从显式先验到隐式先验.md#248` |
| 80 | Dinh et al | 2017 | 1 | `ch14\14.2 连续归一化流(CNF).md#234` |
| 81 | Fox & Parker | 2017 | 2 | `ch04\4.6 加速采样方法：过松弛与惯性Langevin算法.md#42` |
| 82 | Heusel et al | 2017 | 1 | `ch01\附录1A 图像质量度量.md#228` |
| 83 | Higgins et al | 2017 | 5 | `ch08\附录8C KL散度的性质与变分推断的信息论视角.md#77` |
| 84 | Jang et al | 2017 | 2 | `ch09\9.2 重参数化技巧.md#299` |
| 85 | Vaswani et al | 2017 | 5 | `ch15\15.2 时间步嵌入：让网络感知噪声水平.md#36` |
| 86 | Zhang et al | 2017 | 4 | `ch05\5.5 PnP框架：用去噪器替换先验梯度.md#158` |
| 87 | Adler & Öktem | 2018 | 2 | `ch16\16.4 学习型重建方法.md#79` |
| 88 | Barratt & Sharma | 2018 | 1 | `ch01\附录1A 图像质量度量.md#213` |
| 89 | Blau & Michaeli | 2018 | 2 | `ch13\13.3 近似方法分类与DPS深度剖析.md#736` |
| 90 | Chen et al | 2018 | 4 | `ch14\14.2 连续归一化流(CNF).md#52` |
| 91 | Durmus et al | 2018 | 4 | `ch04\4.4 MYULA：近端ULA.md#182` |
| 92 | Durmus, Moulines & Pereyra | 2018 | 1 | `ch04\附录4A 马尔可夫链收敛性与Gibbs分布.md#122` |
| 93 | Gozcu et al | 2018 | 1 | `ch16\16.4 学习型重建方法.md#107` |
| 94 | Jamil | 2018 | 4 | `ch08\8.1 为什么需要变分推断？.md#238` |
| 95 | Kim & Mnih | 2018 | 1 | `ch09\9.3 ELBO训练与KL正则化.md#273` |
| 96 | Kingma & Dhariwal | 2018 | 1 | `ch14\14.2 连续归一化流(CNF).md#234` |
| 97 | Kumar et al | 2018 | 1 | `ch09\9.3 ELBO训练与KL正则化.md#274` |
| 98 | Lehtinen et al | 2018 | 2 | `ch17\17.2 Noise2Noise：用噪声替代干净标签.md#33` |
| 99 | Perez et al | 2018 | 1 | `ch15\15.2 时间步嵌入：让网络感知噪声水平.md#320` |
| 100 | Ardizzone et al | 2019 | 1 | `ch14\14.2 连续归一化流(CNF).md#242` |
| 101 | Batson & Royer | 2019 | 2 | `ch17\17.4 盲点网络与UNSURE：Noise2Self-Noise2Void及未知噪声.md#62` |
| 102 | Belkin et al | 2019 | 4 | `ch09\附录9C 过参数化与双重下降.md#72` |
| 103 | Belthangady & Royer | 2019 | 2 | `ch17\17.1 为什么自监督——数据困境与学习设定的谱系.md#30` |
| 104 | Bubba et al | 2019 | 3 | `ch16\16.2 CT的不适定性与正则化重建.md#85` |
| 105 | Devlin et al | 2019 | 1 | `ch14\14.6 离散扩散与掩码语言模型：从向量场到速率矩阵.md#204` |
| 106 | Durmus & Moulines | 2019 | 8 | `ch04\4.3 ULA：Langevin采样的Euler离散.md#115` |
| 107 | Grathwohl et al | 2019 | 2 | `ch14\14.2 连续归一化流(CNF).md#85` |
| 108 | Karras et al | 2019 | 1 | `ch15\15.2 时间步嵌入：让网络感知噪声水平.md#320` |
| 109 | Krull et al | 2019 | 2 | `ch17\17.4 盲点网络与UNSURE：Noise2Self-Noise2Void及未知噪声.md#33` |
| 110 | Laine et al | 2019 | 2 | `ch17\17.4 盲点网络与UNSURE：Noise2Self-Noise2Void及未知噪声.md#46` |
| 111 | Loshchilov & Hutter | 2019 | 2 | `ch15\15.5 训练最佳实践.md#23` |
| 112 | Narnhofer et al | 2019 | 1 | `ch02\2.4 从显式先验到隐式先验.md#248` |
| 113 | Pereyra | 2019 | 1 | `ch03\3.1 MAP估计：从后验众数到优化问题.md#69` |
| 114 | Peyré & Cuturi | 2019 | 1 | `ch14\14.1 最优传输基础.md#203` |
| 115 | Ryu et al | 2019 | 3 | `ch05\5.5 PnP框架：用去噪器替换先验梯度.md#248` |
| 116 | Song & Ermon | 2019 | 13 | `ch05\5.2 得分函数：对数概率的梯度.md#214` |
| 117 | Song et al | 2019 | 2 | `ch06\6.4 切片得分匹配（SSM）与Hutchinson迹估计.md#123` |
| 118 | Weiler & Cesa | 2019 | 1 | `ch17\17.6 测量一致性与等变架构.md#77` |
| 119 | Zhang et al | 2019 | 1 | `ch08\8.3 变分推断作为优化问题.md#343` |
| 120 | Bahadir et al | 2020 | 2 | `ch16\16.4 学习型重建方法.md#107` |
| 121 | Ho et al | 2020 | 40 | `ch06\6.5 多尺度得分匹配：从单一噪声到噪声条件网络.md#111` |
| 122 | Kobler, Effland, Kunisch & Pock | 2020 | 1 | `ch02\2.4 从显式先验到隐式先验.md#204` |
| 123 | Ongie et al | 2020 | 1 | `ch09\9.4 VAE作为逆问题的学习先验.md#212` |
| 124 | Pereyra, Mieles & Zygalakis | 2020 | 1 | `ch04\附录4A 马尔可夫链收敛性与Gibbs分布.md#131` |
| 125 | Soltanayev et al | 2020 | 2 | `ch17\17.3 SURE：Stein无偏风险估计与R2R.md#91` |
| 126 | Song & Ermon | 2020 | 1 | `ch06\6.5 多尺度得分匹配：从单一噪声到噪声条件网络.md#136` |
| 127 | Song et al | 2020 | 4 | `ch07\7.4 概率流ODE：随机采样的确定性等价.md#3` |
| 128 | Tancik et al | 2020 | 1 | `ch15\附录15B 正弦位置编码的数学性质.md#60` |
| 129 | Austin et al | 2021 | 2 | `ch14\14.6 离散扩散与掩码语言模型：从向量场到速率矩阵.md#345` |
| 130 | Bellec et al | 2021 | 1 | `ch17\17.7 本章小结：从数据困境到自监督统一.md#104` |
| 131 | Chen, Tachella & Davies | 2021 | 1 | `ch17\17.5 等变成像：从不完整测量中学习.md#210` |
| 132 | Dhariwal & Nichol | 2021 | 6 | `ch13\13.3 近似方法分类与DPS深度剖析.md#944` |
| 133 | Dosovitskiy et al | 2021 | 3 | `ch15\15.3 从ViT到DiT：Transformer接管扩散模型.md#59` |
| 134 | Huang et al | 2021 | 2 | `ch17\17.2 Noise2Noise：用噪声替代干净标签.md#99` |
| 135 | Jalal et al | 2021 | 2 | `ch13\13.3 近似方法分类与DPS深度剖析.md#41` |
| 136 | Kadkhodaie & Soltanolkotabi | 2021 | 1 | `ch13\13.3 近似方法分类与DPS深度剖析.md#944` |
| 137 | Kim & Ye | 2021 | 2 | `ch17\17.3 SURE：Stein无偏风险估计与R2R.md#117` |
| 138 | Kingma | 2021 | 7 | `ch11\附录11B 连续时间VLB与VDM.md#3` |
| 139 | Kingma et al | 2021 | 5 | `ch08\8.5 从变分推断到生成模型：路线图.md#113` |
| 140 | Laumont et al | 2021 | 5 | `ch03\3.7 从MAP到后验：点估计的局限与分叉.md#182` |
| 141 | Nakkiran et al | 2021 | 3 | `ch09\附录9C 过参数化与双重下降.md#72` |
| 142 | Nichol & Dhariwal | 2021 | 12 | `ch07\7.6 实践：用扩散SDE实现图像生成.md#37` |
| 143 | Pang et al | 2021 | 2 | `ch17\17.3 SURE：Stein无偏风险估计与R2R.md#121` |
| 144 | Song et al | 2021 | 19 | `ch06\6.5 多尺度得分匹配：从单一噪声到噪声条件网络.md#111` |
| 145 | Tran et al | 2021 | 1 | `ch08\8.5 从变分推断到生成模型：路线图.md#113` |
| 146 | Winkler | 2021 | 1 | `ch07\附录7A Anderson逆向时间SDE定理的证明概要.md#79` |
| 147 | Zhang | 2021 | 1 | `ch06\6.6 去噪器作为得分估计器：实践与架构.md#34` |
| 148 | Zhang et al | 2021 | 3 | `ch05\5.5 PnP框架：用去噪器替换先验梯度.md#159` |
| 149 | Campbell et al | 2022 | 3 | `ch14\14.6 离散扩散与掩码语言模型：从向量场到速率矩阵.md#14` |
| 150 | Chung et al | 2022 | 3 | `ch13\13.3 近似方法分类与DPS深度剖析.md#81` |
| 151 | Ho & Salimans | 2022 | 2 | `ch13\13.4 引导采样：从分类器引导到逆问题引导.md#138` |
| 152 | Holden et al | 2022 | 1 | `ch09\9.4 VAE作为逆问题的学习先验.md#212` |
| 153 | Holden, Pereyra & Zygalakis | 2022 | 1 | `ch02\2.4 从显式先验到隐式先验.md#293` |
| 154 | Karras et al | 2022 | 14 | `ch07\7.2 正向SDE：从数据到噪声的连续过程.md#89` |
| 155 | Kawar et al | 2022 | 7 | `ch07\附录7B VE-SDE与VP-SDE的推理等价性.md#61` |
| 156 | Laumont et al | 2022 | 3 | `ch13\13.3 近似方法分类与DPS深度剖析.md#601` |
| 157 | Lu et al | 2022 | 9 | `ch07\7.5 数值离散化：从连续方程到可执行算法.md#109` |
| 158 | Lugmayr et al | 2022 | 1 | `ch18\18.6 拓展方向与前沿挑战.md#58` |
| 159 | Rombach et al | 2022 | 5 | `ch13\13.3 近似方法分类与DPS深度剖析.md#506` |
| 160 | Tachella et al | 2022 | 1 | `ch17\17.5 等变成像：从不完整测量中学习.md#39` |
| 161 | Terris et al | 2022 | 1 | `ch05\5.5 PnP框架：用去噪器替换先验梯度.md#249` |
| 162 | Zamir et al | 2022 | 2 | `ch15\15.3 从ViT到DiT：Transformer接管扩散模型.md#92` |
| 163 | Zhang & Chen | 2022 | 1 | `ch07\7.7 采样加速：从高阶求解器到一致性模型.md#155` |
| 164 | Albergo & Vanden-Eijnden | 2023 | 1 | `ch14\14.5 前沿组合：SD3-Flux = Rectified Flow + DiT.md#312` |
| 165 | Cardoso et al | 2023 | 3 | `ch13\13.3 近似方法分类与DPS深度剖析.md#630` |
| 166 | Chung et al | 2023 | 10 | `ch13\13.2 后验得分分解：条件化的理论基础.md#71` |
| 167 | Feng et al | 2023 | 3 | `ch13\13.3 近似方法分类与DPS深度剖析.md#452` |
| 168 | Kim et al | 2023 | 2 | `ch07\7.7 采样加速：从高阶求解器到一致性模型.md#92` |
| 169 | Kingma & Gao | 2023 | 1 | `ch11\附录11B 连续时间VLB与VDM.md#44` |
| 170 | Lipman et al | 2023 | 8 | `ch14\14.3 Flow Matching.md#68` |
| 171 | Liu et al | 2023 | 5 | `ch14\14.4 Rectified Flow.md#3` |
| 172 | Mardani et al | 2023 | 3 | `ch13\13.3 近似方法分类与DPS深度剖析.md#434` |
| 173 | Peebles & Xie | 2023 | 10 | `ch14\14.5 前沿组合：SD3-Flux = Rectified Flow + DiT.md#199` |
| 174 | Pooladian et al | 2023 | 1 | `ch14\14.3 Flow Matching.md#719` |
| 175 | Song & Dhariwal | 2023 | 2 | `ch07\7.7 采样加速：从高阶求解器到一致性模型.md#92` |
| 176 | Song et al | 2023 | 8 | `ch07\7.7 采样加速：从高阶求解器到一致性模型.md#72` |
| 177 | Tachella et al | 2023 | 2 | `ch17\17.5 等变成像：从不完整测量中学习.md#182` |
| 178 | Zhang et al | 2023 | 3 | `ch13\13.4 引导采样：从分类器引导到逆问题引导.md#408` |
| 179 | Zhao et al | 2023 | 1 | `ch07\7.7 采样加速：从高阶求解器到一致性模型.md#155` |
| 180 | Zhu et al | 2023 | 5 | `ch13\13.3 近似方法分类与DPS深度剖析.md#516` |
| 181 | Black Forest Labs | 2024 | 3 | `ch14\14.5 前沿组合：SD3-Flux = Rectified Flow + DiT.md#262` |
| 182 | Chen & Lipman | 2024 | 1 | `ch14\14.5 前沿组合：SD3-Flux = Rectified Flow + DiT.md#312` |
| 183 | Daras et al | 2024 | 6 | `ch13\13.3 近似方法分类与DPS深度剖析.md#28` |
| 184 | Esser et al | 2024 | 5 | `ch14\14.5 前沿组合：SD3-Flux = Rectified Flow + DiT.md#3` |
| 185 | Gat et al | 2024 | 2 | `ch14\14.6 离散扩散与掩码语言模型：从向量场到速率矩阵.md#346` |
| 186 | He et al | 2024 | 1 | `ch13\13.3 近似方法分类与DPS深度剖析.md#747` |
| 187 | Holderrieth et al | 2024 | 1 | `ch14\14.6 离散扩散与掩码语言模型：从向量场到速率矩阵.md#348` |
| 188 | Li & Pereira | 2024 | 8 | `ch13\13.5 扩散最优控制（进阶）.md#73` |
| 189 | Liu et al | 2024 | 2 | `ch14\14.4 Rectified Flow.md#313` |
| 190 | Lou et al | 2024 | 1 | `ch14\14.6 离散扩散与掩码语言模型：从向量场到速率矩阵.md#347` |
| 191 | Lu & Song | 2024 | 2 | `ch07\7.7 采样加速：从高阶求解器到一致性模型.md#92` |
| 192 | Nakkiran et al | 2024 | 3 | `ch14\14.3 Flow Matching.md#453` |
| 193 | Patel et al | 2024 | 1 | `ch13\13.3 近似方法分类与DPS深度剖析.md#750` |
| 194 | Sahoo et al | 2024 | 2 | `ch14\14.6 离散扩散与掩码语言模型：从向量场到速率矩阵.md#347` |
| 195 | Tong et al | 2024 | 2 | `ch14\14.3 Flow Matching.md#719` |
| 196 | Wu et al | 2024 | 4 | `ch13\13.3 近似方法分类与DPS深度剖析.md#630` |
| 197 | Yang et al | 2024 | 1 | `ch13\13.3 近似方法分类与DPS深度剖析.md#748` |
| 198 | Chung et al | 2025 | 2 | `ch12\12.7 全书叙事的汇合.md#158` |
| 199 | Falk, Habring & Pock | 2025 | 3 | `ch04\4.6 加速采样方法：过松弛与惯性Langevin算法.md#50` |
| 200 | Gutha et al | 2025 | 1 | `ch13\13.3 近似方法分类与DPS深度剖析.md#539` |
| 201 | Habring et al | 2025 | 2 | `ch05\5.7 本章小结：去噪器即隐式先验即得分估计器.md#56` |
| 202 | Habring, Falk, Zach & Pock | 2025 | 2 | `ch05\5.4 MAP与MMSE的结构对偶性.md#212` |
| 203 | Kim, Kim & Ye | 2025 | 1 | `ch13\13.3 近似方法分类与DPS深度剖析.md#750` |
| 204 | Kuric et al | 2025 | 1 | `ch04\4.5 Gibbs采样：利用条件结构的采样.md#83` |
| 205 | Kuric, Zach, Habring, Unser & Pock | 2025 | 1 | `ch04\附录4A 马尔可夫链收敛性与Gibbs分布.md#129` |
| 206 | Monroy, Bacca & Tachella | 2025 | 1 | `ch17\17.3 SURE：Stein无偏风险估计与R2R.md#145` |
| 207 | Tachella | 2025 | 1 | `ch16\16.4 学习型重建方法.md#40` |
| 208 | Tachella et al | 2025 | 1 | `ch17\17.4 盲点网络与UNSURE：Noise2Self-Noise2Void及未知噪声.md#301` |
| 209 | Terris et al | 2025 | 1 | `ch17\17.5 等变成像：从不完整测量中学习.md#284` |
| 210 | Xu et al | 2025 | 1 | `ch13\13.3 近似方法分类与DPS深度剖析.md#745` |
| 211 | Chung et al | 2209.14687 | 5 | `ch13\13.4 引导采样：从分类器引导到逆问题引导.md#402` |

## 人工复核注意项（生成 bib 前须逐条裁决）

### A. 同名/同形多条目——须合并或区分的

1. **同一文献的多种作者写法**（工具已按写法分列，bib 须合并）：
   - `Roberts et al` / `Roberts, Gelman & Gilks` / `Roberts-Gelman-Gilks`（1997，均 RWM 最优缩放一文；第三种为 4.2#378 的连字符简写）
   - `Cotter et al` / `Cotter, Roberts, Stuart & White`（2013，MCMC for functions）
   - `Durmus et al` / `Durmus, Moulines & Pereyra`（2018，近端 MCMC）；`Pereyra`（2015/2019）为另一篇（近端 MCMC 算法），勿混
   - `Kuric et al` / `Kuric, Zach, Habring, Unser & Pock`（2025，Gaussian latent machine）
   - `Falk, Habring & Pock`（2025，Langevin meets Gibbs）与 `Habring, Falk, Zach & Pock`（2025，绝对零度扩散）为**两篇不同文献**
   - `Ronneberger` / `Ronneberger et al`（2015，U-Net 同一篇）
   - `Holden et al`（2022，9.4#212）与 `Holden, Pereyra & Zygalakis`（2022，2.4#293）**可能不同篇**，须查
   - `Kingma`（2021，VDM 简写）与 `Kingma et al`（2021，8.5#113）**疑为同一篇**（VDM），须确认
   - `Ambrosio et al` / `Ambrosio, Gigli & Savaré`（2008，Gradient flows 一书）
   - `Otto`（2001）与 `Otto, F` 已由工具合并；`Villani`（2008/2009 两本书）已合并同年份条目
2. **同名不同年份——按上下文消歧**（bib 键须带年份或后缀）：
   - `Song et al`：2019（SSM）/ 2020（Score-SDE 前身）/ 2021（Score-SDE）/ 2023（一致性/采样）
   - `Zhang et al`：2017（DnCNN）/ 2019 / 2021（PnP 语境）/ 2023（ControlNet，题名见 13.4#590）
   - `Chung et al`：2022（原始 DPS）/ 2023（DPS for General Noisy Inverse Problems）/ 2025（综述，ch12#158 与 ch13#742 两处，**疑同一篇**）
   - `Tachella et al`：2022（MOI, NeurIPS）/ 2023（EI, JMLR）/ 2025（UNSURE, ICLR）；`Tachella`（2025，16.4#40 单条，题名待查）
3. **arXiv 编号式引用**：`Chung et al (2209.14687)` ×5——**疑与 `Chung et al (2023)`（DPS general）为同一篇的 arXiv 版**，须查证后合并；若是，bib 用已发表年份。

### B. 工具已知不覆盖、需人工补入 bib 的条目

1. **`Daras et al. Survey (2410.00083)`**（ch13/13.7#50）：作者在括号外、标题间隔，工具无法捕获；题名与年份须人工补（arXiv 2410.00083）。
2. 标题+年份式括号（如 ch11/11.0#11 "（Denoising Diffusion Probabilistic Models… (2020)）"）：对应条目已在别处捕获（Ho et al 2020），仅计数差异。
3. 嵌套括号外层引用（如 ch12/12.7#161 "（Li & Pereira (NeurIPS 2024)…）"）：条目已覆盖，仅计数差异。
4. 裸 arXiv 编号无作者（如 "2406.08929v2 Sec 5"、"2209.14687v1 Sec 3.1"）：为教程/预印本内文指针，非独立文献，不进 bib。

### C. 非正式来源（建议不进 bib，或作 @misc）

- `Tachella 讲座 2025`（ch13/13.6#286、ch17/17.5#262）——讲座
- `LectureNotes2020_v2`（ch04/4.8#57 等 4 处）——课程讲义
- `winter school lab`（ch03/3.6#160，与 Vogel 2002 并列的实验设定来源）
- `Helsinki Tomography Challenge 2022`（ch16/16.2、ch18/18.6）、`Helsinki Deblur Challenge 2021`（ch18/18.6）——挑战赛
- `Black Forest Labs`（2024，Flux）——机构作者，作 @misc/@techreport
- `（Stein 引理，1956/1972/1981）`（ch05/5.2#18）——历史多年期提及，非单一条目

### D. 计数口径说明

次数列为工具实例计数，仍有少量实例因"作者与年份间隔标题"等形态未计入（不影响条目集合）；生成 bib 时以条目集合为准，次数仅作热度参考。
