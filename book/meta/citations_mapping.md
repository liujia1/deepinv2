# 正文引用 ↔ bib 键 对照表（Phase 3 交付物）

> 用途：Phase 5 排版时，按本表把正文里每一处散文式 `Author (Year)` 替换为 `\cite{bibkey}`。
> 输入：`meta/citations_review.md` 反查 211 条（实例计数）；产出键见 `meta/references.bib`（去重合并后 **197 键**，经 `book/tools/validate_bib.py` 校验通过；历史：200→198→197→196→197——废弃 2 键（holden2022langevin 后平反、chung2025survey redirect）+ 新增 laumont2023convergence + 2026-10-21 tachella2025unknown 合并入 tachella2025unsure + 新增 wu2024dpps（DPPS 坐实））。
> 状态图例：`✓`=字段完整可信｜`⚠`=bib 中带 `note={⚠待补/待核}`，须作者坐实｜`合并`=多条正文写法并入同一键。
> 口径（作者已拍板）：优先已发表版；非正式来源剔除；`Black Forest Labs (2024) Flux` 保留为 `@misc`；citekey=作者+年份+关键词小写。

## 一、非正式来源（按口径 Q3 剔除，不进 bib / 不进对照主表）

| 正文出处 | 说明 |
|---|---|
| Tachella 讲座 2025（ch13/13.6#286、ch17/17.5#262） | 讲座，非可引用文献 |
| LectureNotes2020_v2（ch04/4.8#57 等 4 处） | 课程讲义 |
| winter school lab（ch03/3.6#160，与 Vogel 2002 并列） | 实验设定来源，非文献 |
| Helsinki Tomography Challenge 2022（ch16/16.2、ch18/18.6） | 挑战赛 |
| Helsinki Deblur Challenge 2021（ch18/18.6） | 挑战赛 |
| 裸 arXiv 内文指针（2406.08929v2 / 2209.14687v1 的 "Sec …"） | 教程指针，非独立文献（其"论文本体"已分别并入对应键或不单列） |
| Stein 引理（1956/1972/1981）多年期历史提及（ch05/5.2#18） | 1981 已单列 `stein1981estimate`；余为历史叙述 |

> 注：`#207 Tachella (2025)`（16.4#40 单作者）若确为上述讲座，则应移入本节剔除；当前保守列入主表并标 ⚠。

## 二、主对照表（211 条反查 → 197 bib 键，1:1；#198 为 2026-10-21 消除行，编号留空）

| # | 正文写法 | 年 | 首见位置 | bib 键 | 状态 |
|---|---|---|---|---|---|
| 1 | Bloch | 1946 | ch16/16B#71 | bloch1946nmr | ✓ |
| 2 | Frankel | 1950 | ch04/4.6#26 | frankel1950sor | ✓ |
| 3 | Robbins | 1956 | ch05/5.3#206 | robbins1956robayes | ✓ |
| 4 | Miyasawa | 1961 | ch05/5.3#207 | miyasawa1961empirical | ✓ |
| 5 | Polyak | 1964 | ch04/4.6#67 | polyak1964heavysphere | ✓ |
| 6 | Moreau | 1965 | ch05/5.4#245 | moreau1965proximity | ✓ |
| 7 | Hastings | 1970 | ch04/4A#128 | hastings1970mh | ✓ |
| 8 | Rockafellar | 1970 | ch08/8A#183 | rockafellar1970convex | ✓ |
| 9 | Hudson | 1978 | ch17/17A#97 | hudson1978sure | ✓ |
| 10 | Adler | 1981 | ch04/4.6#32 | adler1981optimization | ✓ |
| 11 | Stein | 1981 | ch13/13.6#284 | stein1981estimate | ✓ |
| 12 | Anderson | 1982 | ch07/7.1#78 | anderson1982reverse | ✓ |
| 13 | Shepp & Vardi | 1982 | ch16/16D#130 | shepp1982mlem | ✓ |
| 14 | Geman & Geman | 1984 | ch04/4A#126 | geman1984stochastic | ✓ |
| 15 | Natterer | 1986 | ch16/16A#65 | natterer1986ct | ✓ |
| 16 | Parisi | 1988 | ch08/8B#111 | parisi1988fieldtheory | ✓ |
| 17 | Cybenko | 1989 | ch09/9C#125 | cybenko1989approximation | ✓ |
| 18 | Greenleaf & Uhlmann | 1989 | ch16/16C#33 | greenleaf1989visibility | ✓ |
| 19 | Hutchinson | 1990 | ch06/6.4#23 | hutchinson1990stochastic | ✓ |
| 20 | Hornik | 1991 | ch09/9C#125 | hornik1991approximation | ✓ |
| 21 | Geman & Reynolds | 1992 | ch04/4.5#39 | geman1992convergence | ✓ |
| 22 | Geyer | 1992 | ch04/4.2#393 | geyer1992convergence | ✓ |
| 23 | Rudin-Osher-Fatemi | 1992 | ch02/2.2#513 | rudin1992nofurnish | ✓ |
| 24 | Williams | 1992 | ch09/9.2#323 | williams1992reinforce | ✓ |
| 25 | Quinto | 1993 | ch16/16C#23 | quinto1993limitedangle | ✓ |
| 26 | Hudson & Larkin | 1994 | ch16/16D#130 | hudson1994osem | ✓ |
| 27 | Risken | 1996 | ch05/5.1#212 | risken1996fokkerplanck | ✓ |
| 28 | Roberts et al | 1997 | ch04/4.2#82 | roberts1997optimal | 合并 |
| 29 | Roberts, Gelman & Gilks | 1997 | ch04/4.2#314 | roberts1997optimal | 合并 |
| 30 | Roberts-Gelman-Gilks | 1997 | ch04/4.2#378 | roberts1997optimal | 合并 |
| 31 | Otto | 2001 | ch14/14C#33 | otto2001geometry | ✓ |
| 32 | Vogel | 2002 | ch03/3.6#160 | vogel2002computational | ✓ |
| 33 | Wang et al | 2004 | ch01/1A#86 | wang2004ssim | ✓ |
| 34 | Buades et al | 2005 | ch05/5.5#153 | buades2005nlm | ✓ |
| 35 | Hyvärinen | 2005 | ch05/5.2#18 | hyvarinen2005scorematching | ✓ |
| 36 | Minka | 2005 | ch08/8.4#331 | minka2005divergence | ✓ |
| 37 | Roth & Black | 2005 | ch04/4A#134 | roth2005redundant | ✓ |
| 38 | Cover & Thomas | 2006 | ch08/8C#87 | cover2006information | ✓ |
| 39 | Dabov et al | 2007 | ch05/5.5#152 | dabov2007bm3d | ✓ |
| 40 | Rahimi & Recht | 2007 | ch15/15B#60 | rahimi2007random | ✓ |
| 41 | Ramani et al | 2007 | ch17/17.3#79 | ramani2007montecarlo | ✓ |
| 42 | Ambrosio et al | 2008 | ch14/14C#3 | ambrosio2008gradient | 合并 |
| 43 | Ambrosio, Gigli & Savaré | 2008 | ch14/14C#62 | ambrosio2008gradient | 合并 |
| 44 | Lustig et al | 2008 | ch16/16.3#204 | lustig2008sparse | ✓ |
| 45 | Villani | 2008 | ch14/14.1#203 | villani2008optimal | ✓ |
| 46 | Villani | 2009 | ch05/5A#162 | villani2009hypocoercivity | ✓ |
| 47 | Borwein & Lewis | 2010 | ch08/8A#183 | borwein2010convex | ✓ |
| 48 | Bredies, Kunisch & Pock | 2010 | ch02/2B#21 | bredies2010tgv | ✓ |
| 49 | Glorot & Bengio | 2010 | ch15/15.5#187 | glorot2010init | ✓ |
| 50 | Efron | 2011 | ch05/5.3#208 | efron2011tweedie | ✓ |
| 51 | Kloeden & Platen | 2011 | ch07/7.5#146 | kloeden2011sde | ✓ |
| 52 | Vincent | 2011 | ch06/6.0#49 | vincent2011stacked | ✓ |
| 53 | Cotter et al | 2013 | ch04/4.2#326 | cotter2013mcmc | 合并 |
| 54 | Cotter, Roberts, Stuart & White | 2013 | ch04/4A#120 | cotter2013mcmc | 合并 |
| 55 | Cuturi | 2013 | ch14/14.1#140 | cuturi2013sinkhorn | ✓ |
| 56 | Frikel & Quinto | 2013 | ch16/16C#33 | frikel2013deconv | ✓ |
| 57 | Venkatakrishnan et al | 2013 | ch05/5.5#266 | venkatakrishnan2013red | 合并 |
| 58 | Venkatakrishnan, Bouman & Wohlberg | 2013 | ch05/5.5#245 | venkatakrishnan2013red | 合并 |
| 59 | Bakry, Gentil & Ledoux | 2014 | ch05/5A#162 | bakry2014analysis | ✓ |
| 60 | Chen, Ranftl & Pock | 2014 | ch02/2.4#107 | chen2014foe | ✓ |
| 61 | Hörmann & Leydold | 2014 | ch04/4A#92 | hormann2014rejection | ✓ |
| 62 | Kingma & Welling | 2014 | ch08/8.2#277 | kingma2014vae | ✓ |
| 63 | Rezende et al | 2014 | ch08/8.5#113 | rezende2014neso | ✓ |
| 64 | He et al | 2015 | ch15/15.5#188 | he2015rectifier | ✓ |
| 65 | Kingma & Ba | 2015 | ch15/15.5#19 | kingma2015adam | ✓ |
| 66 | Pereyra | 2015 | ch04/4.4#42 | pereyra2015proximal | ✓ |
| 67 | Rezende & Mohamed | 2015 | ch09/9.2#311 | rezende2015flows | ✓ |
| 68 | Ronneberger | 2015 | ch06/6.6#32 | ronneberger2015unet | 合并 |
| 69 | Ronneberger et al | 2015 | ch14/14.5#198 | ronneberger2015unet | 合并 |
| 70 | Santambrogio | 2015 | ch14/14.1#203 | santambrogio2015ot | ✓ |
| 71 | Bowman et al | 2016 | ch09/9.3#321 | bowman2016sentences | ✓ |
| 72 | Cohen & Welling | 2016 | ch17/17.6#63 | cohen2016gcnn | ✓ |
| 73 | Hoffman & Johnson | 2016 | ch09/9.3#317 | hoffman2016elbo | ✓ |
| 74 | Salimans et al | 2016 | ch01/1A#188 | salimans2016gan | ✓ |
| 75 | Sreehari et al | 2016 | ch05/5.5#246 | sreehari2016csmri | ✓ |
| 76 | Sønderby et al | 2016 | ch10/10.1#255 | sonderby2016ladder | ✓ |
| 77 | Alemi et al | 2017 | ch08/8C#87 | alemi2017vib | ✓ |
| 78 | Blei et al | 2017 | ch08/8.1#238 | blei2017vi | ✓ |
| 79 | Bora et al | 2017 | ch02/2.4#248 | bora2017ambientgan | ✓ |
| 80 | Dinh et al | 2017 | ch14/14.2#234 | dinh2017realnvp | ✓ |
| 81 | Fox & Parker | 2017 | ch04/4.6#42 | fox2017accelerated | ✓ |
| 82 | Heusel et al | 2017 | ch01/1A#228 | heusel2017wgan | ✓ |
| 83 | Higgins et al | 2017 | ch08/8C#77 | higgins2017betavae | ✓ |
| 84 | Jang et al | 2017 | ch09/9.2#299 | jang2017gumbel | ✓ |
| 85 | Vaswani et al | 2017 | ch15/15.2#36 | vaswani2017attention | ✓ |
| 86 | Zhang et al | 2017 | ch05/5.5#158 | zhang2017dncnn | ✓ |
| 87 | Adler & Öktem | 2018 | ch16/16.4#79 | adler2018lpd | ✓ |
| 88 | Barratt & Sharma | 2018 | ch01/1A#213 | barratt2018inception | ✓ |
| 89 | Blau & Michaeli | 2018 | ch13/13.3#736 | blau2018perception | ✓ |
| 90 | Chen et al | 2018 | ch14/14.2#52 | chen2018node | ✓ |
| 91 | Durmus et al | 2018 | ch04/4.4#182 | durmus2018proximal | 合并 |
| 92 | Durmus, Moulines & Pereyra | 2018 | ch04/4A#122 | durmus2018proximal | 合并 |
| 93 | Gozcu et al | 2018 | ch16/16.4#107 | gozcu2018compressive | ✓ |
| 94 | Jamil | 2018 | ch08/8.1#238 | jamil2018vi | ✓ |
| 95 | Kim & Mnih | 2018 | ch09/9.3#273 | kim2018factorvae | ✓ |
| 96 | Kingma & Dhariwal | 2018 | ch14/14.2#234 | kingma2018glow | ✓ |
| 97 | Kumar et al | 2018 | ch09/9.3#274 | kumar2018disentangled | ✓ |
| 98 | Lehtinen et al | 2018 | ch17/17.2#33 | lehtinen2018n2n | ✓ |
| 99 | Perez et al | 2018 | ch15/15.2#320 | perez2018film | ✓ |
| 100 | Ardizzone et al | 2019 | ch14/14.2#242 | ardizzone2019inn | ✓ |
| 101 | Batson & Royer | 2019 | ch17/17.4#62 | batson2019n2s | ✓ |
| 102 | Belkin et al | 2019 | ch09/9C#72 | belkin2019double | ✓ |
| 103 | Belthangady & Royer | 2019 | ch17/17.1#30 | belthangady2019fluorescence | ✓ |
| 104 | Bubba et al | 2019 | ch16/16.2#85 | bubba2019learning | ✓ |
| 105 | Devlin et al | 2019 | ch14/14.6#204 | devlin2019bert | ✓ |
| 106 | Durmus & Moulines | 2019 | ch04/4.3#115 | durmus2019ula | ✓ |
| 107 | Grathwohl et al | 2019 | ch14/14.2#85 | grathwohl2019ffjord | ✓ |
| 108 | Karras et al | 2019 | ch15/15.2#320 | karras2019stylegan | ✓ |
| 109 | Krull et al | 2019 | ch17/17.4#33 | krull2019n2v | ✓ |
| 110 | Laine et al | 2019 | ch17/17.4#46 | laine2019highquality | ✓ |
| 111 | Loshchilov & Hutter | 2019 | ch15/15.5#23 | loshchilov2019adamw | ✓ |
| 112 | Narnhofer et al | 2019 | ch02/2.4#248 | narnhofer2019inversegan | ✓ |
| 113 | Pereyra | 2019 | ch03/3.1#69 | pereyra2019proximal | ✓ |
| 114 | Peyré & Cuturi | 2019 | ch14/14.1#203 | peyre2019computational | ✓ |
| 115 | Ryu et al | 2019 | ch05/5.5#248 | ryu2019pnp | ✓ |
| 116 | Song & Ermon | 2019 | ch05/5.2#214 | song2019ncsn | ✓ |
| 117 | Song et al | 2019 | ch06/6.4#123 | song2019ssm | ✓ |
| 118 | Weiler & Cesa | 2019 | ch17/17.6#77 | weiler2019e2cnn | ✓ |
| 119 | Zhang et al | 2019 | ch08/8.3#343 | zhang2019avi | ✓ |
| 120 | Bahadir et al | 2020 | ch16/16.4#107 | bahadir2020loupe | ✓ |
| 121 | Ho et al | 2020 | ch06/6.5#111 | ho2020ddpm | ✓ |
| 122 | Kobler, Effland, Kunisch & Pock | 2020 | ch02/2.4#204 | kobler2020vargauss | ✓ |
| 123 | Ongie et al | 2020 | ch09/9.4#212 | ongie2020inverse | ✓ |
| 124 | Pereyra, Mieles & Zygalakis | 2020 | ch04/4A#131 | pereyra2020metropolis | ✓ |
| 125 | Soltanayev et al | 2020 | ch17/17.3#91 | soltanayev2020r2r | ✓ |
| 126 | Song & Ermon | 2020 | ch06/6.5#136 | song2020improvedscore | ✓ |
| 127 | Song et al | 2020 | ch07/7.4#3 | song2020ddim | ✓ |
| 128 | Tancik et al | 2020 | ch15/15B#60 | tancik2020fourier | ✓ |
| 129 | Austin et al | 2021 | ch14/14.6#345 | austin2021d3pm | ✓ |
| 130 | Bellec et al | 2021 | ch17/17.7#104 | bellec2021sure | ✓ |
| 131 | Chen, Tachella & Davies | 2021 | ch17/17.5#210 | chen2021equivariant | ✓ |
| 132 | Dhariwal & Nichol | 2021 | ch13/13.3#944 | dhariwal2021guidance | ✓ |
| 133 | Dosovitskiy et al | 2021 | ch15/15.3#59 | dosovitskiy2021vit | ✓ |
| 134 | Huang et al | 2021 | ch17/17.2#99 | huang2021neighbor2neighbor | ✓ |
| 135 | Jalal et al | 2021 | ch13/13.3#41 | jalal2021scoreald | ✓ |
| 136 | Kadkhodaie & Soltanolkotabi | 2021 | ch13/13.3#944 | kadkhodaie2021cspm | ✓ |
| 137 | Kim & Ye | 2021 | ch17/17.3#117 | kim2021n2score | ✓ |
| 138 | Kingma | 2021 | ch11/11B#3 | kingma2021vdm | 合并 |
| 139 | Kingma et al | 2021 | ch08/8.5#113 | kingma2021vdm | 合并 |
| 140 | Laumont et al | 2021 | ch03/3.7#182 | laumont2022pnpdm | 合并 |
| 141 | Nakkiran et al | 2021 | ch09/9C#72 | nakkiran2021double | ✓ |
| 142 | Nichol & Dhariwal | 2021 | ch07/7.6#37 | nichol2021improved | ✓ |
| 143 | Pang et al | 2021 | ch17/17.3#121 | pang2021r2r | ✓ |
| 144 | Song et al | 2021 | ch06/6.5#111 | song2021scoresde | ✓ |
| 145 | Tran et al | 2021 | ch08/8.5#113 | tran2021tutorialvb | ✓ |
| 146 | Winkler | 2021 | ch07/7A#79 | winkler2021reverse | ✓ |
| 147 | Zhang | 2021 | ch06/6.6#34 | zhang2021pnp | 合并 |
| 148 | Zhang et al | 2021 | ch05/5.5#159 | zhang2021pnp | 合并 |
| 149 | Campbell et al | 2022 | ch14/14.6#14 | campbell2022discrete | ✓ |
| 150 | Chung et al | 2022 | ch13/13.3#81 | chung2023dps | 合并 |
| 151 | Ho & Salimans | 2022 | ch13/13.4#138 | ho2022cfg | ✓ |
| 152 | Holden et al | 2022 | ch09/9.4#212 | holden2022datadriven | ✓ |
| 153 | Holden, Pereyra & Zygalakis | 2022 | ch02/2.4#293 | holden2022datadriven | 合并 |
| 154 | Karras et al | 2022 | ch07/7.2#89 | karras2022edm | ✓ |
| 155 | Kawar et al | 2022 | ch07/7B#61 | kawar2022ddrm | ✓ |
| 156 | Laumont et al | 2022 | ch13/13.3#601 | laumont2023convergence | ✓ |
| 157 | Lu et al | 2022 | ch07/7.5#109 | lu2022dpmsolver | ✓ |
| 158 | Lugmayr et al | 2022 | ch18/18.6#58 | lugmayr2022repaint | ✓ |
| 159 | Rombach et al | 2022 | ch13/13.3#506 | rombach2022ldm | ✓ |
| 160 | Tachella et al | 2022 | ch17/17.5#39 | tachella2022moi | ✓ |
| 161 | Terris et al | 2022 | ch05/5.5#249 | terris2022radio | ✓ |
| 162 | Zamir et al | 2022 | ch15/15.3#92 | zamir2022restormer | ✓ |
| 163 | Zhang & Chen | 2022 | ch07/7.7#155 | zhang2022deis | ✓ |
| 164 | Albergo & Vanden-Eijnden | 2023 | ch14/14.5#312 | albergo2023interpolants | ✓ |
| 165 | Cardoso et al | 2023 | ch13/13.3#630 | cardoso2023mcgdiff | ✓ |
| 166 | Chung et al | 2023 | ch13/13.2#71 | chung2023dps | 合并 |
| 167 | Feng et al | 2023 | ch13/13.3#452 | feng2023scoreprior | ✓ |
| 168 | Kim et al | 2023 | ch07/7.7#92 | kim2023ctm | ✓ |
| 169 | Kingma & Gao | 2023 | ch11/11B#44 | kingma2023vdmobj | ✓ |
| 170 | Lipman et al | 2023 | ch14/14.3#68 | lipman2023fm | ✓ |
| 171 | Liu et al | 2023 | ch14/14.4#3 | liu2023rectified | ✓ |
| 172 | Mardani et al | 2023 | ch13/13.3#434 | mardani2023reddiff | ✓ |
| 173 | Peebles & Xie | 2023 | ch14/14.5#199 | peebles2023dit | ✓ |
| 174 | Pooladian et al | 2023 | ch14/14.3#719 | pooladian2023msfm | ✓ |
| 175 | Song & Dhariwal | 2023 | ch07/7.7#92 | songdhariwal2023ict | ✓ |
| 176 | Song et al | 2023 | ch07/7.7#72 | song2023consistency | ✓ |
| 177 | Tachella et al | 2023 | ch17/17.5#182 | tachella2023me | ✓ |
| 178 | Zhang et al | 2023 | ch13/13.4#408 | zhang2023controlnet | ✓ |
| 179 | Zhao et al | 2023 | ch07/7.7#155 | zhao2023unipc | ✓ |
| 180 | Zhu et al | 2023 | ch13/13.3#516 | zhu2023diffpir | ✓ |
| 181 | Black Forest Labs | 2024 | ch14/14.5#262 | bfl2024flux | ✓ |
| 182 | Chen & Lipman | 2024 | ch14/14.5#312 | chen2024flowgeom | ✓ |
| 183 | Daras et al | 2024 | ch13/13.3#28 | daras2024survey | ✓ |
| 184 | Esser et al | 2024 | ch14/14.5#3 | esser2024sd3 | ✓ |
| 185 | Gat et al | 2024 | ch14/14.6#346 | gat2024dfm | ✓ |
| 186 | He et al | 2024 | ch13/13.3#747 | he2024mpgd | ✓ |
| 187 | Holderrieth et al | 2024 | ch14/14.6#348 | holderrieth2024genmatch | ✓ |
| 188 | Li & Pereira | 2024 | ch13/13.5#73 | li2024doc | ✓ |
| 189 | Liu et al | 2024 | ch14/14.4#313 | liu2024instaflow | ✓ |
| 190 | Lou et al | 2024 | ch14/14.6#347 | lou2024sedd | ✓ |
| 191 | Lu & Song | 2024 | ch07/7.7#92 | lusong2024scm | ✓ |
| 192 | Nakkiran et al | 2024 | ch14/14.3#453 | nakkiran2024step | ✓ |
| 193 | Patel et al | 2024 | ch13/13.3#750 | patel2024flowchef | ✓ |
| 194 | Sahoo et al | 2024 | ch14/14.6#347 | sahoo2024mdlm | ✓ |
| 195 | Tong et al | 2024 | ch14/14.3#719 | tong2024cfm | ✓ |
| 196 | Wu et al | 2024 | ch13/13.3#746 | wu2024dpps | ✓ |
| 197 | Yang et al | 2024 | ch13/13.3#748 | yang2024dsg | ✓ |
| 199 | Falk, Habring & Pock | 2025 | ch04/4.6#50 | falk2025overrelax | ✓ |
| 200 | Gutha et al | 2025 | ch13/13.3#539 | gutha2025mapga | ✓ |
| 201 | Habring et al | 2025 | ch05/5.7#56 | habring2025discretization | ✓ |
| 202 | Habring, Falk, Zach & Pock | 2025 | ch05/5.4#212 | habring2025abszero | ✓ |
| 203 | Kim, Kim & Ye | 2025 | ch13/13.3#750 | kim2025flowdps | ✓ |
| 204 | Kuric et al | 2025 | ch04/4.5#83 | kuric2025glm | 合并 |
| 205 | Kuric, Zach, Habring, Unser & Pock | 2025 | ch04/4A#129 | kuric2025glm | 合并 |
| 206 | Monroy, Bacca & Tachella | 2025 | ch17/17.3#145 | monroy2025gr2r | ✓ |
| 207 | Tachella | 2025 | ch16/16.4#40 | tachella2025unsure | 合并 |
| 208 | Tachella et al | 2025 | ch17/17.4#301 | tachella2025unsure | ✓ |
| 209 | Terris et al | 2025 | ch17/17.5#284 | terris2025ram | ✓ |
| 210 | Xu et al | 2025 | ch13/13.3#745 | xu2025dmap | ✓ |
| 211 | Chung et al | 2209.14687 | ch13/13.4#402 | chung2023dps | 合并 |
| 212 | Trippe et al | 2022 | ch13/13.3#630 | trippe2022smcdiff | ✓（2026-10-21 正文修正后新增行；原 #196 的 SMCDiff 三处已并入本行） |

## 三、需作者优先坐实的 ⚠ 清单（按影响排序）

> ✅ **第一组 11 个 `*unknown`** 已于 2026-10-20 经作者确认 + OpenAlex 坐实并回填；**第二组 17 个会场/卷页键**已于同日经作者裁决 + 联网核查并回填（含拆分 laumont、废弃 chung2025survey、删除占位；其中 holden2022langevin 已于 2026-10-21 平反——与 holden2022datadriven 为同一篇，正文 #152/#153 两处已合并）。剩余 **0** 键——全部完成！

> ✅ **trippe2022smcdiff**（原 wu2024smcdiff）张冠李戴**已修正到正文**（2026-10-21）：13.3#630 及 #815/#944 两处来源行的 "Wu et al. (2024) SMCDiff" 已全部改为 "Trippe et al. (2022)"（写作源与成书树同步），citekey 同日更名为 trippe2022smcdiff。**同章 #746 的 "Wu et al. (2024)" 是另一篇真引用**（DPPS），已于同日坐实为 `wu2024dpps`（Hongjie Wu 等 9 人，ACM MM 2024, pp. 214–223, doi:10.1145/3664647.3681556，arXiv:2402.16907；多候选+数据一致性筛选，与正文描述一致）——当时暂列 ⚠ 待核系与 SMCDiff 混淆，现已解除，⚠ 待核项清零。

> 多轮联网/正文核实已将 ⚠ 由 80 降至 **0**（全部三组已坐实）；2026-10-20 坐实第一组 11 + 第二组 14 + 第三组 bubba + wu→Trippe 张冠李戴纠正，废弃 2 键、新增 laumont2023convergence；**2026-10-21 复核修订**：holden2022langevin 平反（与 holden2022datadriven 合并）、tachella2025unknown 合并入 tachella2025unsure、14 个 `*unknown` 键规范更名、新增 `book/tools/validate_bib.py` 结构校验（全通过）、**两处正文张冠李戴提前修正**（Wu→Trippe SMCDiff ×3；Chung 2025→Daras 2024 ×2）、**DPPS 坐实**（13.3#746 "Wu et al. (2024)" = Hongjie Wu 等 ACM MM 2024，新增 wu2024dpps，⚠ 待核清零），总键数 **197**。

> 生成时间：Phase 3（2026-10-19）。正文 `Author (Year)` 一字未改，Phase 5 依本表替换为 `\cite{}`。
