# book/ —— 成书工作区

本目录是"从草稿到成书"的书稿区。**写作源仍在仓库根下的各章目录**，本目录通过工具脚本同步、审计、构建。

## 目录结构

```
book/
  manuscript/    # 成书树（同步产物，可删除重建）：ch00=绪论, ch01..ch18=各章
  tools/         # 工具脚本
  meta/          # 审计报告、符号表、参考文献等成书元数据
  build/         # 构建产物（PDF 等，已 gitignore，暂未启用）
```

## 工具用法

```bash
# 同步成书树（写作源 → book/manuscript/，跳过数据/checkpoint/审计备份）
python book/tools/sync_manuscript.py          # 全部同步
python book/tools/sync_manuscript.py --dry    # 预览
python book/tools/sync_manuscript.py --ch 12  # 只同步某章

# 三项内容审计（详见 meta/ 下报告）
python book/tools/audit_crossrefs.py          # 交叉引用：章/节/附录/实验 引用是否存在
python book/tools/audit_notation.py           # 符号一致性：各章记法混用统计
python book/tools/audit_numbers.py            # 数值对账：正文数字 ↔ results_summary.json

# 参考文献反查（Phase 3）：正文 Author (Year) 引用扫描 → meta/citations_raw.csv + citations_review.md
# 覆盖形态：散文式/括号内式、"等人"式、中文连接符多作者、连字符姓氏、APA 全名式/斜体文献式、
#           arXiv 编号式、括号内前缀/后缀/会场名、同括号多引用拆分（2026-10-18 升级 v3）
python book/tools/scan_citations.py

# 参考文献结构校验（Phase 3）：bib 解析/重复键/必填字段/活跃⚠/映射表与 CSV 覆盖
python book/tools/validate_bib.py
```

## 成书阶段（当前进度）

- [x] Phase 0  书稿区分离与版本管理（2026-10-16）
- [x] Phase 1  结构审计（2026-10-16，见 `meta/全书缺口报告.md`）
- [x] Phase 2  技术一致性校对（符号表 → 逐章修正）
- [x] Phase 3  参考文献体系（references.bib）——反查 211 条（`meta/citations_review.md`，独立交叉复核+工具 v3）→ 去重合并生成 `meta/references.bib`（**197 键**）+ `meta/citations_mapping.md`（正文↔bib键对照表）；结构自检通过（`book/tools/validate_bib.py`）；经多轮联网/正文逐条查证（坐实 DiffPIR/Hörmann/Monte-Carlo SURE/Quinto/DOC 等，纠正 Ramani/Hörmann/Frikel/Tachella 等作者失实）；wu→Trippe、chung2025→daras 两处正文张冠李戴**已于 2026-10-21 在排版前修正**（写作源+成书树同步），DPPS（Wu et al. 2024, ACM MM）同日坐实，**遗留 0 键 ⚠、0 项待核**——全部完成（2026-10-20 三组清零；2026-10-21 复核修订：holden 平反合并、tachella 合并、14 键规范更名、正文两处误引修正、DPPS 坐实）
- [ ] Phase 4  成书组件补写（前言/索引/后记）——**符号总表、术语索引、实验环境附录已交稿**（2026-10-21，均在 `绪论/`，经 `sync_intro()` 进 ch00）：`符号总表.md`（218 行八节，含一符多义消歧表与两套记号体系对照）、`术语索引.md`（460 行：按章序主索引 163 条 + 英文速查 163 行 + 易混辨析 8 对，291 处节号双检通过；并按索引词频统一正文用词，两轮共 162 处：前向过程→正向过程 30、前向 SDE→正向 SDE 2、收敛率→收敛速率 6、前向算子→正向算子 72、前向模型→正向模型 12、前向加噪→正向加噪 10 等，附录10B 与 18.2 两个文件名同步更名（保留前向传播/前向 KL/前向自动微分等深度学习固定译法））、`实验环境.md`（116 行七节）、`前言.md`（24 行，据作者提供的缘起/主线/读者/读法/差异五段素材成稿；待补致谢、署名、配套资源表述）；均待作者审阅。余项：后记
- [ ] Phase 5  排版构建管线（待定：Quarto / LaTeX / 出版社流程）
- [ ] Phase 6  出版流程（定位：学术专著，Springer 式）
