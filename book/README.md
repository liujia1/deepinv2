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
python book/tools/scan_citations.py
```

## 成书阶段（当前进度）

- [x] Phase 0  书稿区分离与版本管理（2026-10-16）
- [x] Phase 1  结构审计（2026-10-16，见 `meta/全书缺口报告.md`）
- [x] Phase 2  技术一致性校对（符号表 → 逐章修正）
- [ ] Phase 3  参考文献体系（references.bib）——反查已完成（`meta/citations_review.md`，198 条），待逐条补全生成 bib
- [ ] Phase 4  成书组件补写（前言/符号总表/索引/后记）
- [ ] Phase 5  排版构建管线（待定：Quarto / LaTeX / 出版社流程）
- [ ] Phase 6  出版流程（定位：学术专著，Springer 式）
