# ComPartMENT 仓库工作约定

本文件是本仓库的长期工作说明。新增内容前先读它，不要另起一套规则。

## 仓库目标

SemEval-2027 Task 5（ComPartMENT）基线与数据分析的公开仓库：任务原文、三条指标口径、
两条主干的训练管线（v1–v3）、云端实跑日志、可复现的数据分析脚本与图、文献导读。

## 目录职责

| 目录 | 内容 | 规则 |
|---|---|---|
| 仓库根 | 只放入口文档 | 根目录不放中间产物；新增入口文档必须先在 `README.md` 的目录树里登记 |
| `TASK.md` | 官方任务说明原文（逐字） | 指标口径的唯一权威来源，改动前先读 |
| `METRICS.md` | 官方三项指标的逐步演算 + 本地实现的口径对照 | 文中每个数字必须能由 `scripts/metric_walkthrough.py` 复现 |
| `data/train/` | 官方训练集 TSV | 只读不加工；官方更新时整体替换，文件名保持 `{lang}-{task}-{split}.tsv` |
| `notebooks/v{n}/` | 该版本的完整训练管线，两个基线各一份 | 新版本**新建目录，不覆盖旧版本**；两个基线必须逐格一致，只允许标题 / `MODEL_KEY` / 结尾命令三处不同 |
| `logs/v{n}/` | 该版本的云端实跑原始 stdout | 是所有结论数字的唯一来源；新版本新建子目录，v3 用 `round{n}` 区分多轮实跑 |
| `reports/` | 结论性文档 | 实验报告 `v{n}_baseline.md`；变更说明 `v{n}_changelog.md`；崩溃分析 `v{n}_run_{原因}_crash.md` |
| `reports/figures/` | 实测图 | 命名 `fig{n}_{主题}.png`；图上文字一律英文 |
| `scripts/` | 分析与复现脚本，及其产出的 JSON | 一个脚本只回答一个问题；JSON 与脚本同放；一次性诊断不入库 |
| `literature/papers/` | 论文 PDF | 命名 `venue-id.pdf`（与 ACL Anthology ID 一致） |
| `literature/guide/` | 综述 / 入门指南 / 教学指南 | 改动后同步更新 `literature/README.md` 的索引 |
| `literature/README.md` | 文献索引 | 新增论文即登记 |

## 写作与提交纪律

1. **数字只从日志或脚本取**，不引用文档表格的转述；写进结论的每个数字都要能被一个脚本复现。
2. **判定收益必须带噪声阈值**：目标级 cluster bootstrap 给出的最小可检测差为
   指标 1 = 0.065、指标 2 = 0.153、RMSE = 0.046（`scripts/bootstrap_metric_noise.py`）。
   差值落在阈值内一律写「噪声内」，不断言谁更好。
3. **对照实验除被比较变量外一切设置相同**；为跑通某一条支线而调整共享设置（优化器、精度、批次、checkpointing、随机种子）时，必须同步调整其余支线。
4. **代码里不写教学式注释**：解释放 notebook 的 markdown 格或 `reports/` 文档；`.py` 只保留可执行的语句与必要的函数签名说明。
5. **文档不写过程日志**：只写「做了什么 → 测到什么 → 结论」，不写「我发现了什么、我们讨论了什么」。
6. **不把否证结论删掉**：无效或有害的尝试同样要留在报告与 README 里，并标注「无可测收益 / 疑似有害」。
7. 提交前跑通 `python3 scripts/<脚本>` 验证路径与输出仍然正确；notebook 改动后用 `nbformat.validate` 检查。

## 新增一个版本的标准流程

1. `notebooks/v{n}/` 下新建两个基线 notebook（以 `notebooks/v3/` 为模板，只改标题、`MODEL_KEY`、结尾命令）。
2. 云端实跑，日志存到 `logs/v{n}/`。
3. `python3 scripts/parse_v3_logs.py`、`scripts/three_gen_compare.py`、`scripts/v3_crosscheck.py` 取数，
   其中 `v3_crosscheck.py` 负责确认两条基线的口径逐字相同。
4. 写 `reports/v{n}_baseline.md`（结论）与 `reports/v{n}_changelog.md`（改了什么、为什么、效果）。
5. 更新 `README.md` 的版本表、创新性尝试表与遗留问题表。
6. 提交并推送。

## 当前状态

- 提交主线：`notebooks/v3/ComPartMENT基线_Qwen2.5-1.5B-Instruct.ipynb`，
  joint 指标 1 = 0.4616 / 指标 2 = 0.4168 / RMSE = 0.9853，六个评测设置的指标 2 全部非 nan。
- 已知阻断：官方测试集推理与 zip 打包未实现，是通往合法提交的唯一硬阻断。
- 未解问题：DeBERTa 在两个 nn 子集上相对 v1 崩塌 0.40，根因锁定为有效 batch 16 → 8，未修。
