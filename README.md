# ComPartMENT · SemEval-2027 Task 5

预测一个多词表达（MWE）在**具体语境**中的组合性评分（0–5）。

| | |
|---|---|
| 任务 | ComPartMENT — Noun Compound and Particle Verb Compositionality Prediction in Context |
| 组织者 | Filip Miletić, Chris W. Jenkins, Sabine Schulte im Walde（斯图加特大学 IMS） |
| 官方页 | <https://sites.google.com/view/semeval-compartment> ｜ <https://www.codabench.org/competitions/17971/> |
| 数据 | 1,539 个目标 / 12,380 条例句；含官方训练集 1,221 目标 / 9,825 例句 |

官方任务原文见 [`TASK.md`](TASK.md)，指标口径见 [`METRICS.md`](METRICS.md)。

---

## 1. 任务

给定一个多词表达（复合词 `account book`、小品词动词 `crack down`）和一句真实语料例句，
输出该表达**在这句语境里**的组合性分数（0–5）。四类设置：`en-nn` / `de-nn` / `en-pv` / `de-pv`；
复合词的修饰语（Mod）与中心语（Head）分别打分，小品词动词只打一个整体分。

数据按目标划分 80/20，同一表达的全部例句落在同一个划分里；当代与历史语料混排，测试集不公开时间来源。

---

## 2. 指标

官方在**每个 language×subtask 设置**上算三个指标，再取跨设置平均。

| # | 指标 | 定义 | 意义 |
|---|---|---|---|
| 1 | **Compositionality correlation**（主指标） | 逐例句 `Spearman(预测分, 人工分)` | 只看顺序是否一致，不看差多少 |
| 2 | **Context variance correlation**（次要指标） | 先按目标算预测分的 std，再算人工分的 std，最后跨目标 `Spearman` | 同一目标所有例句给同一个分时组内 std ≡ 0，相关系数未定义 → **nan**，榜单按 0 计 |
| 3 | **RMSE**（补充） | 逐例句误差的均方根 | 唯一看绝对值的指标，惩罚整体校准偏差 |

六个评测设置：`en-nn·ModAvg`、`en-nn·HeadAvg`、`de-nn·ModAvg`、`de-nn·HeadAvg`、`en-pv·Avg`、`de-pv·Avg`。

---

## 3. 难点

1. 组合性依赖语境，但四个子集的语境贡献量差近 20 倍：跨语境极差中位数 `de-nn` 0.70 对 `en-pv` 2.30，德语复合词的组内信号基本被标注噪声淹没。
2. 样本量小，每目标只有 7–10 条例句，而 EACL 2023 的系统搜索显示例句数从 10 增到 100 时 ρ 只提升 0.067。
3. 标签是 10 位标注者的均值，均值噪声 = `Std²/10`，指标 1 的 ρ 上限只有 0.845–0.927。
4. 历史例句取自公开文本，LLM 系统有旧数据泄漏风险：LREC 2026 实测无泄漏时 LLM 与 Skip-gram 基本持平（0.44 对 0.46）。
5. 德语闭合复合词必须先切分，EMNLP 2023 实测不切分时 MLM 的部分匹配率只有 0.06。
6. 数据带行序泄漏：组内 `Spearman(ContextID, 分数)` 达 0.98–0.99，只按组内秩输出的零文本基线在 `en-pv` 上就有 ρ = 0.623。
7. 多词表达在句中的定位有缺口，级联定位后 `en-pv` 86.3%、`de-pv` 77.7%。
8. 指标 2 的金标准被目标级歧义度污染，nn 上 40–74% 的 gold 组内 std 不是语境效应。

---

## 4. 目录结构

```
.
├── README.md               本文件
├── TASK.md                 官方任务说明原文（指标口径的唯一权威来源）
├── METRICS.md              官方三项指标的逐步演算 + 本地实现的口径对照
├── PREPROCESSING.md        预处理规则集（R 编号规则 + 流水线 + 提交格式规范）
├── data/train/             官方训练集 TSV：{en,de}-{nn,pv}-train.tsv
├── notebooks/              训练管线（按版本分目录）
│   ├── v1/                 首个跑通的完整管线（DeBERTa + Qwen）
│   ├── v2/                 超参与选择指标调整版
│   ├── v3/                 指标→损失项显式映射版
│   ├── v4/                 两组机制对照（Qwen 无回归头、DeBERTa 软标签对比学习）
│   ├── v6/                 V6 收敛配方（v5 消融结论落地；v5 本身未入库）
│   └── v9/                 XLM-RoBERTa-large 多seed×轮数扫描（v7/v8/v10 未入库）
├── logs/                   云端实跑原始日志
│   ├── v1/ v2/             各自一轮完整实跑
│   ├── v3/round1..3/       三轮实跑
│   ├── v4/                 两组机制对照的实跑日志
│   ├── v6/                 V6 实跑日志（每基线 single_en-nn + joint 两次运行）
│   └── v9/                 V9 实跑日志（6 次联合训练：seed 42/43/44 × 4/5 轮）
├── figures/                19 张实测图
└── literature/             文献与导读
    ├── papers/             39 篇相关论文 PDF
    ├── guide/              综述、入门指南、从零教学指南
    └── README.md           文献索引
```

---

## 5. 结果

表 1　各模型对比（joint 合计）

| 模型 | 指标1 Compositionality correlation | 指标2 Context variance correlation | 指标3 RMSE |
|---|---|---|---|
| XLM-RoBERTa-large（多seed扫描·最佳单运行） | 0.6004 | 0.4071 | 0.7574 |
| DeBERTa-v2-xlarge | 0.2992 | 0.2834 | 0.9789 |
| Qwen2.5-1.5B-Instruct | 0.4982 | 0.3548 | 0.8274 |

表 2　joint 逐设置对比

| 评测设置 | 指标1（XLM-RoBERTa-large） | 指标2（XLM-RoBERTa-large） | 指标3（XLM-RoBERTa-large） | 指标1（DeBERTa-v2-xlarge） | 指标2（DeBERTa-v2-xlarge） | 指标3（DeBERTa-v2-xlarge） | 指标1（Qwen2.5-1.5B-Instruct） | 指标2（Qwen2.5-1.5B-Instruct） | 指标3（Qwen2.5-1.5B-Instruct） |
|---|---|---|---|---|---|---|---|---|---|
| en-nn ModAvg | 0.4871 | 0.2769 | 0.8453 | 0.2154 | 0.2297 | 1.0527 | 0.4988 | 0.2912 | 0.8222 |
| en-nn HeadAvg | 0.5705 | 0.2894 | 0.7040 | 0.3139 | 0.1042 | 0.8578 | 0.4164 | 0.2728 | 0.7985 |
| de-nn ModAvg | 0.4506 | 0.3311 | 0.5427 | −0.0117 | −0.0234 | 0.6836 | 0.2963 | 0.1627 | 0.5725 |
| de-nn HeadAvg | 0.7867 | 0.5539 | 0.7047 | 0.4637 | 0.3230 | 1.2215 | 0.6940 | 0.4461 | 0.8991 |
| en-pv Avg | 0.6831 | 0.6850 | 0.8402 | 0.6546 | 0.6908 | 0.8960 | 0.5612 | 0.7486 | 0.9258 |
| de-pv Avg | 0.6246 | 0.3063 | 0.9072 | 0.1594 | 0.3762 | 1.1616 | 0.5223 | 0.2076 | 0.9461 |
| 合计 | 0.6004 | 0.4071 | 0.7574 | 0.2992 | 0.2834 | 0.9789 | 0.4982 | 0.3548 | 0.8274 |

---

## 6. 文献

综述、39 篇相关论文与两篇导读见 [`literature/`](literature/README.md)。与本任务最相关的：

- Miletić & Schulte im Walde (EACL 2023) — 组合性预测的方法论骨架与样本量边际收益曲线
- Jenkins, Miletić & Schulte im Walde (EMNLP 2023) — 德语闭合复合词的预切分；word2vec 反而强于 BERT
- Miletić & Schulte im Walde (ACL 2025) — 直接前作，其自述局限（仅英语、仅复合词）是本任务的立项理由
- Jenkins et al. (2026, arXiv 2607.11667) — 与本任务同构的数据；「组合性随时间下降」不成立
- Godbersen et al. (LREC 2026) / Kurtyigit et al. (ACL Findings 2025) — 「必须给语境」被反复证实，但 LLM 的优势可能来自旧数据泄漏
