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
│   └── v4/                 两组机制对照（Qwen 无回归头、DeBERTa 软标签对比学习）
├── logs/                   云端实跑原始日志
│   ├── v1/ v2/             各自一轮完整实跑
│   ├── v3/round1..3/       三轮实跑
│   └── v4/                 两组机制对照的实跑日志
├── figures/                19 张实测图
└── literature/             文献与导读
    ├── papers/             39 篇相关论文 PDF
    ├── guide/              综述、入门指南、从零教学指南
    └── README.md           文献索引
```

---

## 5. 结果

四个版本的管线在 `notebooks/v1` – `notebooks/v4`，实跑日志在 `logs/v1` – `logs/v4`。下表为 joint 运行
（四个子集合并训练，覆盖 6 个评测设置）的数值，表格中的数字直接取自对应日志。

### Qwen

表 1　Qwen 无回归头（v4）

| 评测设置 | Compositionality correlation | Context variance correlation | RMSE |
|---|---|---|---|
| en-nn ModAvg | 0.4988 | 0.2912 | 0.8222 |
| en-nn HeadAvg | 0.4164 | 0.2728 | 0.7985 |
| de-nn ModAvg | 0.2963 | 0.1627 | 0.5725 |
| de-nn HeadAvg | 0.6940 | 0.4461 | 0.8991 |
| en-pv Avg | 0.5612 | 0.7486 | 0.9258 |
| de-pv Avg | 0.5223 | 0.2076 | 0.9461 |
| 合计 | 0.4982 | 0.3548 | 0.8274 |

表 2　Qwen 有回归头（v3）

| 评测设置 | Compositionality correlation | Context variance correlation | RMSE |
|---|---|---|---|
| en-nn ModAvg | 0.4480 | 0.2679 | 0.9246 |
| en-nn HeadAvg | 0.4627 | 0.2792 | 0.8242 |
| de-nn ModAvg | 0.3169 | 0.3070 | 0.5934 |
| de-nn HeadAvg | 0.6609 | 0.4448 | 1.0999 |
| en-pv Avg | 0.4767 | 0.5546 | 1.2440 |
| de-pv Avg | 0.4046 | 0.6472 | 1.2257 |
| 合计 | 0.4616 | 0.4168 | 0.9853 |

表 3　Qwen 差值（无回归头 − 有回归头）

| 评测设置 | Δ Compositionality correlation | Δ Context variance correlation | Δ RMSE |
|---|---|---|---|
| en-nn ModAvg | +0.0508 | +0.0233 | −0.1024 |
| en-nn HeadAvg | −0.0463 | −0.0064 | −0.0257 |
| de-nn ModAvg | −0.0206 | −0.1443 | −0.0209 |
| de-nn HeadAvg | +0.0331 | +0.0013 | −0.2008 |
| en-pv Avg | +0.0845 | +0.1940 | −0.3182 |
| de-pv Avg | +0.1177 | −0.4396 | −0.2796 |
| 合计 | +0.0366 | −0.0620 | −0.1579 |

### DeBERTa

表 4　DeBERTa 无软标签（v1）

| 评测设置 | Compositionality correlation | Context variance correlation | RMSE |
|---|---|---|---|
| en-nn ModAvg | 0.4594 | 0.4729 | 0.8829 |
| en-nn HeadAvg | 0.3803 | 0.2113 | 0.8630 |
| de-nn ModAvg | 0.0383 | 0.1259 | 0.6259 |
| de-nn HeadAvg | 0.5272 | 0.3583 | 1.1613 |
| en-pv Avg | 0.5860 | 0.4879 | 0.9017 |
| de-pv Avg | 0.2830 | 0.3846 | 1.0626 |
| 合计 | 0.3790 | 0.3402 | 0.9162 |

表 5　DeBERTa 有软标签（v4）

| 评测设置 | Compositionality correlation | Context variance correlation | RMSE |
|---|---|---|---|
| en-nn ModAvg | 0.2154 | 0.2297 | 1.0527 |
| en-nn HeadAvg | 0.3139 | 0.1042 | 0.8578 |
| de-nn ModAvg | −0.0117 | −0.0234 | 0.6836 |
| de-nn HeadAvg | 0.4637 | 0.3230 | 1.2215 |
| en-pv Avg | 0.6546 | 0.6908 | 0.8960 |
| de-pv Avg | 0.1594 | 0.3762 | 1.1616 |
| 合计 | 0.2992 | 0.2834 | 0.9789 |

表 6　DeBERTa 差值（有软标签 − 无软标签）

| 评测设置 | Δ Compositionality correlation | Δ Context variance correlation | Δ RMSE |
|---|---|---|---|
| en-nn ModAvg | −0.2440 | −0.2432 | +0.1698 |
| en-nn HeadAvg | −0.0664 | −0.1071 | −0.0052 |
| de-nn ModAvg | −0.0500 | −0.1493 | +0.0577 |
| de-nn HeadAvg | −0.0635 | −0.0353 | +0.0602 |
| en-pv Avg | +0.0686 | +0.2029 | −0.0057 |
| de-pv Avg | −0.1236 | −0.0084 | +0.0990 |
| 合计 | −0.0798 | −0.0568 | +0.0627 |

---

## 6. 文献

综述、39 篇相关论文与两篇导读见 [`literature/`](literature/README.md)。与本任务最相关的：

- Miletić & Schulte im Walde (EACL 2023) — 组合性预测的方法论骨架与样本量边际收益曲线
- Jenkins, Miletić & Schulte im Walde (EMNLP 2023) — 德语闭合复合词的预切分；word2vec 反而强于 BERT
- Miletić & Schulte im Walde (ACL 2025) — 直接前作，其自述局限（仅英语、仅复合词）是本任务的立项理由
- Jenkins et al. (2026, arXiv 2607.11667) — 与本任务同构的数据；「组合性随时间下降」不成立
- Godbersen et al. (LREC 2026) / Kurtyigit et al. (ACL Findings 2025) — 「必须给语境」被反复证实，但 LLM 的优势可能来自旧数据泄漏
