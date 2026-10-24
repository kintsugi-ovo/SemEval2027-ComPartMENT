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
├── notebooks/              训练管线（两个基线各一份，按版本分目录）
│   ├── v1/                 首个跑通的完整管线
│   ├── v2/                 超参与选择指标调整版
│   └── v3/                 指标→损失项显式映射版（当前主线）
├── logs/                   云端实跑原始日志
│   ├── v1/ v2/             各自一轮完整实跑
│   └── v3/round1..3/       三轮实跑
├── figures/                19 张实测图
└── literature/             文献与导读
    ├── papers/             39 篇相关论文 PDF
    ├── guide/              综述、入门指南、从零教学指南
    └── README.md           文献索引
```

---

## 5. 版本与结果

`single_*` 为单子集训练，`joint` 为四个子集合并训练，其数值为六个评测设置的平均。

### v1

**DeBERTa-v2-xlarge**

| 运行 | Compositionality correlation | Context variance correlation | RMSE |
|---|---|---|---|
| single_en-nn | 0.4659 | 0.3446 | 0.8210 |
| single_de-nn | 0.4017 | 0.2792 | 0.7935 |
| single_en-pv | 0.5574 | 0.6538 | 0.9229 |
| single_de-pv | 0.0776 | 0.3775 | 1.0890 |
| joint | 0.3790 | 0.3402 | 0.9162 |

**Qwen2.5-1.5B-Instruct**

| 运行 | Compositionality correlation | Context variance correlation | RMSE |
|---|---|---|---|
| single_en-nn | 0.4692 | 0.1925 | 0.8423 |
| single_de-nn | 0.5035 | 0.2512 | 0.7707 |
| single_en-pv | 0.4989 | 0.6574 | 0.9525 |
| single_de-pv | 0.3795 | 0.3317 | 1.2004 |
| joint | 0.4950 | 0.3864 | 0.8429 |

### v2

**DeBERTa-v2-xlarge**

| 运行 | Compositionality correlation | Context variance correlation | RMSE |
|---|---|---|---|
| single_en-nn | 0.4608 | 0.3631 | 0.8369 |
| single_de-nn | 0.0605 | 0.1847 | 0.9514 |
| single_en-pv | 0.2509 | 0.1964 | 1.1711 |
| single_de-pv | 0.2689 | 0.2133 | 1.0635 |
| joint | 0.2002 | 0.0690 | 1.1237 |

**Qwen2.5-1.5B-Instruct**

| 运行 | Compositionality correlation | Context variance correlation | RMSE |
|---|---|---|---|
| single_en-nn | 0.4168 | 0.1369 | 0.9163 |
| single_de-nn | 0.4620 | 0.3871 | 0.7976 |
| single_en-pv | 0.4781 | 0.7335 | 1.2073 |
| single_de-pv | 0.4037 | 0.4990 | 1.0005 |
| joint | 0.4999 | 0.3648 | 0.8459 |

### v3

**DeBERTa-v2-xlarge**

| 运行 | Compositionality correlation | Context variance correlation | RMSE |
|---|---|---|---|
| single_en-nn | 0.0670 | 0.0662 | 0.9652 |
| single_de-nn | 0.0044 | 0.0661 | 0.9925 |
| single_en-pv | 0.6252 | 0.7179 | 0.8583 |
| single_de-pv | 0.1838 | 0.2721 | 1.1128 |
| joint | 0.3431 | 0.1956 | 0.9711 |

**Qwen2.5-1.5B-Instruct**

| 运行 | Compositionality correlation | Context variance correlation | RMSE |
|---|---|---|---|
| single_en-nn | 0.4492 | 0.2654 | 0.8560 |
| single_de-nn | 0.4521 | 0.3184 | 0.8367 |
| single_en-pv | 0.5182 | 0.7019 | 1.0380 |
| single_de-pv | 0.3457 | 0.2036 | 1.1048 |
| joint | 0.4616 | 0.4168 | 0.9853 |

---

## 6. 文献

综述、39 篇相关论文与两篇导读见 [`literature/`](literature/README.md)。与本任务最相关的：

- Miletić & Schulte im Walde (EACL 2023) — 组合性预测的方法论骨架与样本量边际收益曲线
- Jenkins, Miletić & Schulte im Walde (EMNLP 2023) — 德语闭合复合词的预切分；word2vec 反而强于 BERT
- Miletić & Schulte im Walde (ACL 2025) — 直接前作，其自述局限（仅英语、仅复合词）是本任务的立项理由
- Jenkins et al. (2026, arXiv 2607.11667) — 与本任务同构的数据；「组合性随时间下降」不成立
- Godbersen et al. (LREC 2026) / Kurtyigit et al. (ACL Findings 2025) — 「必须给语境」被反复证实，但 LLM 的优势可能来自旧数据泄漏
