# ComPartMENT · SemEval-2027 Task 5

预测一个多词表达（MWE）在**具体语境**中的组合性评分（0–5）。

| | |
|---|---|
| 任务 | ComPartMENT — Noun Compound and Particle Verb Compositionality Prediction in Context |
| 组织者 | Filip Miletić, Chris W. Jenkins, Sabine Schulte im Walde（斯图加特大学 IMS） |
| 官方页 | <https://sites.google.com/view/semeval-compartment> ｜ <https://www.codabench.org/competitions/17971/> |
| 数据 | 1,539 个目标 / 12,380 条例句；含官方训练集 1,221 目标 / 9,825 例句 |

官方任务原文见 **[`TASK.md`](TASK.md)**，指标口径与逐步演算见 **[`METRICS.md`](METRICS.md)**。

---

## 1. 任务

给定一个多词表达（复合词 `account book`、小品词动词 `crack down`）和一句真实语料例句，
输出该表达**在这句语境里**的组合性分数。覆盖四类设置：`en-nn` / `de-nn` / `en-pv` / `de-pv`；
复合词的修饰语（Mod）与中心语（Head）分别打分，小品词动词只打一个整体分。

本仓库给出两条基线的训练管线（`microsoft/deberta-v2-xlarge` 与 `Qwen2.5-1.5B-Instruct`），
两者共用同一套数据处理、超参、指标与绘图，只切换 `MODEL_KEY` 一个变量。

---

## 2. 指标

官方在**每个 language×subtask 设置**上算三个指标，再取跨设置平均。

| # | 指标 | 定义 | 意义 |
|---|---|---|---|
| 1 | **组合性相关 ρ**（主指标） | 逐例句 `Spearman(预测分, 人工分)` | 只看顺序是否一致，不看差多少 |
| 2 | **语境方差相关 ρ**（次要指标） | 先按目标算预测分的 std，再算人工分的 std，最后跨目标 `Spearman` | 同一目标所有例句给同一个分时组内 std ≡ 0，相关系数未定义 → **nan**，榜单按 0 计 |
| 3 | **RMSE**（补充） | 逐例句误差的均方根 | 唯一看绝对值的指标，惩罚整体校准偏差 |

六个评测设置：`en-nn·ModAvg`、`en-nn·HeadAvg`、`de-nn·ModAvg`、`de-nn·HeadAvg`、`en-pv·Avg`、`de-pv·Avg`。

---

## 3. 这个任务难在哪

**组合性判断本身依赖语境，而语境的贡献量在四个子集上差近 20 倍。**
跨语境极差中位数：`de-nn` 0.70、`en-nn` 1.40、`de-pv` 1.75、`en-pv` 2.30。
`de-nn` 的组内真实信号方差为 0.0000（观测 0.1175 vs 噪声期望 0.1264），
即德语复合词上「同一表达在不同句子里」的差异基本被标注噪声淹没。

**样本量小，且正处在文献给出的边际收益最陡段。**
每个目标只有 7–10 条例句。Miletić & Schulte im Walde (EACL 2023) 对 41,496 组配置做过系统搜索，
给出的曲线是例句数 10 → 100 时 ρ 提升 +0.067，100 → 1000 只再 +0.017。
本任务的标注规模下，模型容量和超参的收益很容易被抽样噪声盖过。

**评分噪声给 ρ 划了硬天花板。**
标签是 10 位标注者 0–5 分的均值，均值噪声 = `Std²/10` ⇒ 指标 1 的 ρ 上限只有 **0.845–0.927**，
信噪比 2.51（en-pv，最脏）–6.10（de-nn Head，最干净）。人工组内 std 中只有约 10–15% 是可复现方差。
标注者之间的分歧是任务的固有属性：逐行 `Std > 1.5` 的比例在 `en-pv` 达 60.7%，
且评分越低分歧越大（Roller et al. (MWE 2013) 的标注清洗实验同样表明，高方差条目无法用阈值剔除）。

**历史语料的泄漏风险。**
本任务的历史例句取自十九世纪小说等公开文本，训练与评测之间存在旧数据重叠的可能。
Kurtyigit et al. (ACL Findings 2025) 与 Godbersen et al. (LREC 2026) 都报告了 LLM 系统在旧数据上分数畸高
（Reddy 旧数据 ρ = 0.736 vs ComposiGen 新数据 ρ = 0.44），无泄漏时 LLM 与 Skip-gram 基本持平（0.44 vs 0.46）。
因此必须做 unseen MWE 划分，且不能只靠随机划分。

**德语闭合复合词必须先切分。**
Jenkins, Miletić & Schulte im Walde (EMNLP 2023) 指出德语复合词不做预切分时 MLM 的部分匹配率只有 0.06，
切分后升到 0.26；同一篇里还给出更锋利的负面结果：word2vec 0.525 高于最好的 BERT 0.368。
本仓库实测 `Mod + Head` 直接拼接在 `de-nn` 只有 59.1% 能还原，其余 40.9% 带 Fugen-s（-s 25.2%、-n 8.0% 等）。
英德小品词动词的语序还相反（英 `Base + Particle`，德 `Particle + Base`）。

**「组合性随时间下降」这一假设不成立。**
Jenkins et al. (2026, arXiv 2607.11667) 在与本任务几乎同构的数据上给出德语 Δ = −0.0012、英语 Δ = −0.0161，
即两期数据的组合性没有显著差异。年代信号本身也很稀疏：表层启发式（显式年份、现代词表、古拼写正则、
德语 ß/ASCII 转写）在英语上只覆盖 6–7% 的例句、德语 36–38%，与评分的相关性全部落在 |ρ| ≤ 0.17 的噪声区间。
年代只能作分层评估变量，不能作特征。

**数据本身带一条现成的作弊通道。**
组内 `Spearman(ContextID, 分数)` 高达 0.98–0.99，且 100% 的目标 ρ > 0.5（`ModAvg` 列是随机序，
说明泄漏编码的是中心语/整体贡献）。后果：一个完全不看文本、只按组内秩输出的基线，
在 `en-pv` 上拿 ρ = 0.623、`de-pv` 0.521，已追平已发表最好系统（ComposiGen 0.50）。
逐行随机划分与 MWE 互斥划分都挡不住它 —— 载入后必须先删 `ContextID`、再物理打乱。

**多词表达在句中的定位有硬缺口。**
级联定位（精确 → 转写还原 → 德语前缀屈折/后置分离 → 词干前缀 → 相邻窗口）之后：
`en-nn` 100%、`de-nn` 100%、`en-pv` 86.3%、`de-pv` 77.7%。简单子串匹配会漏掉 66%–79% 的例句，
剩下的 `de-pv` 22.3% 需要形态分析器。

**指标 2 的金标准被目标级歧义度污染。**
nn 上 40–74% 的 gold 组内 std 来自「这个 MWE 本身就难说」，不是语境效应。
蒙特卡洛天花板：`en-nn·Mod` 0.052、`en-nn·Head` 0.081、`de-nn·Mod` 0.001、`de-nn·Head` 0.002、
`en-pv` 0.413、`de-pv` 0.257。同时，指标 1 与 RMSE 并不强制使用语境 ——
「每行填该目标的真实均值」的作弊上界是指标 1 ρ = 0.70–0.92、RMSE 0.33–0.76，但指标 2 全为 nan。

---

## 4. 目录结构

```
.
├── README.md               本文件
├── TASK.md                 官方任务说明原文（指标口径的唯一权威来源）
├── METRICS.md              官方三项指标的逐步演算 + 本地实现的口径对照
├── PREPROCESSING.md        预处理规则集（R 编号规则 + 流水线 + 提交格式规范）
├── AGENTS.md               仓库工作约定
├── data/train/             官方训练集 TSV：{en,de}-{nn,pv}-train.tsv
├── notebooks/              训练管线（两个基线各一份，按版本分目录）
│   ├── v1/                 首个跑通的完整管线
│   ├── v2/                 超参与选择指标调整版
│   └── v3/                 指标→损失项显式映射版（当前主线）
├── logs/                   云端实跑原始日志
│   ├── v1/ v2/             各自一轮完整实跑
│   └── v3/round1..3/       三轮实跑，round3 为 5×2 次全部成功的完整一轮
├── figures/                19 张实测图
└── literature/             文献与导读
    ├── papers/             39 篇相关论文 PDF
    ├── guide/              综述、入门指南、从零教学指南
    └── README.md           文献索引
```

---

## 5. 版本

| 版本 | 内容 |
|---|---|
| v1 | 首版基线：两个主干 × 5 次运行（`single_en-nn` / `single_de-nn` / `single_en-pv` / `single_de-pv` / `joint`）；六个评测设置的三项指标见 `figures/fig12_eval_metrics.png`，数据侧实测见 `figures/fig1`–`fig17` |
| v2 | 超参、best-checkpoint 准则与组内损失护栏的调整版（`notebooks/v2`、`logs/v2`） |
| v3 | 三个官方指标各自落到显式损失项的版本（`notebooks/v3`、`logs/v3`），当前提交主线 |

---

## 6. 文献

综述、39 篇相关论文与两篇导读见 [`literature/`](literature/README.md)。与本任务最相关的：

- Miletić & Schulte im Walde (EACL 2023) — 组合性预测的方法论骨架与样本量边际收益曲线
- Jenkins, Miletić & Schulte im Walde (EMNLP 2023) — 德语闭合复合词的预切分；word2vec 反而强于 BERT
- Miletić & Schulte im Walde (ACL 2025) — 直接前作，其自述局限（仅英语、仅复合词）正是本任务的立项理由
- Jenkins et al. (2026, arXiv 2607.11667) — 与本任务同构的数据；「组合性随时间下降」不成立
- Godbersen et al. (LREC 2026) / Kurtyigit et al. (ACL Findings 2025) — 「必须给语境」被反复证实，但 LLM 的优势可能来自旧数据泄漏
