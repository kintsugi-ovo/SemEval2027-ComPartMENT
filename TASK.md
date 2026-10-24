# ComPartMENT 官方任务说明（原文存档）

> 权威来源：CodaBench 竞赛页 Overview 全文（`https://www.codabench.org/competitions/17971/`），
> 存档日期 2026-10-22。
> 官方任务网站（`https://sites.google.com/view/semeval-compartment`）与 CfP 公告**均不含指标定义**，
> 指标定义只在本页 ⇒ **本文件是指标口径的唯一权威来源**。
>
> ⚠️ 本机网络到 `codabench.org` 不可达（连接超时），所以这份原文**只能靠本存档**。
> **任何改动指标口径的动作前，先读本文件。**

---

## 1. 官方原文（逐字）

### Updates

> September 5, 2026: You may already submit results on trial data (English only) for a formatting check. Note that the leaderboard is not currently operational; however, the scoring program will execute and you can download its output (scores.txt) with the computed evaluation metrics.

### Task summary

> We propose a shared task on compositionality prediction which relies on a newly constructed dataset of ordinal in-context compositionality ratings. Systems are expected to output numerical compositionality scores for a target expression in an example sentence.
>
> - Two MWE types: noun compounds (e.g., climate change) and particle verbs (e.g., go off)
> - Two languages: English and German
> - Two time periods: present-day and historical data
>
> Sample human compositionality ratings are shown below. Noun compounds are rated separately for the contribution of modifier (left) and head (right) constituents; particle verbs have a single, overall rating.
>
> | Noun compound example | Mod | Head |
> |---|---|---|
> | she propelled herself into the firing line in taking the stance she did | 0.6 | 0.7 |
> | our flea market will feature some of the most creative individuals | 0.0 | 4.7 |
>
> | Particle verb example | Overall |
> |---|---|
> | it's gonna get very cold so pull up your leggings | 4.8 |
> | I will probably wind up as a Web Haunter […] blogging for eternity | 0.2 |

### Subtasks

> Systems are expected to output numerical compositionality scores for a target expression in an example sentence. We propose two subtasks.
>
> **Subtask A (noun compounds)**: The task is to produce **two rankings** of sentences containing target noun compounds: one in terms of the compositionality of their modifier (first) constituents, and one in terms of the compositionality of their head (second) constituents.
>
> **Subtask B (particle verbs)**: The task is to produce a ranking of sentences by the compositionality of target particle verbs. We rely on a single (expression-level) compositionality rating for particle verbs.
>
> Both subtasks are available in two languages, English and German; and they include instances from two time periods, historical and present-day. Note that **the temporal origin of the test instances (present-day vs. historical data) will not be revealed** in order to encourage system designs which are robust to data points from different type periods. We will however incorporate this information in the final analysis of submitting systems.

### Evaluation metrics（★ 三个指标的官方定义）

> For each subtask/language combination, we apply the following evaluation metrics.
>
> **Compositionality correlation**: As the **main** ranking-based evaluation, we compute **Spearman's rank correlation coefficient (rho)** between the system-produced and gold standard compositionality ratings.
>
> **Context variance correlation**: From a **secondary perspective**, we will assess **sensitivity to contextual variability**: we will compute **standard deviations for system-produced compositionality ratings for a given target (across all its example sentences)**, and then calculate **Spearman's rho relative to the corresponding standard deviations computed over human ratings**.
>
> **RMSE**: Complementing the ranking-based evaluations, we will more directly assess error magnitude by computing the **root mean squared error** between the system-produced and gold standard compositionality ratings.

### Reporting of results across task settings

> We will report each of our evaluation metrics on two levels:
>
> - for **individual subtask/language combinations** (e.g., English -- noun compound modifiers, English -- noun compound head, English -- particle verb, etc.);
> - as the **mean of the scores obtained across subtask/language combinations**, to provide dedicated insights into submissions targeting all task settings.

### Dataset structure and release

> Our dataset includes:
>
> - 560 English noun compound targets;
> - 579 German noun compound targets;
> - 200 particle verb targets in each language.
>
> For each target, we aimed to annotate 10 example sentences, sample **5 each from present-day and historical data**. This number is lower in some cases due to data sparsity, particularly in historical data, with a mean of ~9 total examples per target.
>
> We created **80/20 train/test splits on the target level**, i.e., the splits do not overlap in targets and each split therefore contains all example sentences of its targets. To maintain a similar distribution of gold standard ratings across the splits, we **first binned the instances into compositionality ranges and then sampled from those**. The splits are created **independently for each language/subtask combination**.
>
> In addition, as an illustration of the data format, we are also releasing small-scale **trial data for English (2 targets x 5 examples for each subtask)**. These targets are **not included in the main dataset** used for the train/test splits.
>
> | Phase | Release date | EN NN | EN PV | DE NN | DE PV | Download |
> |---|---|---|---|---|---|---|
> | Trial | September 5, 2026 | 2/10 | 2/10 | N/A | N/A | [link] |
> | Training | September 8, 2026 | 444/3480 | 158/1557 | 461/3298 | 158/1490 | In Files tab |
> | Evaluation | January 10, 2027 | **116/908** | **42/414** | **118/833** | **42/400** | TBD |
>
> （表内数字 = 唯一目标数 / 例句总数）

### Data format

> The data is in tab-separated text files. The file names are in the format `[language]-[task]-[split].tsv`, e.g., `en-nn-train.tsv`. Language can be en or de; task can be nn or pv.
>
> The files contain the following column names in the header:
>
> - `ContextID`: unique instance identifier
> - For noun compounds:
>   - `Compound`: target noun compound
>   - `Mod`: modifier (left) constituent
>   - `Head`: head (right) constituent
>   - `ModAvg`: mean average of human judgments for modifier
>   - `ModStd`: standard deviation of human judgments for modifier
>   - `HeadAvg`: mean average of human judgments for head
>   - `HeadStd`: standard deviation of human judgments for head
> - For particle verbs:
>   - `ParticleVerb`: target particle verb
>   - `Base`: base verb
>   - `Particle`: particle
>   - `Avg`: mean average of human judgments
>   - `Std`: standard deviation of human judgments
> - `Context`: rated example sentence
>
> **The *Avg and *Std columns are omitted from the test split.**

### Submission format

> The submission is a **single zip archive** containing text files named following the format `[language]-[task]-pred.tsv`.
>
> The file **should not have a header**. Each line corresponds to one prediction, with the following tab-separated values:
>
> - `ContextID`
> - For noun compounds: modifier prediction, head prediction
> - For particle verbs: single prediction

---

## 2. 逐条对照：官方要求 vs 本项目实现

| 官方条款 | 本项目实现 | 位置 | 判定 |
|---|---|---|---|
| 指标 1 = Spearman(系统分, gold) | `stats.spearmanr(p, g)` | `tools/nbgen/cells_b.py::_three_metrics` | ✅ 一致 |
| 指标 2 = 按目标算**系统分**的 std → 与**人工分**对应 std 求 Spearman | `ps = groupby("g")["p"].std(ddof=1)`；`gs = groupby("g")["y"].std(ddof=1)`；`spearmanr(ps[ok], gs[ok])` | 同上 | ✅ **完全一致**（含"across all its example sentences"） |
| 指标 3 = RMSE | `sqrt(((p-g)**2).mean())` | 同上 | ✅ 一致 |
| "for **each** subtask/language combination" 分别算 | 6 个设置（en-nn·ModAvg / en-nn·HeadAvg / de-nn·ModAvg / de-nn·HeadAvg / en-pv·Avg / de-pv·Avg）各自算 | `compute_compartment_metrics` | ✅ 一致 |
| 跨设置报告 = **各设置的 mean** | `合计|_指标*_ρ = _nanmean(...)`（nan 安全） | 同上 | ✅ 一致 |
| NN 出 **modifier / head 两列** | `eval_cols` 两列；pv 第 2 列 `loss_mask=0` 占位 | `cells_a.py` / `cells_b.py` | ✅ 一致 |
| 80/20 **目标级**划分，子集不重叠 | 子集内部目标级互斥 | `cells_a.py` 切分逻辑 | ✅ 一致 |
| 先**分箱**再抽样以保持 gold 分布 | 先分箱后分层抽样 | 同上 | ✅ 一致 |
| 每个 **language/subtask 独立**划分 | 四子集独立划分 | 同上 | ✅ 一致 |
| 提交 = 单个 zip，4 个 `[lang]-[task]-pred.tsv`，**无表头**；NN 3 列 / PV 2 列 | `export_pred()` 无表头、NN 3 列、PV 2 列 | `cells_c.py` | ✅ 一致（zip 打包待做，交付物 10） |
| 测试集**不含** `*Avg`/`*Std` 列 | 载入逻辑容忍缺列 | `cells_a.py` | ✅ 一致 |

### 唯一未被官方规定的细节：std 的 ddof

官方只写 "standard deviations"，**未指定有偏 (ddof=0) 还是无偏 (ddof=1)**。
本项目用 `ddof=1`（pandas `groupby.std()` 默认）。

**实测风险量化**（`analysis/` 可复现，2026-10-22）：

- 两种 ddof 下的**目标级 std 的秩相关 ≥ 0.9968**（en-nn 最低 0.9968，en-pv 最高 0.9997）→ 目标间的先后顺序几乎不变；
- 在"带噪 oracle"模拟下（预测 = 真值 + N(0, σ)），指标 2 的两种口径之差为 **0.0001 ~ 0.034**（中位 ≈ 0.006），
  **远小于指标 2 的可检测阈值 0.153** ⇒ **对结论与排名无实质影响**。

⇒ **该细节不必再纠结**；若仍要绝对保险，可在提交前用 Trial 校验一次（但见 §3 的限制）。

---

## 3. ⚠️ 两条必须更正的项目判断

### 3.1 「指标 2 是硬门槛 / 唯一及格线」是**的保守策略**，不是官方规定

官方原文的定位是：

| 指标 | 官方措辞 | 地位 |
|---|---|---|
| Compositionality correlation | "As the **main** ranking-based evaluation" | **主要指标** |
| Context variance correlation | "From a **secondary perspective**, we will assess sensitivity to contextual variability" | **次要指标** |
| RMSE | "Complementing the ranking-based evaluations…" | 补充 |

- 官方**没有**任何"指标 2 = nan 即出局"的条文；
- 官方把**指标 1 定位为主指标**，却把指标 2 当"唯一及格线"。

⇒ **工程上继续坚持"指标 2 非 nan"仍然是对的**（nan 在榜单上通常按 0/None 计，会直接拖垮跨设置平均，
且它证明系统完全没在用语境）—— 但这应表述为**内部红线 / 防御性策略**，而不是"官方硬门槛"。

### 3.2 Trial 渠道**校验不了指标 2**

官方原文：trial data = "**2 targets x 5 examples for each subtask**"（仅英语）。

⇒ 每个子集只有 **2 个目标** ⇒ 指标 2 要在目标之间算 Spearman ⇒ **2 个点只能得到 ±1**（秩的排列只有两种
⇒ 任何非恒定输入都拿 ±1，**零判别力**；数学上 n=2 可定义，只有 n=1 才真正是 nan）
⇒ **Trial 只能校验：① 提交格式 ② 指标 1 ③ RMSE**。
⇒ 「用 Trial 校验本地指标实现」这一条**必须收窄**：**指标 2 的口径无法用 Trial 校验**，
只能靠本文件（官方原文）+ 本地实现的逐条对照来保证（已全部 ✅）。
（证据见 `scripts/nan_anatomy.py` §2：`n=1 → nan`；`n=2 → ±1`。）

---

## 4. 与官方数据规模的一致性（交叉验证通过）

| 来源 | EN NN | EN PV | DE NN | DE PV |
|---|---|---|---|---|
| 官方 Training 表（目标/例句） | 444/3480 | 158/1557 | 461/3298 | 158/1490 | ✅ 与本项目实测**逐个数字一致** |
| 官方 Evaluation 表（目标/例句） | 116/908 | 42/414 | 118/833 | 42/400 | ✅ 合计 **318 目标 / 2555 例句** = 本项目实测值 |
| 官方总目标数（未划分前） | 560 | 158+42=200 | 579 | 200 | ✅ 与"1,539 目标"一致 |
| 官方"每目标 ~10 例句，5 现在 + 5 历史" | 训练集均 7.15–9.85/目标 | | | ✅ 因 80% 划分所致 |
