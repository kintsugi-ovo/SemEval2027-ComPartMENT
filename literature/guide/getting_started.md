# ComPartMENT 入门指南（重写版）

> 面向已有 PyTorch / Transformer 基础、但没做过这个任务的人。
> 目标：**读完后能独立跑通一条完整且合法的提交管线**，而不是读完一篇综述。
>
> 本指南以**数据结构**为骨架组织——因为这个任务的 90% 的坑都不在模型里，而在「这张表到底是什么结构」上。
> 学术脉络与论文详解见 [文献综述](literature_review.md)；
> 从零理解「组合性预测」这个概念本身，见 [`教学指南`](tutorial.md)(教学指南_组合性预测从零到一.md)。

---

## 目录

- [第 0 步：这个任务到底让你做什么](#第-0-步这个任务到底让你做什么)
- [第 1 步：把数据结构彻底看穿](#第-1-步把数据结构彻底看穿)
- [第 2 步：认清三条硬约束](#第-2-步认清三条硬约束)
- [第 3 步：跑通一条最小管线](#第-3-步跑通一条最小管线)
- [第 4 步：三项指标怎么算](#第-4-步三项指标怎么算)
- [第 5 步：四条必须先做的预处理](#第-5-步四条必须先做的预处理)
- [第 6 步：建模的四条路线与推荐顺序](#第-6-步建模的四条路线与推荐顺序)
- [第 7 步：导出与提交](#第-7-步导出与提交)
- [第 8 步：评估与自检](#第-8-步评估与自检)
- [常见问题速查](#常见问题速查)

---

## 第 0 步：这个任务到底让你做什么

一句话：**给定一个多词表达和一句包含它的真实句子，输出这个表达在这句话里有多「组合」的分数（0–5）。**

「组合」在这里是技术术语，含义是：**这个整体的意思，能不能从两个成分的意思拼出来。**

看官方给的例子就懂了：

| 表达 | 句子 | 分数 | 解读 |
|---|---|---|---|
| `flea market` | our **flea market** will feature some of the most creative individuals | **Mod 0.0 / Head 4.7** | `flea`（跳蚤）一点贡献都没有，`market`（市场）贡献全部 |
| `pull up` | it's gonna get very cold so **pull up** your leggings | **4.8** | 字面义：把裤腿拉上去 → 完全组合 |
| `wind up` | I will probably **wind up** as a Web Haunter […] blogging for eternity | **0.2** | 完全不组合：「卷起来」是胡编 |

四件事必须一次记住：

1. **两类目标**：名词复合词（`climate change`）和小品词动词（`go off`）。
2. **两种语言**：英语 + 德语。
3. **两个时期**：当代 + 历史。官方明确说**测试集不会告诉你哪句是历史**。
4. **两个等级**：
   - 名词复合词要输出**两个**分数——修饰语（`Mod`）一个、中心语（`Head`）一个。
   - 小品词动词只有**一个**整体分数。

---

## 第 1 步：把数据结构彻底看穿

**这一节是全篇最重要的部分。** 数据结构决定了一切建模选择。

### 1.1 你手上是四张表，不是一张

```
train/en-nn-train.tsv    3,480 行   Compound=Mod + " " + Head      → 9 列
train/de-nn-train.tsv    3,298 行   德语复合词（有 Fugen-s）        → 9 列
train/en-pv-train.tsv    1,557 行   ParticleVerb = Base + " " + Particle → 7 列
train/de-pv-train.tsv    1,490 行   ParticleVerb = Particle + Base（无空格）→ 7 列
```

**四个子集的词库完全不重叠**（实测：两两目标串交集均为 0）。也就是说 NN 和 PV 是两门独立的课，模型可以完全分开训练。

### 1.2 关键：这是**两层嵌套**，不是一张扁平表

```
┌── 外层：目标 (MWE) ─────────────────────────────────────────────────────┐
│ 目标 "account book"        （NN 数据里 444 个）                         │
│                                                                         │
│   ┌── 内层：该目标下的例句 (Context)   每目标 7-10 条 ────────────┐     │
│   │  例句 3470                                                    │     │
│   │  例句 3471                                                    │     │
│   │  例句 3472                                                    │     │
│   └───────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────┘
```

**一行 = 一个「某表达在某句中的」实例。一个目标 = 7–10 行。**

这个嵌套结构是理解全部三个指标的前提：

| 指标 | 在哪一层计算 |
|---|---|
| 指标 1（ρ）、指标 3（RMSE） | **例句层**（把所有行混在一起算） |
| 指标 2（语境方差相关） | **先在目标层聚合 std，再跨目标算相关** |

### 1.3 真数据长什么样（`account book` 这个目标的全部 10 行）

```
ContextID   Mod   Head   ModAvg  HeadAvg   Context
10001       3.50  3.80   ...     ...       Correspondence, letterbooks, account books, and writings…
10002       3.00  3.90   ...     ...       Let me see, (looking at my account book) that is two weeks wages making $8.
10003       4.40  4.10   ...     ...       She drops her black silk bag; out of it drop crackers, an account book, …
10004       4.30  4.10   ...     ...       Whether you are running or planning a business, this seminar…
10005       4.10  4.30   ...     ...       When he had finished, he took out his account book, and entered…
10006       3.90  4.40   ...     ...       Mr. Force went into the little den at the back of the hall…
10007       4.10  4.60   ...     ...       The records consist of two of the company's financial ledgers…
10008       4.30  4.70   ...     ...       Not the least among them, as she regarded it, was an account book.
10009       4.50  4.80   ...     ...       The following details derive from Meacci's account book…
10010       4.10  4.90   ...     ...       For example, typing 'book' would result in any record where the field…
```

**注意三件事：**

1. **同一个目标，10 条例句的分数完全不同**（HeadAvg 从 3.80 一路到 4.90）。这就是「in-context」的含义——分数是**逐例句**的，不是每个词一个数。
2. **`ContextID` 在组内是单调递增的，而且和分数几乎完全同序**（上面 `HeadAvg` 就是 3.8→4.9 一路涨）。**这是个陷阱，见第 5 步。**
3. **`Context` 里的目标词不一定字面等于目标串**。`account book` 在 10003 行里是 `an account book`，10010 行里目标短语甚至被拆开引号引起来。

再看一个对照——小品词动词 `crack down`：

```
ContextID   Avg    Context
30001       0.30   The manifesto also committed the Tories to backing new 'drugalyser' technology…   ← 意为「严厉打击」，不组合
30005       0.60   Both reforms are part of the government's wide-ranging plans to crack down on…
30011       0.80   This is shown by Ezra's completely disingenuous "Do n't crack down on…
30025       1.20   Concerned with the perceived radicalization of this rights movement…
30054       1.55   This canon was as black as Erebus down there — a sea of sombre firs;…              ← 完全习语化
30061       1.60   Such an end would make it easier for the Pakistani government to crack down on…
```

德语 `abhauen` 更极端，跨度 **0.20 → 4.50**：

```
70001  0.20  Wir waren zusammen, aber dann habe ich die High School abgebrochen und…   ← 意为「跑路」，完全不组合
70166  2.30  Nichts half ihm mehr die Bitte, sein Haupt hieb Jener ab, …              ← 意为「砍下」，半组合
70311  3.40  Deine Stiefschwester hat längst den Tod verdient, heute Nacht wenn sie…  ← 字面义「打掉」，组合
```

**德语还有一层额外难点**：粒子会跑到句尾。`abhauen` 在原文里写作 `hieb Jener ab`（`ab` 跑到最后）。所以德语 PV 的目标定位需要专门处理**可分动词分离**。

### 1.4 成分拼接的构造律（实测，可直接编码）

| 子集 | 规则 | 命中率 |
|---|---|---|
| `en-nn` | `Compound == Mod + " " + Head` | **100%** |
| `de-nn` | `Compound == Mod + lower(Head[0]) + Head[1:]` | **59.1%** |
| `en-pv` | `ParticleVerb == Base + " " + Particle` | **100%** |
| `de-pv` | `ParticleVerb == Particle + Base`（**无空格**，粒子在前） | **100%** |

三条可直接使用的结论：

- **英语 PV 是「基干 + 粒子」，德语 PV 是「粒子 + 基干」，顺序相反。**
- **德语复合词有连接成分（Fugen-s）**，超过 40% 的目标**无法**用字符串拼接还原。连接成分分布：无 57.7% / `-s` 25.2% / `-n` 8.0% / `-en` 1.5% / `-e` 1.1% / `-er` 0.7% / `-es` 0.7% / `-ens` 0.2%。
- **绝不要用 `Compound[:len(Mod)]` / `Compound[-len(Head):]` 这样的字符串切片来切分德语复合词**——会切错。

### 1.5 训练集 vs 测试集的列差异

| 表 | NN 列数 | PV 列数 |
|---|---|---|
| 训练集 | **9** | **7** |
| 测试集 | **7** | **5** |

测试集去掉了所有 `*Avg` / `*Std` 列。**推理时你只有 `ContextID` + `Context` + 成分列。**

> ⚠️ 由此得到一条硬性工程约束：**读入测试集时不能无条件引用 `ModAvg` / `HeadAvg` / `Avg` / `Std`**，否则代码在本地跑得好好的，一提交就崩。

![图14](../../figures/fig14_data_structure.png)


---

## 第 2 步：认清三条硬约束

这三条不是「建议」，是**踩了就直接失去名次**的。

### 约束一：`ContextID` 里有金标泄漏，必须第一时间销毁

实测（`figures/fig11_order_leakage.png`）：

| 关系 | Spearman ρ |
|---|---|
| 组内 `ContextID` vs `HeadAvg`（en-nn） | **0.985** |
| 组内 `ContextID` vs `HeadAvg`（de-nn） | **0.976** |
| 组内 `ContextID` vs `Avg`（en-pv） | **0.994** |
| 组内 `ContextID` vs `Avg`（de-pv） | **0.990** |
| 组内 `ContextID` vs `ModAvg` | 0.298 / 0.221（随机序） |

**100% 的目标 ρ > 0.5。** 这意味着：一个**完全不看文本**、只按「组内行号排名」输出的垃圾基线，在 `en-pv` 上就能拿到 **ρ = 0.62**、`de-pv` **0.52**——已经追平 ComposiGen 已发表最好系统（0.50）。

这不是学术问题，是**评测漏洞**。而且**按行随机划分也挡不住**（必须按目标划分，见约束二），只删除列但保留原始行序也挡不住。

```python
# 载入后立刻执行，顺序不能反
df = df.drop(columns=["ContextID"])
df = df.sample(frac=1, random_state=42)
```

### 约束二：必须目标级互斥划分，且要复刻官方的分箱分层

官方原文：splits do not overlap in targets；先按组合性分箱再分层抽样；四个组合各自独立划分。

本地 dev 必须照抄，否则本地调参结论不可迁移：

```python
# 分箱 → 按箱配额在「目标层面」抽样 → 同一目标的所有行整体归属一侧
bins = [0, 1, 2, 3, 4, 5]
df["_bin"] = pd.cut(df["ModAvg"], bins)        # NN 用两列，PV 用 Avg
targets = df.groupby(key)["_bin"].first()      # 一个目标一个箱
dev_targets = targets.groupby("_bin", group_keys=False).apply(
    lambda s: s.sample(frac=0.2, random_state=42))
dev_mask = df[key].isin(dev_targets.index)
```

### 约束三：输出不能是常数，否则指标 2 直接 `nan`

指标 2 要求「同一目标下，系统预测的 std」与「人工评分的 std」算相关。
如果你每个目标输出同一个数 → std ≡ 0 → 相关系数的输入是常数 → **未定义**。

---

## 第 3 步：跑通一条最小管线

目标是**今天就能跑出合法提交**，而不是拿到好成绩。先建立下限。

```
读四个 TSV
   ↓
去重（仅 5 条重复）→ 删 ContextID → 物理打乱        ← 约束一
   ↓
目标级互斥划分 + 分箱分层（80/20 或 K 折）            ← 约束二
   ↓
定位目标在句中的跨度（hit_start / hit_end）           ← 第 5 步的级联策略
   ↓
【基线 A】目标均值常量         → 指标 1 高，指标 2 = nan（不合格）
【基线 B】目标均值 + 小噪声   → 指标 2 有值，但两项都低（仅验证管线通）
【基线 C】静态词向量 PPMI/SG  → 第一个像样的结果
   ↓
导出 4 个 pred.tsv → zip → 提交
```

**第 3 步当天就应该做完基线 B 并提交一次**，用官方返回的 `scores.txt` 校验你的本地指标实现和提交格式。这是唯一能确认「你算的 ρ 和官方算的 ρ 是同一个东西」的渠道。

### 定位级联（第 5 步会展开，这里先给命中率）

```
S0 给出各子集的直接命中率；S1-S4 只对前一级失败的目标生效，列出增量。

┌──────┬────────────────────────────────────────┬───────┬───────┬────────┬────────┐
│ 级别 │ 策略                                   │ en-nn │ de-nn │ en-pv  │ de-pv  │
├──────┼────────────────────────────────────────┼───────┼───────┼────────┼────────┤
│ S0   │ 精确字符串匹配                         │ 98.6% │ 95.9% │ 33.6%  │ 20.8%  │
│ S1   │ 转写还原 (变音符号展开为 ae/oe/ue)     │ +0.0% │ +4.0% │ +0.0%  │ +0.1%  │
│ S2a  │ 德语前缀屈折 (ab+gehauen -> abgehaten) │ n/a   │ +0.1% │ n/a    │ +49.3% │
│ S2b  │ 德语后置分离 (粒子跑到句尾)            │ n/a   │ +0.0% │ n/a    │ +7.5%  │
│ S3   │ 词干 / 前缀匹配                        │ +1.4% │ n/a   │ +52.6% │ n/a    │
│ S4   │ 相邻窗口                               │ +0.0% │ +0.0% │ +0.1%  │ +0.0%  │
│ 累计 │ 级联命中总率                           │ 100%  │ 100%  │ 86.3%  │ 77.7%  │
└──────┴────────────────────────────────────────┴───────┴───────┴────────┴────────┘
```

**一个有用的副产品**：德语粒子 `ab`/`an`/`aus`/`auf` 在 **100%** 的德语 PV 例句中**字面出现**。所以「先找粒子、再找它前面的词」是一条可靠的锚定策略。

---

## 第 4 步：三项指标怎么算

```python
import numpy as np, pandas as pd
from scipy import stats

def eval_all(df, key, cols):
    """df 需含 key 列、cols 中的预测列与对应金标列。"""
    out = {}
    for c in cols:
        gold, pred = df[c + "_gold"], df[c + "_pred"]
        # 指标 1：全例句层的 Spearman
        rho = stats.spearmanr(pred, gold).statistic
        # 指标 3：全例句层的 RMSE
        rmse = float(np.sqrt(((pred - gold) ** 2).mean()))
        # 指标 2：先在目标层各自算 std，再跨目标算 Spearman
        sd_p = df.groupby(key)[c + "_pred"].std(ddof=1)
        sd_g = df.groupby(key)[c + "_gold"].std(ddof=1)
        var = stats.spearmanr(sd_p, sd_g).statistic
        out[c] = dict(rho=float(rho), var_rho=float(var), rmse=rmse,
                      n_targets_std=int(sd_p.notna().sum()))
    return out
```

三点必须注意：

1. **指标 2 只在例句数 ≥ 2 的目标上可算**。实测复合词有 443–461 个目标可算，小品词动词只有 158 个 → **PV 上的指标 2 方差很大，不要过度解读 0.05 量级的差异**。
2. **指标 2 有一个被很多人忽略的「作弊上界」**。我实测了一个「每行填该目标的真实均值」的系统：指标 1 可达 ρ = 0.70–0.92（`de-nn Head` 高达 **0.918**），RMSE 0.33–0.76，但**指标 2 全部 `nan`**。这说明指标 1 和 3 **并不强制你使用语境**——只有指标 2 在管这件事。
3. **指标 1 和指标 2 存在张力**：过度压缩输出方差 → 指标 2 掉分；过度放大/加噪声 → 指标 1 的排序被破坏。**把「输出缩放系数」当显式旋钮，在 dev 上画 指标1 ρ vs 指标2 ρ 的帕累托前沿**，不要用单一验证指标选点。

![图12](../../figures/fig12_eval_metrics.png)

---

## 第 5 步：四条必须先做的预处理

| # | 规则 | 为什么 | 删掉的后果 |
|---|---|---|---|
| **P1** | 载入即 `drop(columns=["ContextID"])` + `sample(frac=1, random_state=42)` | 行序编码的是金标（ρ=0.98） | 评测分数虚高，且**无法在本地发现** |
| **P2** | 目标级互斥划分 + 分箱分层 | 官方就是这么划的 | 同一表达跨划分泄漏，泛化能力被高估 |
| **P3** | **不要全量剥离括号** | 实测末位括号内容等于目标词的比例是 **0/2、0/6、0/5、0/5**，括号是语料原文正当括注 | 剥离后 `en-pv` **66.7%**、`de-pv` **79.4%** 的例句再也定位不到目标 |
| **P4** | **不要用字符串切片切德语复合词** | Fugen-s 连接成分命中率只 59.1% | 切出来的「成分」根本不是成分 |

关于 P3 的正确做法：括号**只在定位失败时**作为备选匹配面，而不是无条件删除。

关于 P4 的正确做法：用**形态分析器**做德语复合词预切分，把切分版与未切分版做 ensemble，再把「切分置信度」作为一个特征送进模型。

---

## 第 6 步：建模的四条路线与推荐顺序

按投入产出比排序。**不要一开始就上重型模型**——实测证据显示小数据下重型模型没有优势。

### 🥇 路线 1：静态分布式表示（第一天就能跑完）

用 `word2vec` / `GloVe` 取两个成分的向量，拼接后回归到人工分数。

**为什么排第一**：ComposiGen 实测中 **Skip-gram（0.565）赢过 BERT/FastText**；Couch Potato 的纯文本 Skip-gram 做到 0.565；EACL 2023 里 **word2vec（0.525）超过所有 BERT 配置（最好 0.368）**。在去语境化的评测上，静态向量就是够的。

**但记住**：静态向量天然没有语境敏感性，**指标 2 拿不到**。所以它只适合当基线和特征源。

⚠️ **不要用 PPMI**。在历时/低频场景下它**恒负相关**（−0.12 / −0.10 / −0.14）。改用 LMI 或 LLR。

### 🥈 路线 2：上下文编码器 + 系统性层扫描（主力路线）

对句子做 token-level 前向，取目标词的隐藏层向量 → 回归。

**四条来自 EACL 2023 的硬结论**：

1. **不要用最后四层**。最后四层的最大 ρ 只有 **0.248**，而 **layer 1 最优 0.706**。
2. **不要用 CLS**。所有含 CLS 的估计都是最弱的一批，最低 **−0.649**。
3. **token-level 优于 type-level**（+0.01 ~ 0.06）。本任务要的就是 token-level。
4. **例句数的边际收益递减很快**：10→100 是 +0.067，100→1000 只 +0.017。每目标只有 7–10 条，正处在这条曲线最陡处之前——**所以逐例预测是必须的，没有「先汇总再预测」的捷径**。

**关键设计**：德语复合词在 subword tokenize 下会被切碎，成分向量拿不到 → **必须预切分**（EMNLP 2023 实测预切分收益最大）。

### 🥉 路线 3：加专用特征分支

三个特征源各自有依据：

| 特征 | 依据 |
|---|---|
| 成分共现 / 信息论度量（LMI、LLR） | `2020.winlp` 系列 |
| 成分独立性：把句子里的目标替换成单个成分，看语言模型的困惑度变化 | `S13-1038`、ACL 2021「边际似然」 |
| 频率 / 具体性 | 复合性研究的基本控制变量 |

注意 S13-1038 的一个**重要负面结果**：人类标注者自己就认为**中心语对整体组合性几乎没有解释力（ρ = 0.1385）**。中心语维度存在天然天花板，不要在上面死磕。

### 4️⃣ 路线 4：LLM 分支（务必带未见表达切分）

Couch Potato、ComposiGen、ACL 2025 都做了 LLM 路线，但有两个必须警惕的信号：

- **优势可能来自数据泄漏**。Reddy 数据 2011 年就公开，ChatGPT 得 0.736；ComposiGen 新采数据只有 0.44。**本任务的历史语料例句有高泄漏风险**，必须做未见表达（unseen MWE）切分。
- **ACL 2025 实测历时特征只有 +0.006 的微弱增益**。历时信息的价值更多在**可解释性**而非性能。

### 组件融合的已知结论

- **不要把两个成分简单相加当整体表示**（LREC 2024：compound−constituents 几乎从不优于单成分，0.710 vs 0.741）。
- **别用错成分预测目标**（Konvens 2023：特征错配直接崩到随机水平 0.585）。
- **在四个子集上四个模型都输给 late fusion**（ComposiGen）。

---

## 第 7 步：导出与提交

这是最容易翻车的地方。完整规则与自检骨架见 [`预处理分析计划.md` §9](../../PREPROCESSING.md)，这里只列清单。

### 只有 4 个文件，不是 6 个

| 文件名 | 内容 |
|---|---|
| `en-nn-pred.tsv` | **3 列**：`ContextID` ⇥ modifier 预测 ⇥ head 预测 |
| `en-pv-pred.tsv` | **2 列**：`ContextID` ⇥ 预测 |
| `de-nn-pred.tsv` | 3 列 |
| `de-pv-pred.tsv` | 2 列 |

> **6 个评测设置 vs 4 个提交文件**：NN 的「修饰语」和「中心语」是**同一个文件里的两列**，不是两个文件。别混淆。

### 五条硬性要求

1. **无表头。** 写成 `ContextID\tModAvg\tHeadAvg` 不会报格式错误——评测脚本会把它当成一条预测而**整体错位**，后续所有行全部串位，静默给出离谱分数。
2. **`ContextID` 原样透传。** 不重编号、不去重、不做 `astype(int)`（可能不是数字串）。
3. **NN 列序是 modifier 在前、head 在后。** 建议在导出函数里写单元测试断言列序。
4. **值域保持在 0–5。** 做了标准化/校准的，导出前必须反变换，否则 RMSE 灾难性变差。
5. **不能是常数**（约束三）。

```python
body = pd.DataFrame({"ContextID": test["ContextID"].astype(str),
                     "mod": mod_pred, "head": head_pred})
assert np.isfinite(body[["mod", "head"]].to_numpy()).all()
assert body["mod"].between(0, 5).all() and body["head"].between(0, 5).all()
assert len(body) == len(test)
assert set(body["ContextID"]) == set(test["ContextID"].astype(str))
body.to_csv("en-nn-pred.tsv", sep="\t", index=False, header=False)  # 无表头
```

然后 `zip en-nn-pred.tsv en-pv-pred.tsv de-nn-pred.tsv de-pv-pred.tsv`。

---

## 第 8 步：评估与自检

### 八个必须过的验收项

- [ ] `ContextID` 已删除，全管线无行号衍生特征
- [ ] 「组内秩」自检基线已跑出（`en-pv` ≈ 0.62 附近即为泄漏预警）
- [ ] 未做全量括号剥离
- [ ] 定位成功率按子集报告：≥100% / 100% / 86% / 78%
- [ ] 训练/验证按目标互斥划分，且做了分箱分层
- [ ] NN 保留 `ModAvg` 与 `HeadAvg` 两个头
- [ ] 本地能同时算出三项指标，且已与 Trial 的 `scores.txt` 对齐
- [ ] 输出不是常数

### 报告口径

官方在四格（语言 × 类型）之上还拆出 NN 的 modifier/head，实际是 **6 格 + 跨格均值**。所有结果都按 6 格分层报告，总分只作为附加项。

### 历时分层（作为卖点）

虽然测试集不告诉你年代，但官方说「会在最终分析中补充按时间来源的统计」。用表层启发式构造 `era_proxy` 做**分层评估**（不是入模特征），报告「模型在 `era_proxy` 两端的性能差」——这是本任务区别于普通组合性预测的地方。

⚠️ 但要知道：**这条路的信号很弱**。实测表层年代信号覆盖率只有英语 6.2–6.7%、德语 36.2–37.6%，与评分的所有 |ρ| ≤ 0.17，且符号在子集间翻转。**别指望它当特征拿分。**

---

## 常见问题速查

| 症状 | 原因 | 处理 |
|---|---|---|
| dev ρ 高得离谱（`en-pv` > 0.65） | 行序泄漏没清干净 | 检查 `ContextID` 是否真的删了、行序是否真的打乱了 |
| 指标 2 = `nan` | 组内输出是常数 | 加上下文依赖；或换逐例预测模型 |
| 指标 1 高但指标 2 差 | 输出被压缩了 | 加大输出方差，或画帕累托前沿找折中点 |
| 德语复合词定位失败率高 | Fugen-s | 形态分析器预切分，不要字符串切片 |
| 德语 PV 定位失败 | 粒子分离到句尾 | 用粒子锚定（`ab/an/aus/auf` 100% 字面出现） |
| 剥离括号后定位率暴跌 | 括号被当成了标注标记 | 别剥（实测比例 0/2、0/6、0/5、0/5） |
| 提交报格式错误但内容看着没问题 | 写了表头 | `header=False` |
| 分数提交后离谱地好 | 泄漏或列序错位 | 检查 NN 列序是 mod→head |
| 德语效果远差于英语 | 复合词切分 + 语料更稀疏（7.15 条/目标） | 预切分 + 从英语迁移 |

---

## 附：下一步

1. **今天**：跑通基线 B（目标均值 + 噪声），用 Trial 的 `scores.txt` 校验指标实现与提交格式。
2. **本周**：完成定位级联，把定位成功率打到 100% / 100% / 86% / 78% 以上。
3. **下周**：上路线 2，做完整的层扫描。这是主力投入所在。
4. 完整规则清单（20+ 条 R 编号规则）：[`预处理分析计划.md`](../../PREPROCESSING.md)
5. 概念理解：[`教学指南`](tutorial.md)(教学指南_组合性预测从零到一.md)
