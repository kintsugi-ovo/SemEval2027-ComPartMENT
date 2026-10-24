# 文献索引 · SemEval-2027 Task 5 ComPartMENT

## 📄 文档（先按用途挑一份）

| 文档 | 用途 | 什么时候读 |
|---|---|---|
| [**入门指南**](guide/getting_started.md) | 以数据结构为骨架的工程路径 | 准备动手时，**优先读这份** |
| [**教学指南：从零基础到读懂 ComPartMENT**](guide/tutorial.md) | 五段式讲透核心概念 | 完全不懂「组合性预测」时 |
| [文献综述](guide/literature_review.md) | 学术脉络与 39 篇文献详解 | 要写论文 / 找引用时 |
| [`PREPROCESSING.md`](../PREPROCESSING.md) | 20+ 条预处理规则与流水线 | 写代码前查清单 |
| [`AGENTS.md`](../AGENTS.md) | 项目记忆 | **任何工作开始前先读** |

---

## 📁 资源目录

- PDF 原文：`papers/`
- 导读文档：`guide/`
- 任务官方训练数据：`../data/train/`

---

## 完整清单（39 篇，均已下载 PDF）

### 综述 / 总览

| 文件 | 文献 | 年份/出处 |
|---|---|---|
| `2024.tacl-1.33.pdf` | Miletić & Schulte im Walde. *Semantics of Multiword Expressions in Transformer-Based Models: A Survey* | TACL 2024 |
| `2021.acl-long.543.pdf` | Hamon et al. *Lexical Semantic Change Discovery* | ACL 2021 |
| `2023.emnlp-main.572.pdf` | Cassotti et al. *Understanding Computational Models of Semantic Change* | EMNLP 2023 |

### 组合性预测 · 核心方法

| 文件 | 文献 | 年份/出处 |
|---|---|---|
| `2025.acl-long.984.pdf` | Miletić & Schulte im Walde. *Modeling the Evolution of English Noun Compounds with Feature-Rich Diachronic Compositionality Prediction* | ACL 2025 |
| `2023.eacl-main.110.pdf` | Miletić & Schulte im Walde. *A Systematic Search for Compound Semantics in Pretrained BERT* | EACL 2023 |
| `2023.emnlp-main.1002.pdf` | Jenkins, Miletić & Schulte im Walde. *To Split or Not to Split* | EMNLP 2023 |
| `2020.lrec-1.539.pdf` | Alipoor & Schulte im Walde. *Variants of Vector Space Reductions...* | LREC 2020 |
| `x_S13-1038.pdf` | Schulte im Walde, Müller & Roller. *Exploring Vector Space Models to Predict the Compositionality of German Noun-Noun Compounds* | *SEM 2013 |
| `2025.findings-acl.561.pdf` | Kurtyigit, He, Phelps, Idiart, Schulte im Walde et al. *A Couch Potato is not a Potato on a Couch* | ACL Findings 2025 |
| `2026.lrec-1.732.pdf` | Godbersen et al. *The ComposiGen Dataset of Compounds and their Compositionality* | LREC 2026 |
| `x_2607.11667v1.pdf` | Jenkins, Schulz, Miletić, Schulte im Walde. *Losing My Composure: Predicting Compositionality Over Time* | arXiv 2026 / CL 投稿 |
| `2022.mwe-1.1.pdf` | Miletić & Schulte im Walde et al. *Figurative Language in Noun Compound Models across Target Properties, Domains and Time* | MWE 2022 |
| `2025.swisstext-1.8.pdf` | *Probing BERT for German Compound Semantics* | SwissText 2025 |
| `2026.nonliteral-1.9.pdf` | Jenkins. *Contextualising (Im)plausible Events Triggers Figurative Language* | NLiteral 2026 |
| `2026.cmcl-1.12.pdf` | *Floating or Suggesting Ideas? Metaphorical and Literal Verb–Object Constructions* | CMCL 2026 |
| `2026.tacl-1.54.pdf` | *Literally Concrete or Figuratively Abstract? Multilingual Concreteness Norms for Verb-Object Expressions* | TACL 2026 |
| `2024.starsem-1.18.pdf` | *VOLIMET: Parallel Corpus of Literal and Metaphorical Verb-Object Pairs (EN–DE / EN–FR)* | StarSem 2024 |

### 历时建模与语义变化

| 文件 | 文献 | 年份/出处 |
|---|---|---|
| `x_W19-4729.pdf` | Dhar, Pagel & van der Plas. *Measuring the Compositionality of Noun-Noun Compounds over Time* | CLfHL 2019 |
| `x_1906.02563v1.pdf` | 同上（arXiv 版） | arXiv 1906.02563 |
| `2023.konvens-main.4.pdf` | Maurer et al. *Classifying Noun Compounds for Present-Day Compositionality* | KONVENS 2023 |
| `2024.lrec-main.1517.pdf` | Mahdizadeh Sani et al. *What Can Diachronic Contexts and Topics Tell Us...?* | LREC-COLING 2024 |
| `2025.findings-acl.566.pdf` | *Multi-word Measures: Modeling Semantic Change in Compound Nouns* | ACL Findings 2025 |
| `2024.nlp4dh-1.23.pdf` | *Visualising Changes in Semantic Neighbourhoods of English Noun Compounds over Time* | NLP4DH 2024 |
| `x_P16-1141.pdf` | Hamilton, Leskovec & Jurafsky. *Diachronic Word Embeddings Reveal Statistical Laws of Semantic Change* | ACL 2016 |
| `2021.eacl-srw.25.pdf` | Schlechtweg et al. *Explaining and Improving BERT Performance on Lexical Semantic Change Detection* | EACL SRW 2021 |
| `2020.semeval-1.8.pdf` | IMS at SemEval-2020 Task 1: How Low Can You Go? | SemEval 2020 |
| `2020.evalita-1.65.pdf` | CL-IMS @ DIACR-Ita: BERT is still not Outperforming SGNS | EVALITA 2020 |
| `2026.lrec-1.142.pdf` | *What Are LLMs Doing to Scientific Communication?* | LREC 2026 |
| `2026.dialres-1.17.pdf` | *WoVis: Interactive Visualization of Word Embeddings for Semantic Change...* | DialRes 2026 |

### 数据集与标注方法学

| 文件 | 文献 | 年份/出处 |
|---|---|---|
| `x_L16-1362.pdf` | Schulte im Walde et al. *GhoSt-NN: A Representative Gold Standard of German Noun-Noun Compounds* | LREC 2016 |
| `x_W13-1005.pdf` | Roller, Schulte im Walde & Scheible. *The (Un)expected Effects of Applying Standard Cleansing Models to Human Ratings on Compositionality* | MWE 2013 |
| `x_W14-0818.pdf` | Roller & Schulte im Walde. *Feature Norms of German Noun Compounds* | MWE 2014 |
| `2020.lrec-1.537.pdf` | *A Domain-Specific Dataset of Difficulty Ratings for German Noun Compounds* | LREC 2020 |
| `2021.starsem-1.24.pdf` | *Compound or Term Features? Analyzing Salience in Predicting the Difficulty of German Noun Compounds* | StarSem 2021 |
| `2024.lrec-main.1534.pdf` | *Willkommens-Merkel, Chaos-Johnson, and Tore-Klose: Modeling the Evaluative Meaning of German Personal Name Compounds* | LREC 2024 |
| `2025.analogyangle-1.1.pdf` | *Tore-Klose: Human Association Norms for German Personal Name Compounds* | 2025 |
| `x_2511.19816v1.pdf` | *Breaking Bad: Norms for Valence, Arousal, and Dominance for 10k English MWEs* | arXiv 2511.19816 |

### 方法学支撑 / 工具性文献（1 篇）

| 文件 | 文献 | 年份/出处 |
|---|---|---|
| `x_2024.emnlp-main.354.pdf` | *On the Reliability of Psychological Scales on Large Language Models* | EMNLP 2024 |
| `2021.acl-long.161.pdf` | *You should evaluate your language model on marginal likelihood over tokenisations* | ACL 2021 |

---

## 关于 CoMWE 数据集的溯源

任务描述中提到的 **CoMWE** 是对既有组合性金标准的**再分发与统一格式版本**，不是新采集的数据集。其源头论文为：

| 语言 | 源头 | 文件 |
|---|---|---|
| 英语 | Reddy, McCarthy & Manandhar (2011)，90 词，0–5 | 经 `2020.lrec-1.539` / `2025.findings-acl.561` 使用 |
| 英语 | Cordeiro et al. (2019)，280 词 | `2023.eacl-main.110` |
| 英语 | Salehi et al. (2014, 2015) | — |
| 德语 | GhoSt-NN，Schulte im Walde et al. (2016)，868 词，1–6 | `x_L16-1362.pdf` |
| 法语 / 葡萄牙语 | 各自的金标准数据集 | — |

> 注意：`2020.lrec-1.539` 全文并未出现 "CoMWE" 字样，其金标准实为 Reddy 90 词。撰写论文引用时请核实这一溯源关系。
