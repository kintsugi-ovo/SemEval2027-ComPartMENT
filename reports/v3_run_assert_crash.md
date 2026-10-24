# v3.2 云端实跑：口径断言崩溃分析（2026-10-23 第二轮）

> 日志：`logs/v3/round2/compartment-deberta-baseline.log`、`logs/v3/round2/compartment-qwen-instruct-baseline.log`
> 上一轮分析：`reports/v3_run_nan_crash.md`（NaN 根因）
> 本阶段结论：**上一轮的 NaN 修复已生效**（`grad_norm` 从 `nan` 变成 `23.68`），
> 但暴露出 **v3.2 新加的检查本身有 bug** ⇒ 崩在「口径一致性断言」上。

---

## 1. 两条基线的崩溃点（完全一致）

```
—— 探针（t3, micro-batch=8）——
{'loss': '1.267', 'grad_norm': '23.68', 'learning_rate': '1.905e-07', ...}   ← ★ 不再是 nan
t3 micro-batch=8: 通过，峰值显存 8.09 GiB，每样本 225.25 ms

—— 开训前检查（preflight）——
AssertionError: warmup_steps=105 ≠ 按同一口径算出的 35：口径漂移，拒绝开训
  at _assert_args_consistent(args, n_train)
  at make_trainer(m, out_dir, data, epochs, check_consistency, **overrides)
  at _preflight_body → _gc_trainer = make_trainer(model, out_dir / "_gradcheck", data,
                                                  max_steps=1, num_train_epochs=1, ...)
```

**两条基线同一行、同一句话** ⇒ 共因，与模型无关。

---

## 2. 根因：体检 trainer 覆盖了 `num_train_epochs`，而 `warmup_steps` 只跟 epochs 走

两处公式（`notebooks/vN/ComPartMENT基线_*.ipynb`）：

| 位置 | 公式 | 云端取值（en-nn, n_train=2801, bs=8, ga=1） |
|---|---|---|
| `build_training_args` | `steps_per_epoch = ceil(n/(bs·ga))`；`total = steps_per_epoch·ep`；`warmup = int(total·0.1)` | ep=3 ⇒ 351·3=1053 ⇒ `int(105.3)` = **105** |
| `_assert_args_consistent` | `_expected_warmup = int(ceil(n/(bs·ga))·ep·0.1)` | ep=3 ⇒ **105**（一致 ✓）<br>ep=1 ⇒ **35**（与实际 105 不一致 ✗） |

⇒ **正式训练路径（ep=3）两式一致、不会崩**；
⇒ **但任何 `num_train_epochs=1` 的 trainer**，断言会拿 ep=1 的期望值（35）去比 ep=3 的实际值（105）⇒ **必然报「口径漂移」**。

本阶段崩的就是两处「体检用途」的 trainer：

1. **冒烟 trainer**：`max_steps=5, num_train_epochs=1` —— **v1 就存在**；
2. **v3.2 新加的「真实损失反向体检」trainer**：`max_steps=1, num_train_epochs=1` —— 我上轮刚写的。

---

## 3. 为什么 v1 能跑通（用户要求对照的第一版）

逐格对比 v1 与当前版本（`v1/ComPartMENT基线_DeBERTa-v2-xlarge.ipynb` vs 当前 ipynb）：

| 项 | v1（能跑通） | 当前 |
|---|---|---|
| preflight 的 Trainer 数量 | 1 个（冒烟） | 3 个（探针 / 真实损失体检 / 冒烟） |
| **口径一致性断言** | **没有**（v3 才加） | 有 |
| 档位降级链 | 没有 | 有（t0–t6） |
| 损失分项 | 只有 Huber | Huber + order/mse/guard/xrank |

★ **v1 的 preflight 冒烟 trainer 同样传了 `max_steps=5, num_train_epochs=1`，
但 v1 没有那条断言 ⇒ 直接跑通。**
⇒ **「v3 加了口径断言」这件事本身，把 v1 的冒烟 trainer 变成了潜伏 bug。**
它一直存在，但被上一轮的 NaN **提前崩溃掩盖**了 —— 崩在梯度流验证，根本走不到冒烟。

### 为什么本地测不出来

| 条件 | 本地（SMOKE） | 云端 |
|---|---|---|
| `EPOCHS` / `JOINT_EPOCHS` | **1** | **3** |
| 每子集训练集行数 | 400（截断） | 全量（en-nn 2801） |
| ⇒ `_expected_warmup` | 35，**而 args 也是按 ep=1 算的 35** | 期望 35 vs 实际 105 ⇒ 崩 |

★ 本地 `EPOCHS=1` 时「期望值」与「实际值」**恰好相等** ⇒ 断言自洽 ⇒ 永远测不出来。
★ 这就是「本地过、云端炸」的又一例，而且这次踩坑的是**刚写完不久的那段代码**。

---

## 4. 修法：根治，不再依赖调用点自觉

### 4.1 判定逻辑收敛到**一个函数**（`notebooks/vN/ComPartMENT基线_*.ipynb`）

```python
def should_check_consistency(overrides, check_consistency=None):
    if check_consistency is not None:
        return bool(check_consistency)
    # 只要覆盖了 max_steps / num_train_epochs ⇒ 这是「体检用途」的 trainer ⇒ 不做口径断言
    return not (("max_steps" in overrides) or ("num_train_epochs" in overrides))
```

`make_trainer` 统一调用它 ⇒ **以后任何人新增体检 trainer 都不会再踩这个坑**，
不需要每个调用点记得传 `check_consistency=False`。

### 4.2 三处调用点全部标注（双保险）

- 探针：显式 `check_consistency=False`（它故意改 batch/累积）
- 真实损失反向体检：自动豁免 + 注释说明
- 冒烟：自动豁免 + 注释说明（**v1 的老坑**）

### 4.3 把「云端口径」搬进本地单测（`test_fit_logic.py` 第 6 组）

用**云端 5 次运行的真实参数**（训练集大小取自云端日志实测）在本地运行一遍正式训练路径的断言：

| 运行 | n_train | 每轮批数 | 总更新步 | warmup 步（占比） | 结果 |
|---|---|---|---|---|---|
| en-nn | 2801 | 351 | 1053 | 105（10.0%） | ✅ |
| de-nn | 2638 | 330 | 990 | 99（10.0%） | ✅ |
| en-pv | 1245 | 156 | 468 | 46（9.8%） | ✅ |
| de-pv | 1192 | 149 | 447 | 44（9.8%） | ✅ |
| joint | 7860 | 983 | 2949 | 294（10.0%） | ✅ |

外加三条判定规则断言（体检 trainer 自动豁免 / 正式训练保持开启 / 显式传参优先）。

★ **这是本阶段最重要的一步**：把「云端才会求值的那段断言」变成**本地每次都能跑**的回归测试。
★ 顺带验证了训练后回读断言的容差：`wf = warmup/max_steps` 与 `WARMUP_RATIO` 的偏差
最大是 en-pv 的 0.0017，容差 `max(0.02, 2/max_steps)` ≥ 0.0043 ⇒ 全部通过 ✓

---

## 5. 本地验证结果

| 检查项 | 结果 |
|---|---|
| DeBERTa 本地实跑 | **46.2 s / 0 报错** |
| Qwen 本地实跑 | **51.7 s / 0 报错** |
| preflight 日志 | 打印两处「体检用途的 Trainer：覆盖了 ['max_steps','num_train_epochs'] ⇒ 自动关闭口径断言」 |
| 真实损失反向 | `loss=1.3611 | 最大梯度=3.483e+00 | 非有限梯度的参数数=0` ✅ |
| `test_fit_logic.py` | ✅ 全部通过（新增第 6 组：云端 5 次运行参数的口径断言） |
| 两 ipynb 一致性 | ✅ 仍**仅 3 格不同**（格 0/5/53） |

---

## 6. 这一轮沉淀下来的纪律

1. **「本地测不出」的根因要写进表**：本地 `EPOCHS=1` vs 云端 `EPOCHS=3` 是本项目最隐蔽的差异源；
   凡是**依赖轮数/数据量求值**的断言，都必须用云端真实参数在本地单测里跑一遍。
2. **加断言会引入新 bug**：v1 能跑通 ⇒ v3 加的每一条断言都是「v1 没有的新失败点」，
   加断言时必须回头检查**v1 里已有的代码路径**（这里是冒烟 trainer）是否被新断言波及。
3. **豁免逻辑要收敛到单一入口**：`should_check_consistency()` 一个函数判定，
   而不是「每个调用点记得传 False」——后者在加了新调用点时必然漏。
4. **崩溃点要能自证**：上一轮的「回归头没有梯度」是假错（模型被探针污染）；
   本阶段的「口径漂移」是真错但指向了**检查本身**而不是被检查的东西 ⇒
   报错信息必须指向「谁的责任」（这里是「体检 trainer 不该走正式训练口径」）。
