# PAEMS Noise Model — Complete Specification

> 完整 PAEMS 表面码噪声系统说明文档
> 涵盖：经典噪声、串扰、泄漏、坏点 overlay 四层
> **版本**: v3 (2026-04-30)
> **校准对照**: Google Willow d=5 q4_7 / d=7 q6_7 r=30 真实 detection events

---

## 0. TL;DR

PAEMS 噪声系统由 **4 层完全解耦的机制**叠加组成：

```
Layer 1: 经典噪声 (T1/T2/fid/SPAM/...)        → gen_level_params.py
Layer 2: 串扰 (spectator depolarize on H/CX) → gen_pair_overrides.py
Layer 3: 泄漏 (post-processing flip)         → gen_leakage_overrides.py + post-process
Layer 4: 坏点 overlay (heterogeneous hot)    → 同 1+3 脚本里 --defect-seed
```

每层独立 4 级 (L1/L2/L3/L4 或 X1-X4)；每层都随码距 d² scaling；每层可单独启用或关闭。

---

## 1. 总体架构

```
                 ┌──────────────────────────────────────────────────────┐
                 │              Surface code circuit (Stim)              │
                 │             generate_surface_code_circuit()           │
                 └──────────────────────────────────────────────────────┘
                                          │
              ┌───────────────────────────┼───────────────────────────┐
              ▼                           ▼                           ▼
    ┌─────────────────┐      ┌──────────────────────┐    ┌───────────────────────┐
    │ Layer 1: 经典   │      │ Layer 2: 串扰         │    │ Layer 3: 泄漏          │
    │ T1/T2/fid/SPAM  │      │ DEPOLARIZE1 spectator │    │ post-process flip     │
    │                 │      │ after each H/CX       │    │ on detection events   │
    │ gen_level_      │      │ gen_pair_overrides    │    │ gen_leakage_overrides │
    │ params.py (L1-4)│      │ + crosstalk_X1-X4     │    │ + leakage_L1-L4       │
    └─────────────────┘      └──────────────────────┘    └───────────────────────┘
              │                           │                           │
              └───────────────────────────┼───────────────────────────┘
                                          ▼
                       ┌────────────────────────────────────┐
                       │   Layer 4: Defect overlay (跨层)    │
                       │   Heterogeneous hot qubits / CX    │
                       │   --defect-seed N (None = 关闭)    │
                       └────────────────────────────────────┘
                                          │
                                          ▼
                  最终 PAEMS noise JSON  →  inject + sample + post-process
                                          │
                                          ▼
                            (shots, n_detectors) bool array
                                          │
                                          ▼
                        Spitz 关联矩阵 / LER / 解码 ...
```

**核心设计原则：**

1. **完全解耦** — L1-L4 经典级与 X1-X4 串扰级、Leak1-Leak4 泄漏级、defect overlay **正交**：4×4×4 = 64 种组合 × 是否含坏点 = 128 配置
2. **4 级阶梯** — 每层都覆盖 next-gen → Willow → Sycamore → 早期 NISQ
3. **物理对标 Willow** — Layer 1-3 的 L2 / X2 / Leak2 是 Google 2024 实测靶
4. **随 d² scaling** — 每层都自然按 chip 大小扩展（Layer 1-3 通过 per-qubit/per-CX 参数，Layer 4 通过 fraction）

---

## 2. 生成 Pipeline

```
                    Step 1                  Step 2                  Step 3
              ┌─────────────────┐    ┌─────────────────┐    ┌────────────────────┐
   nothing → │ gen_level_params │ → │ gen_pair_         │ → │ gen_leakage_        │ → final.json
              │ --level 2 ...    │    │ overrides ...    │    │ overrides ...       │
              └─────────────────┘    └─────────────────┘    └────────────────────┘
              [Layer 1 only]         [+Layer 2 (xtalk)]     [+Layer 3 (leak)]
              [+Layer 4 if seed]     [Layer 4 already in]   [+Layer 4 if seed]

JSON 演化:
  step1: {qubits, cx_gates, crosstalk_pairs={}, _metadata}
  step2: 加 frequency_mhz + crosstalk_pairs
  step3: 覆盖 lp/sp on qubits + lp_propagation_prob on cx_gates
```

最终 JSON 喂给 `Surface_Code_Simulation/run_sampling.py`：
- 经典 + 串扰：通过 `inject_surface_code_noise()` 注入 Stim circuit
- 泄漏：通过 `simulate_surface_code_leakage_vectorized()` 单独算 + `postprocess_leakage()` 翻转 detection events

---

## 3. Layer 1: 经典噪声 (L1 / L2 / L3 / L4)

### 3.1 每 qubit 参数

| 参数 | 含义 | 分布 | L1 (next-gen) | **L2 (Willow)** | L3 (Sycamore) | L4 (Eagle/Wukong) |
|------|------|------|---------------|------------------|----------------|---------------------|
| `t1` | T1 (μs) | Normal | 600 ± 150 | **270 ± 70** | 100 ± 30 | 30 ± 10 |
| `t2` | T2 (μs) | Normal, ≤ 2T1 | 500 ± 120 | **200 ± 60** | 70 ± 25 | 20 ± 8 |
| `sqg_length` | 1Q gate 时长 (ns) | Normal | 25 | 30 | 35 | 60 |
| `sqg_fid_err` | 1Q gate err | **LogNormal** | 7e-5 | **2e-4** | 1e-3 | 2e-3 |
| `data_init_error` | reset err | **LogNormal** | 5e-4 | **1.5e-3** | 1e-2 | 3e-2 |
| `data_measurement_error` | M err | **LogNormal** | 1e-3 | **3e-3** | 2e-2 | 6e-2 |
| `measurement_spam_rate` | SPAM | **LogNormal** | 1.5e-3 | **4e-3** | 3e-2 | 9e-2 |
| `rd_length` | 测量时长 (ns) | Normal | 800 | 1500 | 2500 | 3000 |

### 3.2 每 CX 参数

| 参数 | L1 | L2 | L3 | L4 |
|------|----|----|----|----|
| `cx_fid_err` (LogNormal) | 1e-3 | **3e-3** | 1e-2 | 2e-2 |
| `cx_length` (ns) | 30 | 50 | 200 | 400 |

### 3.3 分布选择理由

**Normal (timing 参数)**: T1/T2/length 物理上对称分布，Willow 实测的 T1 直方图无明显偏度。

**LogNormal (error rate 参数)**: 错误率物理上 ≥ 0 且**右偏长尾**（少数比特/CX 的错误率显著高于中位数）。`log_std=0.4-0.5` 让 ~5% 抽样达 2-3× 中位数，~1% 达 5×，匹配 Willow / IBM calibration 实测。

**mean-preserving LogNormal** 实现：

```python
val = mean × exp(N(-log_std²/2, log_std))
```

保证 `E[val] = mean` 精确——这样 4 级的 `mean` 数值直接对应"芯片平均水平"。

### 3.4 文件位置

- spec: `level_params_spec.json`
- 生成器: `gen_level_params.py`
- 采样函数: `clip_norm()` / `clip_lognormal()`（自动按 spec 是否有 `log_std` 字段切换）

---

## 4. Layer 2: 串扰 (X1 / X2 / X3 / X4)

### 4.1 模型

每次 H/CX 门驱动 active qubit `q`，对每个 spectator qubit `s` 注入一次 DEPOLARIZE1 事件，强度 `χ(q, s)`：

```
χ(i, j) = A × max(Lorentzian(Δf, W), zz_floor) × spatial_factor(hop)

Lorentzian(Δf, W) = 1 / (1 + (Δf/W)²)
spatial_factor    = {hop=1: skip, hop=2: 1.0, hop=3: 0.3,
                     hop=4: 0.1, hop=5: 0.03, hop≥6: 0}
```

三个物理通道：
- **A × Lorentzian**：drive crosstalk（Δf 近时驱动微波耦合）
- **A × zz_floor**：always-on ZZ 残留（Δf 远时仍有 floor）
- **× spatial_factor**：物理 wiring 距离衰减

### 4.2 频率分配（cumsum + bipartite + auto-scale）

每个 PAEMS 配置自动给每个 qubit 分一个频率 (MHz)：

```python
# Step A: bipartite 着色（surface code 永远 bipartite）
G.add_edges(cx_gates)
color = nx.bipartite.color(G)

# Step B: 采样 n-1 个 Gaussian gap
gaps = rng.normal(mean_gap, std_gap, n-1)

# Step C: auto-scale（总跨度 > max_spread 时等比缩小到 max_spread）
if sum(gaps) > max_spread: gaps *= max_spread / sum(gaps)

# Step D: cumsum 得位置 + 居中
positions = cumsum(gaps) - center

# Step E: bipartite split（color 0 → 低半段，color 1 → 高半段）
# 1-hop 邻居频率差 ≥ 半跨度，避开 drive crosstalk on CX gates
```

### 4.3 4 级 Preset

| 参数 | X1 (future) | **X2 (Willow)** | X3 (Sycamore) | X4 (worst) |
|------|-------------|------------------|----------------|-------------|
| mean_gap (MHz) | 15 | 15 | 5 | 5 |
| std_gap (MHz) | 5 | 5 | 3 | 3 |
| max_spread (MHz) | 1000 | 1000 | 1000 | 1000 |
| **A** | 5e-4 | **8e-4** | 1e-3 | 2.5e-3 |
| **W** (MHz) | 3 | 5 | 7 | 10 |
| **zz_floor** | 2e-3 | 4e-3 | 4e-3 | 8e-3 |

实测 d=5 r=5 200K shots **defect 增量 ladder**：X1 +0.03pp / X2 +0.12pp / X3 +1.04pp / X4 +3.67pp（120× 跨度）。

### 4.4 1-hop 永远跳过

`generate_pairs()` 中 `if hop == 1: continue`——CX 直邻已被 CX 门噪声 (DEPOLARIZE2 + PAULI_CHANNEL_1) 覆盖，避免重复计算。

### 4.5 文件位置

- spec: `crosstalk_presets/crosstalk_X{1,2,3,4}.json`
- 生成器: `gen_pair_overrides.py`
- 注入器: `Surface_Code_Simulation/inject_basic_noise.py` 中的 `add_spectator_crosstalk()`
- 详细文档: `CROSSTALK_MODEL.md`

---

## 5. Layer 3: 泄漏 (Leak1 / Leak2 / Leak3 / Leak4)

### 5.1 模型

后处理流程（**不**注入 Stim circuit）：

```
1. 用经典+串扰电路采 raw measurement bits
2. 单独跑 leakage state 模拟 (vectorized numpy):
   - 每次 H/CX 上 |1⟩→|2⟩ 概率 lp
   - 每次 gate 上 |2⟩→|1⟩ 概率 sp
   - CX 时若一边已泄漏，partner 以 lp_propagation_prob 也变 |2⟩
3. 标记每个 measurement 是否被泄漏污染（同 round 任何 gate 触及 |2⟩）
4. flip_prob=0.5 翻转受污染的 measurement bit
5. 用 stim m2d converter 转成 detection events
```

### 5.2 三个参数

| 参数 | 含义 | 分布 |
|------|------|------|
| `lp` | per-gate \|1⟩ → \|2⟩ 概率 | LogNormal (mean preserved, log_std=0.7) |
| `sp` | per-gate \|2⟩ → \|1⟩ 概率 | scalar (无抖) |
| `lp_propagation_prob` | per-CX partner-leak 概率 | LogNormal (log_std=0.5) |

### 5.3 4 级 Preset

| Level | lp | sp | prop | 稳态 \|2⟩ |
|-------|-----|------|------|-----------|
| **L1** (next-gen) | 8e-6 | 0.04 | 0.02 | ~0.02% |
| **L2 (Willow)** | **4e-5** | **0.04** | **0.10** | **~0.1%** |
| **L3** (Sycamore) | 2e-4 | 0.04 | 0.30 | ~0.5% |
| **L4** (NISQ) | 1e-3 | 0.04 | 0.50 | ~2.5% |

每个参数有 max cap：
- `lp.max` = 5e-3（lp=0.5%/gate 已极端，5e-3 是物理 sanity 上限）
- `prop.max` = 0.8（不让 100% 传染）
- `sp` 跨级保持 0.04 不变（恢复率主要由 reset gate 决定，跟比特质量关系小）

### 5.4 关键设计决策

**Rule 2 删除（2026-04-29）**：原 `calculate_affected_states_vectorized()` 把 "CX 时一边泄漏 → 另一边的 measurement 也打 affected" 标记。这与 Layer 3.1 的 `lp_propagation_prob` 物理传播双重计数，导致 hop≥2 cross-qubit 关联整体偏高 ~2-3×。已删除。

**flip_prob = 0.5 固定**：受污染的 measurement 50% 概率被翻——对应"测到 |2⟩ 时实际读出值是随机的"。

### 5.5 文件位置

- spec: `leakage_presets/leakage_L{1,2,3,4}.json`
- 生成器: `gen_leakage_overrides.py`
- 模拟器: `Surface_Code_Simulation/inject_leakage_noise_vectorized.py`
- 后处理: `Surface_Code_Simulation/run_sampling.py` 的 `postprocess_leakage()`

---

## 6. Layer 4: Defect Overlay（坏点机制）

### 6.1 物理动机

> 真实芯片每次标定后总有少数 qubit / CX 没达到典型水平——TLS 缺陷、热点谐振器、封装应力。这些点的**某些**（不是全部）参数比中位数高 5-15×。

### 6.2 异质化机制

每个 hot 点用独立 sub-RNG（从 master `--defect-seed` 派生）抽：

1. **K 个候选参数** (K ∈ [k_min, k_max])
2. **每个抽中参数的独立 multiplier** ~ Uniform[mult_min, mult_max]

Canonical 参数顺序（两脚本必须严格一致）：

```python
CANON_QUBIT_PARAMS = ['sqg_fid_err', 'data_init_error',
                      'data_measurement_error',
                      'measurement_spam_rate', 'lp']
CANON_CX_PARAMS    = ['cx_fid_err', 'lp_propagation_prob']
```

**结果：每个 hot 点都有自己的"defect 签名"**，不同点恶化的参数不同，倍数也不同。

### 6.3 d=5 实例 (defect_seed=7)

```
hot qubit q45: data_meas × 6.1 + spam × 8.4         readout 烂比特
hot qubit q31: data_init × 9.4                       reset 烂比特
hot qubit q34: lp × 9.1                              纯泄漏比特（gate/readout 正常）
hot CX  c5:    cx_fid × 9.3                          CX 门保真度烂
hot CX  c18:   cx_fid × 6.0 + prop × 6.5             CX 双烂
hot CX  c60:   cx_fid × 5.7 + prop × 8.8             CX 双烂
hot CX  c66:   cx_fid × 8.6 + prop × 8.5             CX 双烂
```

### 6.4 4 级 Defect Overlay

| Level | qubit_frac | cx_frac | mult range | 总噪声预算 |
|-------|------------|---------|------------|-------------|
| **L1** | 0.04 | 0.03 | [2, 6] | 1× (基线) |
| **L2 (Willow)** | **0.07** | **0.05** | **[3, 8]** | **~2.5×** |
| **L3** (Sycamore) | 0.09 | 0.07 | [5, 12] | ~5× |
| **L4** (NISQ) | 0.12 | 0.10 | [7, 16] | ~8× |

**说明**：fraction 双轴递增（点数多）+ mult range 递增（单点更恶性）→ L4 总噪声预算 ≈ 8× L1。

### 6.5 d=3 floor 保护

`fraction × N` 经 round 取到 0 但 fraction > 0 时强制设 1。避免 d=3 patch 完全没坏点。

### 6.6 mult_max 距离自适应（`--mult-scale-with-d`）

```
mult_max(d) = max(5, 8 + (d - 5) × 2)
   d=3:  5 (floor)
   d=5:  8
   d=7: 12
   d=9: 16
```

物理动机：芯片有固定的"chip-pinned 缺陷"，大 patch 物理上更可能包含它们 → "包进来的最坏点"统计上更恶性。

### 6.7 跨脚本一致性

经典 + 泄漏两个脚本用同一 `--defect-seed` + 同一 canonical 参数顺序：

```
master_rng = default_rng(defect_seed)
hot_q_idx = master.choice(...)               ← 同一组 hot qubit
hot_cx_idx = master.choice(...)              ← 同一组 hot CX
sub_seeds_q = master.integers(...)            ← 每个 hot 点的独立 sub-seed

per-hot-point:
    sub = default_rng(sub_seed)
    n_pick = sub.integers(k_min, k_max+1)
    picks = sub.choice(canonical, n_pick)    ← 选哪些参数
    for p in canonical:
        if p in picks: mult = sub.uniform(min, max)  ← 倍数
```

经典脚本应用 `picks ∩ {sqg_fid_err, init, meas, spam, cx_fid_err}`；
泄漏脚本应用 `picks ∩ {lp, lp_propagation_prob}`。
两脚本独立 RNG advance，但 sub_seed 一致 → 同点同参数得到同 multiplier。

### 6.8 文件位置

- spec: 在 `level_params_spec.json` 和 `leakage_presets/leakage_LX.json` 各级的 `defect_overlay` 块（必须两文件保持一致）
- 实现：`gen_level_params.py` / `gen_leakage_overrides.py` 中的 `pick_hot_params()`

---

## 7. 4 层 × 4 级总览

每层独立 4 级，**完全可任意组合**（也可以单独关闭）：

```
经典噪声     :  L1   L2   L3   L4
                 ⊕    ⊕    ⊕    ⊕
串扰         :  X1   X2   X3   X4   或不启用
                 ⊕    ⊕    ⊕    ⊕
泄漏         :  L1   L2   L3   L4   或不启用
                 ⊕    ⊕    ⊕    ⊕
defect 坏点  :  按经典/泄漏的 level 决定，--defect-seed N 启用
```

**Willow grade 完整配置 = L2 + X2 + Leak2 + defect (任意 seed)**，对应 Google 2024 实测。

每层完全独立的 CLI 控制：
- 经典 level: `gen_level_params.py --level 2`
- 串扰 level: `gen_pair_overrides.py --crosstalk-config crosstalk_X2.json`
- 泄漏 level: `gen_leakage_overrides.py --leakage-config leakage_L2.json`
- 坏点: 任何脚本带 `--defect-seed N`（不带则禁用），`--defect-mult-min/max` 覆盖 spec

---

## 8. 随码距 (d) Scaling

| 层 | scaling 机制 | d=3 → d=5 | d=5 → d=7 | d=7 → d=9 |
|----|------|------|------|------|
| Layer 1 (经典) | 每 qubit / 每 CX 都独立采样 → 总噪声 ∝ N (d²) | n=17→49 | 49→97 | 97→177 |
| Layer 2 (串扰) | hop≥2 pair 数 ∝ d² + 频率 auto-scale 缩到 1 GHz → 大 d 撞车更密 | 同 chip 频率分布 | 接近撞车阈值 | 严重压缩 |
| Layer 3 (泄漏) | per-qubit lp / per-CX prop → 总泄漏池 ∝ d² | 自动 | 自动 | 自动 |
| Layer 4 (坏点) | fraction × N → hot 点数 ∝ d² | 1q+1cx → 3q+4cx | →7q+8cx | →12q+16cx |

**Layer 4 还额外有 mult_max(d) 自适应**：保证大 patch 包含的最坏点更恶性。

---

## 9. CLI 完整参考

### 9.1 `gen_level_params.py`

```bash
python gen_level_params.py \
    --level 2                    # L1/L2/L3/L4
    --distance 5 --rounds 30     # 表面码尺寸
    --basis z                    # z 或 x
    --seed 42                    # 经典 LogNormal 抽样 seed
    --defect-seed 7              # 坏点 seed (None = 不启用)
    --defect-q-fraction 0.07     # 覆盖 qubit_fraction
    --defect-cx-fraction 0.05    # 覆盖 cx_fraction
    --defect-mult-min 3 --defect-mult-max 8   # 覆盖 mult range
    --defect-multiplier 8        # 把 [min, max] 压成单值
    --out class.json
```

### 9.2 `gen_pair_overrides.py`

```bash
python gen_pair_overrides.py \
    --in class.json --merge \
    --crosstalk-config crosstalk_presets/crosstalk_X2.json \
    --out class_xtalk.json
```

或全 CLI（不用 preset）：

```bash
python gen_pair_overrides.py --in class.json --out X.json --merge \
    --mean-gap 5 --std-gap 3 --max-spread 1000 \
    --A 1e-3 --W 7 --zz-floor 4e-3 \
    --spatial-factors '{"2":1.0,"3":0.3,"4":0.1,"5":0.03}'
```

### 9.3 `gen_leakage_overrides.py`

```bash
python gen_leakage_overrides.py \
    --in class_xtalk.json \
    --leakage-config leakage_presets/leakage_L2.json \
    --seed 42                    # LogNormal lp/prop seed
    --defect-seed 7              # 同经典脚本的 defect-seed (强制一致)
    --defect-mult-min 3 --defect-mult-max 8
    --out final.json
```

### 9.4 `compare_corrmat_full.py`（高层 wrapper）

```bash
python compare_corrmat_full.py \
    --distance 5 --rounds 30 --shots 50000 \
    --patch q4_7                   # Willow patch ID
    --mult-scale-with-d            # 自动 mult_max(d)
    --classical-mult / --leakage-mult / --q-frac / --cx-frac / --mult-min / --mult-max
    --tag final_d5                 # 输出 PNG 后缀
```

输出 4 配置（pure / X2 / Leak2 / X2+Leak2）vs Real Willow 的 Spitz 矩阵 + 直方图 PNG + 数值统计。

---

## 10. 校准证据 (vs Google Willow)

最终标定 **C 配置 (mult_max scaling)** 在 d=5 q4_7 + d=7 q6_7 r=30 50K shots 上的残差：

| 指标 | d=5 (X2+Leak2) | d=7 (X2+Leak2) | 评价 |
|------|---------------|----------------|------|
| **defect_rate** | 6.77% vs 6.95% (-0.18pp) | 7.40% vs 7.95% (-0.55pp) | ≤ 1pp ✓ |
| **bulk mean** (Spitz hop≥2) | 0.00046 vs 0.00033 (1.39×) | 0.00036 vs 0.00030 (1.20×) | d=7 更好 |
| **max tail** | 0.044 vs 0.052 (0.85×) | 0.031 vs 0.068 (0.45×) | d=7 chip-pinned 限制 |
| **frac > 1e-3** | 8.5% vs 3.4% (2.5×) | 6.7% vs 3.8% (1.8×) | 可接受 |
| **视觉一致性** | 几乎一致 | 几乎一致 | ✓ |

### 渐进改进历程

| 阶段 | 关键改动 | d=5 defect_rate vs Real |
|------|----------|---------------------------|
| 初版 (uniform leak) | 全 chip 同 lp = 2e-4 | +2.33pp 过强 ✗ |
| Bug fix (lp = 4e-5) | Willow 实测对标 | +0.09pp ✓ |
| Rule 2 删除 | 去掉双重计数泄漏传播 | ~0pp ✓ |
| LogNormal 泄漏 | per-qubit lp 抖动 | -0.14pp ✓ |
| LogNormal 经典 | 5 个 error 参数 LogNormal | -0.20pp ✓ |
| L2 改良 | sqg/init/meas/spam 全部下调 | 拉出余量给坏点 |
| Defect overlay (D3) | uniform mult=8, 1q+2cx | -0.10pp ✓ |
| Heterogeneous (E2) | per-param subset + mult [3,8] | -0.18pp ✓ |
| **C scaling (final)** | **+ mult_max(d) auto** | **d=5 -0.18pp / d=7 -0.55pp** ✓ |

---

## 11. 已知模型局限

1. **d=7 max tail 偏低 (0.45× Real)**：Real Willow 的 0.068 max 来自 chip 上特定的"邻居坏对"。PAEMS 随机选 hot 点，无法显式建模"固定 chip-pinned 邻居对"。
2. **bulk mean 偏高 1.2-1.4×**：异质化坏点机制让多个点同时存在 → 总噪声预算大于单点全恶化模型。E2 是最佳折中。
3. **Layer 4 与 chip 物理位置无关**：每次 random 选 hot 索引，不能"固定 q34 永远是坏点"。要建这种行为需新机制（per-chip 缺陷文件）。
4. **测量串扰未建模**：不同 ancilla 的 readout 之间的 induced flip。可能在 d≥7 上贡献 ~0.3pp defect rate。
5. **4 级 mult_max 的物理上限**：lp.max=5e-3 / prop.max=0.8 是物理 sanity bound——hot 点 multiplier ×12 时 prop 已撞顶，进一步加 mult 不增 tail。
6. **频率分配在大 d 极端压缩**：mean_gap < 1 MHz at d>15，物理上对，但仿真 LER 会爆炸。
7. **DEM corr 不可用**：当前 PAEMS 数据用于解码时，PyMatching `enable_correlations=True` 不能直接套（DEM 不含 instruction grouping）。要做 corr 需要 graph-only 2-pass。

---

## 12. 推荐用法（常见 workflow）

### 12.1 完整 Willow grade 仿真 (L2+X2+Leak2+defect)

```bash
# d=5 r=30, 100K shots
python compare_corrmat_full.py --distance 5 --rounds 30 --shots 100000 \
    --mult-scale-with-d --tag willow_d5

# d=7 r=30, 100K shots (mult_max 自动 = 12)
python compare_corrmat_full.py --distance 7 --rounds 30 --patch q6_7 \
    --shots 100000 --mult-scale-with-d --tag willow_d7
```

### 12.2 干净模型（无串扰、无泄漏、无坏点）

```bash
python gen_level_params.py --level 2 --distance 5 --rounds 30 \
    --seed 42 --out clean.json
# 不带 --defect-seed
# 不接 gen_pair_overrides 和 gen_leakage_overrides
```

### 12.3 单独跑 4 级横扫

```bash
for L in 1 2 3 4; do
    for X in 1 2 3 4; do
        # 每个 (L, X) 配置 ...
    done
done
```

### 12.4 自定义坏点强度（不用 spec 默认）

```bash
python compare_corrmat_full.py --distance 5 --rounds 30 --shots 50000 \
    --q-frac 0.05 --cx-frac 0.03 --mult-min 3 --mult-max 10 --tag custom
```

### 12.5 关闭某层

| 关闭 | 方法 |
|------|------|
| 串扰 | 不调 `gen_pair_overrides.py`，或 `crosstalk_pairs={}` |
| 泄漏 | 不调 `gen_leakage_overrides.py`，或所有 lp/sp/prop = 0 |
| 坏点 | 任何 gen 脚本不带 `--defect-seed` |

---

## 13. 文件清单

```
paems_qubit_noise_tiers/
├── PAEMS_NOISE_MODEL.md          ← 本文档（master spec，覆盖全部 4 层）
├── CROSSTALK_MODEL.md            ← Layer 2 算法细节（频率分配 / density 实验）
├── LEVELS.md                     ← 早期 4-level spec 笔记
│
├── level_params_spec.json        ← Layer 1 + Layer 4 spec (L1-L4)
├── crosstalk_presets/
│   ├── crosstalk_X1.json         ← Layer 2 (4 级)
│   ├── crosstalk_X2.json
│   ├── crosstalk_X3.json
│   └── crosstalk_X4.json
├── leakage_presets/
│   ├── leakage_L1.json           ← Layer 3 + Layer 4 (4 级)
│   ├── leakage_L2.json
│   ├── leakage_L3.json
│   └── leakage_L4.json
│
├── gen_level_params.py           ← Layer 1 + Layer 4 生成器
├── gen_pair_overrides.py         ← Layer 2 生成器
├── gen_leakage_overrides.py      ← Layer 3 + Layer 4 生成器
│
├── compare_corrmat_full.py       ← 完整 4 配置对比 vs Real Willow
├── compare_corrmat_xtalk_levels.py  ← X1-X4 横扫对比
├── compare_corrmat_diff.py       ← differential 矩阵
└── chip_calibration_data/        ← IBM/Google/USTC 真机校准参考
```

PAEMS 主代码：

```
C:\PAEMS-Precise_and_Adaptive_Error_Model_for_superconducting_Quantum_Processors\
└── Surface_Code_Simulation/
    ├── inject_basic_noise.py                    ← Layer 1 + Layer 2 注入
    ├── inject_leakage_noise_vectorized.py       ← Layer 3 模拟器
    ├── run_sampling.py                          ← 主 sampling driver
    ├── surface_code_generate_circuits.py        ← 表面码电路生成
    └── calculate.py                             ← 噪声系数转换
```

---

## 14. 设计决策记录（关键 trade-off）

| 决策 | 选择 | 理由 |
|------|------|------|
| 4 层是否解耦？ | **完全解耦** | 物理上独立，组合灵活，调试方便 |
| 串扰是否跟 L1-L4 同级？ | **解耦 X1-X4** | crosstalk 是芯片 wiring/freq engineering，跟 T1/T2 正交 |
| error 参数用 Normal 还是 LogNormal？ | **LogNormal** | 错误率物理上右偏长尾，Normal 截断会把分布尾巴砍掉 |
| timing 参数用 LogNormal？ | **不用，Normal** | T1/T2 物理上对称分布 |
| 泄漏注入还是后处理？ | **后处理 (post-process flip)** | Stim 不直接支持 \|2⟩ 态；通过 detection event flip 等价 |
| Rule 2 (CX-affected) 保留？ | **删除** | 与 lp_propagation_prob 双重计数 |
| flip_prob 多少？ | **0.5 固定** | 测到 \|2⟩ 时输出随机 |
| 坏点机制：每点全恶化 vs 异质化？ | **异质化** | 物理上不同 qubit 缺陷类型不同 |
| 坏点 multiplier 范围 vs 单值？ | **uniform [min, max]** | 倍数也物理变化 |
| mult_max 跟 d 走？ | **是** | 大 patch 包含的最坏点更恶性（chip-pinned 模型） |
| 4 级阶梯怎么递增？ | **fraction + mult 双轴** | 物理上"频率 + 强度"双因素 |

---

## 15. 进一步阅读

- **Layer 2 算法细节**: `CROSSTALK_MODEL.md`（频率分配 cumsum + bipartite，χ density 实验，4 级 preset 调校历史）
- **早期 LEVELS 笔记**: `LEVELS.md`
- **Willow 数据集**: `chip_calibration_data/`
- **PAEMS 主代码**: `C:\PAEMS-Precise_and_Adaptive_Error_Model_for_superconducting_Quantum_Processors\`
- **Spitz 公式**: P. Spitz et al., Adv. Quantum Technol. 2018
- **Willow paper**: Google Nature 2024
