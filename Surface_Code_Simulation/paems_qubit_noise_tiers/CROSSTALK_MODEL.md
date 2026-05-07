# Crosstalk Modeling — 设计文档

> 解释 PAEMS 串扰噪声建模的整体架构、χ 公式、4 级 preset 和实现细节。
> 用途：日后 review 代码、实验复现、调参参考。
> 最后更新：2026-04-29

---

## 1. 总体架构

PAEMS 的噪声系统**分两个独立轴**：

```
┌──────────────────────────┐         ┌──────────────────────────┐
│  Classical noise (L1-L4) │         │  Crosstalk (X1-X4)       │
│  T1, T2, fid, SPAM, ...  │  ⊥      │  freq + hop based χ      │
│  gen_level_params.py     │         │  gen_pair_overrides.py   │
│  level_params_spec.json  │         │  crosstalk_presets/*.json│
└──────────────────────────┘         └──────────────────────────┘
            ↓                                      ↓
         classical.json     ──merge──>      classical+crosstalk.json
                                                    ↓
                                           PAEMS run_sampling()
```

**关键设计选择**：
- 经典噪声（L1-L4）和串扰（X1-X4）**完全解耦**
- 任意组合：4 classical × 4 crosstalk = 16 实验配置
- 跑 `gen_pair_overrides.py` 不带任何参数 → 输出空 `crosstalk_pairs` → 无串扰

---

## 2. 整体流程

### Step 1：生成纯 classical noise JSON

```bash
python gen_level_params.py --level 2 --distance 5 --rounds 5 --out class.json
```

输出：仅含 T1/T2/fid/SPAM 等 per-qubit 参数 + cx_gates 拓扑。
**不含** `frequency_mhz`，**不含** `crosstalk_pairs`。

### Step 2（可选）：叠加串扰

```bash
python gen_pair_overrides.py --in class.json --out final.json --merge \
    --crosstalk-config crosstalk_presets/crosstalk_X3.json
```

或全 CLI：
```bash
python gen_pair_overrides.py --in class.json --out final.json --merge \
    --mean-gap 5 --std-gap 3 --max-spread 1000 \
    --A 1e-3 --W 7 --zz-floor 4e-3 \
    --spatial-factors '{"2":1.0,"3":0.3,"4":0.1,"5":0.03}'
```

输出：原 JSON + 每个 qubit 加 `frequency_mhz` + `crosstalk_pairs` dict。

### Step 3：注入 PAEMS

`Surface_Code_Simulation/inject_basic_noise.py` 读 `crosstalk_pairs`，对每个 H/CX 后追加 `DEPOLARIZE1` spectator 事件。Stim 自动 propagate Pauli 错误，DEM 自动捕获。

---

## 3. 频率分配算法（cumsum + bipartite split + auto-scale）

### 3.1 步骤

```python
# Step A: 拓扑 bipartite 着色
G = networkx.Graph(cx_gates as edges)
color = nx.bipartite.color(G)        # 2-coloring (surface code 永远 bipartite)
color_0 = [q for q if color[q] == 0]
color_1 = [q for q if color[q] == 1]

# Step B: 采样 n-1 个 Gaussian gaps
gaps = rng.normal(mean_gap, std_gap, n-1)
gaps = np.clip(gaps, 0.1, None)      # 避免零/负 gap

# Step C: 总跨度超 max_spread 时等比缩小
if sum(gaps) > max_total_spread:
    gaps *= max_total_spread / sum(gaps)

# Step D: cumsum 得位置，居中到 chip 中心频率
positions = np.concatenate(([0], np.cumsum(gaps))) - mean + center_freq

# Step E: bipartite split
#   color 0 qubits → 低半段位置（0 ~ n_color_0)
#   color 1 qubits → 高半段位置（n_color_0 ~ n)
#   每个色组内随机洗牌
rng.shuffle(color_0)
rng.shuffle(color_1)
for i, q in enumerate(color_0): freqs[q] = positions[i]
for i, q in enumerate(color_1): freqs[q] = positions[n_color_0 + i]
```

### 3.2 物理意义

- **mean_gap, std_gap**：相邻比特频率间隔的分布（正态）
- **max_spread**：transmon 物理频率范围限制（默认 1 GHz）
- **bipartite split**：保证 1-hop CX 邻居的频率差 ≥ 半跨度（避开 drive crosstalk on CX gates）
- **cumsum**：让"频率间隔"严格服从 Gaussian，不是绝对频率服从 Gaussian

### 3.3 大 d scaling

| d | n_qubits | 自然 spread (mean_gap=15) | 是否触发 auto-scale |
|---|---|---|---|
| 3 | 17 | 240 MHz | ✗ |
| 5 | 49 | 720 MHz | ✗ |
| 7 | 97 | 1440 MHz | ✓ 缩到 1 GHz |
| 11 | 241 | 3600 MHz | ✓ 严重压缩 |
| 21 | 841 | 12.6 GHz | ✓ 极端压缩，撞车很多 |

→ **大 d 时撞车对自然增多**（频率轴变拥挤），跟真机一致（实际大 chip 频率更密集）。

---

## 4. χ 计算公式

### 4.1 Unified formula

```
χ(i, j) = A × max(Lorentzian(Δf, W), zz_floor) × spatial_factor(hop)

Lorentzian(Δf, W) = 1 / (1 + (Δf/W)²)
spatial_factor(hop) ∈ {1: skip, 2: 1.0, 3: 0.3, 4: 0.1, 5: 0.03, ≥6: 0}
```

### 4.2 三个通道的物理意义

1. **A × Lorentzian**：drive crosstalk（频率近时驱动微波泄漏）
2. **A × zz_floor**：always-on ZZ 残留（远 Δf 时仍存在的"地板"）
3. **× spatial_factor**：物理距离衰减（共享 bus / 控制线）

### 4.3 为什么 max(Lorentzian, zz_floor)

- 频率近 → Lorentzian 大 → drive 主导
- 频率远 → Lorentzian 趋零 → zz_floor 接管
- max() 取较大值 → 物理上"两路噪声并行，强者胜"

### 4.4 spatial_factor 跨级共用

理由：芯片 wiring 拓扑决定 spatial 衰减形状，跟工艺质量（A, zz_floor）正交。

| hop | factor | 物理对应 |
|---|---|---|
| 1 | skip | CX 直邻（被 CX 门噪声覆盖） |
| 2 | 1.0 | 同 ancilla 的两个 data |
| 3 | 0.3 | 隔 2 个 qubit |
| 4 | 0.1 | 隔 3 个 qubit |
| 5 | 0.03 | 隔 4 个 qubit |
| ≥6 | 0 | 物理上 ZZ 可忽略 |

### 4.5 1-hop 永远 skip

`gen_pair_overrides.py` 的 `generate_pairs()` 在 `hop == 1` 时 `continue`。
**理由**：CX 门噪声（DEPOLARIZE2 + PAULI_CHANNEL_1）已含 1-hop 串扰贡献，重复算会过估。

---

## 5. 4 级 Preset 数值（最终调校）

```
                   X1         X2         X3         X4
                   future     mild       moderate   worst
═══════════════════════════════════════════════════════════
mean_gap_mhz       15         15         5          5
std_gap_mhz        5          5          3          3
max_spread_mhz     1000       1000       1000       1000
A_peak             5e-4       8e-4       1e-3       2.5e-3
W_mhz              3          5          7          10
zz_floor           2e-3       4e-3       4e-3       8e-3
spatial_factor     {hop2: 1.0, hop3: 0.3, hop4: 0.1, hop5: 0.03}
═══════════════════════════════════════════════════════════
```

### 实测 density 阶梯（d=5 r=5, 200K shots, L2 classical baseline）

| Preset | density | 增量 vs baseline (5.98%) | ratio vs prev |
|---|---|---|---|
| baseline | 5.98% | — | — |
| **X1** | 6.01% | **+0.030pp** | — |
| **X2** | 6.10% | **+0.119pp** | 4.0× |
| **X3** | 7.01% | **+1.036pp** | 8.7× |
| **X4** | 9.65% | **+3.669pp** | 3.5× |

→ 总跨度 **~120×** density 增量。X2→X3 跳幅略大（8.7×），由阈值效应导致（χ 跨过 gate_err 量级时 density 突然显著），物理上合理。

---

## 6. χ 分布特征（Pareto-like）

### 6.1 全 pair 分布

| Level | N pairs | max/median ratio | top 1% χ 占总和 |
|---|---|---|---|
| X1 | 674 | 133× | 31.7% |
| X2 | 674 | 161× | 32.8% |
| X3 | 674 | 710× | 25.3% |
| X4 | 674 | 366× | 19.0% |

→ **极少数 pair 比中位数大 100-700 倍**，**top 1% 贡献 20-33% 总 χ**。
跟真机 (IBM/Google) 实测一致：少数撞车对主导，大量背景对贡献小。

### 6.2 Per-qubit hotspot（X3 例子）

某些 qubit 有特别多/特别强的 crosstalk pair：

```
qubit   total_chi   n_pairs   max_pair_chi
q22     2.07e-03      40      9.16e-04   ← hotspot
q44     2.02e-03      30      7.49e-04
q13     1.88e-03      31      8.93e-04
...
avg q   7.83e-04      27.5
```

Hotspot 比平均高 **~2.5×**。物理对应：要么频率轴上的"边缘"qubit，要么参与了 severe collision。

---

## 7. 实现文件 + 关键函数

```
paems_qubit_noise_tiers/
├── gen_level_params.py             # 仅 classical noise（L1-L4）
├── level_params_spec.json          # L1-L4 spec
├── gen_pair_overrides.py           # crosstalk overlay
├── crosstalk_presets/
│   ├── crosstalk_X1.json           # future
│   ├── crosstalk_X2.json           # mild
│   ├── crosstalk_X3.json           # moderate
│   └── crosstalk_X4.json           # worst
├── chip_calibration_data/          # 真实硬件参考数据
└── CROSSTALK_MODEL.md              # 本文档
```

### 关键函数

`gen_pair_overrides.py`:
- `assign_frequencies_cumsum(qubits, cx_gates, mean_gap, std_gap, max_spread, center, rng)` — bipartite-aware Gaussian gap 采样
- `chi_unified(diff, hop, cfg)` — 单 pair χ 计算
- `generate_pairs(qubits, cx_gates, cfg)` — 全 pair sweep + max combine

`Surface_Code_Simulation/inject_basic_noise.py` (PAEMS):
- `build_crosstalk_lookup(chi_pairs)` — 把 pair dict 翻成 per-active-qubit lookup
- `add_spectator_crosstalk(circuit, active_q, pair_lookup)` — H/CX 后注入 spectator DEPOLARIZE1
- 注入点在 `inject_surface_code_noise()` 的 `H` 和 `CX` 分支

---

## 8. 设计决策记录

| 决策 | 选择 | 理由 |
|---|---|---|
| 串扰要不要跟 classical noise 一起分级？ | **解耦** | 物理上 crosstalk 是 chip wiring/freq engineering，跟 T1/T2 等正交 |
| 是否需要 chi_global default？ | **删除** | 所有串扰**只**通过 `crosstalk_pairs` 显式声明 |
| 频率分配用 tile 还是 cumsum？ | **cumsum + bipartite split** | 更"自然"的 Gaussian gap 分布，bipartite 保护 1-hop |
| 双通道（drive+ZZ 分开）还是合并？ | **合并到 unified χ formula** | 物理上 drive crosstalk 也有空间衰减；公式简洁 |
| 远 Δf 是否有 ZZ 残留？ | **加 zz_floor** | 用 max(Lorentzian, zz_floor) 兜底 |
| 大 d 频率溢出怎么办？ | **auto-scale 等比压缩** | 自然反映"大 chip 撞车多"的物理事实 |

---

## 9. 已知局限

1. **2-hop 同色 pair 仍可能 freq 接近**：bipartite split 只保护 1-hop。同色 cumsum 内随机分布。
   → 真机靠更精细的 graph-distance-aware 频率分配，PAEMS 当前不实现。

2. **spatial_factor 是离散查表**，没有连续物理依据。
   → 如果想要更细致，可改成 `1/(1+(hop-1)^α)` 之类。

3. **大 d (>15) 频率压缩极端**：mean_gap < 1 MHz，几乎所有对都撞车。
   → 物理上对的（大 chip 真做不到稀疏频率），但仿真 LER 会爆炸。

4. **A、zz_floor 跟 density 增量是非线性**：阈值效应导致 X2→X3 跳幅难压到 5×。
   → 调参时需迭代实验，不是公式可推导。

---

## 10. 跟真实 chip 数据对照

| 维度 | 当前模型 | Sycamore (Google) | Wukong (Origin) |
|---|---|---|---|
| 频率范围 | 1 GHz (max_spread) | 580 MHz | 929 MHz |
| 跨级阶梯 | A: 0.5e-3 ~ 2.5e-3 | 实测无标 | 估 ~1e-3 |
| 撞车对集中度 | top 1% 贡献 20-33% | 类似 (Pareto) | 更严重 (gap min 0.012 MHz) |
| 频率工程精度 | std_gap 3-5 MHz | std ~10 MHz (image) | std 5 MHz |

**当前模型贴近真机量纲**，是合理的研究 baseline。

---

## 11. 使用建议

```bash
# 16 配置全 sweep（4 classical × 4 crosstalk）
for L in 1 2 3 4; do
  for X in 1 2 3 4; do
    python gen_level_params.py --level $L --distance 5 --rounds 5 \
        --seed 42 --out classical_L${L}.json
    python gen_pair_overrides.py --in classical_L${L}.json \
        --out L${L}_X${X}.json --merge \
        --crosstalk-config crosstalk_presets/crosstalk_X${X}.json
  done
done
# 喂给 PAEMS run_sampling() + PyMatching 解码 → LER 网格
```

研究方向举例：
- 固定 X 扫 L → 看 classical noise 对 LER 的影响
- 固定 L 扫 X → 看 crosstalk 对 LER 的影响
- 极端组合 L4+X4 → 解码失败边界
- 极端组合 L1+X1 → 看下界

---

## 12. 修订历史

- **2026-04-29 v1**: 初版，4 级 preset 完成，bipartite split 实装
