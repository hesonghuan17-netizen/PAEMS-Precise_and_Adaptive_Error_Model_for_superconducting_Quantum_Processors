# PAEMS Qubit Noise Tier Specification

> 4 级合成噪声参数，用于 PAEMS 表面码仿真的"控制变量"实验。
> 当前版本：**v1.0** (2026-04-27)，**纯经典噪声**（lp/sp/lp_propagation_prob 设为 0）。

## 目录文件

```
paems_qubit_noise_tiers/
├── LEVELS.md                       # 本文件
├── level_params_spec.json          # 4 levels × 13 params 的 (μ, σ, bounds) 规范
├── gen_level_params.py             # 生成器
├── samples/                        # 4 个 d=3 r=3 测试样本
│   ├── d3_level1.json
│   ├── d3_level2.json
│   ├── d3_level3.json
│   └── d3_level4.json
└── chip_calibration_data/          # 真实硬件数据（4 级参数的来源）
    ├── ibm_*_calibrations_*.csv    # 10 台 IBM (Heron r1/r2/r3, Eagle r3)
    └── origin_wukong180.xlsx       # 本源悟空 180 (169 比特实测)
```

## 4 级定义

| Level | 含义 | 数据锚点 |
|---|---|---|
| **L1** | 下一代（2030s，比 L2 优 2-3×） | 推断，行业 roadmap |
| **L2** | 当前 state-of-art (2024-2025) | IBM Heron r3 (boston/pittsburgh) + Google Willow + USTC Zuchongzhi 3.x |
| **L3** | 当前主流 (2023-2024) | IBM Heron r2 (kingston/marrakesh/fez) + Heron r1 + Zuchongzhi 3.0 |
| **L4** | 当前较差 / 旧代际 | IBM Eagle r3 (brussels/strasbourg) + Origin Wukong-180 + Sycamore-era |

## 完整参数表

| Param | L1 | L2 | L3 | L4 |
|---|---|---|---|---|
| **t1** (µs) | 600 ± 150 | 270 ± 70 | 100 ± 30 | 30 ± 10 |
| **t2** (µs) | 500 ± 120 | 200 ± 60 | 70 ± 25 | 20 ± 8 |
| **sqg_fid** | 0.9999 ± 5e-5 | 0.9997 ± 1e-4 | 0.999 ± 4e-4 | 0.998 ± 8e-4 |
| **sqg_length** (ns) | 25 ± 0.3 | 30 ± 1 | 35 ± 2 | 60 ± 3 |
| **rd_length** (ns) | 800 ± 80 | 1500 ± 200 | 2500 ± 300 | 3000 ± 500 |
| **data_init_error** | 8e-4 ± 3e-4 | 2e-3 ± 8e-4 | 1e-2 ± 4e-3 | 3e-2 ± 1e-2 |
| **data_measurement_error** | 2e-3 ± 8e-4 | 5e-3 ± 2e-3 | 2e-2 ± 8e-3 | 6e-2 ± 2e-2 |
| **measurement_spam_rate** | 3e-3 ± 1e-3 | 7e-3 ± 3e-3 | 3e-2 ± 1e-2 | 9e-2 ± 3e-2 |
| **cx_fid** | 0.999 ± 5e-4 | 0.997 ± 1e-3 | 0.99 ± 3e-3 | 0.98 ± 8e-3 |
| **cx_length** (ns) | 30 ± 2 | 50 ± 10 | 200 ± 20 | 400 ± 50 |
| **lp / sp / lp_propagation_prob** | 0 | 0 | 0 | 0 |

## 阶梯关系（每级劣化倍数）

| Param | L1→L2 | L2→L3 | L3→L4 |
|---|---|---|---|
| t1 | 2.2× | 2.7× | 3.3× |
| t2 | 2.5× | 2.9× | 3.5× |
| sqg 错率 | 3× | 3.3× | 2× |
| meas_spam | 2.3× | 4.3× | 3× |
| cx 错率 | 3× | 3.3× | 2× |
| cx_length | 1.7× | 4× | 2× |

→ **L1 ↔ L4 总跨度 ≈ 25-100×**，跨整个量子产业从未来到旧代。

## 物理约束（生成器自动保证）

1. 所有时间参数 > 0（min floor）
2. 所有概率参数 ∈ [0, 1]
3. 所有 fidelity ≤ 1（通过对错率采样 + 转换实现）
4. **T2 ≤ 2 × T1**（物理硬约束，自动 clip）
5. **`measurement_spam_rate` ≈ `data_init_error` + `data_measurement_error`**（μ 满足，per-qubit 松弛）

## 噪声约束语义

`measurement_spam_rate` 在 PAEMS 里是 ancilla 在 MR 前的 X_ERROR 概率，物理含义：
- **轮 1**：ancilla 初始制备失败 + 第 1 轮 M 误读
- **轮 k>1**：第 (k-1) 轮 R 失败 + 第 k 轮 M 误读

→ 与 `data_init_error`(=R 性质) + `data_measurement_error`(=M 性质) **同物理来源**，
所以 mean 上满足近似加和关系。

## 数据来源溯源

### IBM 实测（10 台）
- Heron r3：boston / pittsburgh / aachen → L2 主要锚点
- Heron r2：kingston / marrakesh / fez → L3 主要锚点
- Heron r1：berlin / miami → L3 辅助
- Eagle r3：brussels / strasbourg → L4 辅助（cx_length 660ns 主要来自这里）

### 学术 / 国际 SOTA
- **Google Willow** (2024-12, 105 qubit)：T1 68 µs, T2 89 µs, CZ err 0.33%, single 0.035% → L2
- **USTC Zuchongzhi 3.0** (2024)：T1 72, sqg fid 99.90%, CZ fid 99.62% → L2
- **USTC Zuchongzhi 3.2** (2025-12-22, PRL)：107 qubit, readout err 0.95%, leakage 6.4e-4 → L2 + 未来 leakage 锚点
- **Google Sycamore** (2019)：T1 ~20, CZ err 0.6% → L4 历史

### 国内实测
- **Origin Wukong-180** (2026, 169 qubit)：T1 33±15 µs, T2,Echo 18.7±9.2, sqg fid 99.86%, CZ fid 98.23%, readout err 5.9% → **L4 主要锚点**

## 使用方法

### 生成单个 JSON
```bash
python gen_level_params.py --level 2 --distance 7 --rounds 7 --seed 42 \
    --out my_d7_level2.json
```

参数：
- `--level`: 1, 2, 3, 4
- `--distance`: 表面码距离（PAEMS 用 3-13 居多）
- `--rounds`: 纠错轮数
- `--basis`: z / x（默认 z）
- `--seed`: 随机种子（同 seed 给出同样结果）
- `--out`: 输出 JSON 路径

### 输出 JSON 直接喂给 PAEMS
```python
from Surface_Code_Simulation.run_sampling import run_sampling
results = run_sampling(distance=7, rounds=7, shots=1000,
                      params_file='my_d7_level2.json',
                      include_leakage=False, basis='z')
```

### 跨 4 级对比实验
```bash
for L in 1 2 3 4; do
    python gen_level_params.py --level $L --distance 7 --rounds 7 --seed 42 \
        --out d7_L${L}.json
    # 然后跑 PAEMS / PyMatching / 你的 V5 RTL
done
```

## 局限 + TODO

| 问题 | 状态 |
|---|---|
| 泄漏参数 lp / sp / lp_propagation_prob 没分级 | ⚠️ v1.0 设为 0，v2 加 |
| 跨参数相关性（T1 长 → T2 也长）未建模 | ⚠️ 现 i.i.d. 采样，未来加协方差 |
| L1 是人工外推，无真实数据 | 期望 2030s 验证 |
| Eagle 的极端 cx_length 660ns 没完全反映在 L4 | L4 用了 400ns（折中），如要"古老"L4 改成 660 |
| Wukong 的 readout 不对称（F0≠F1）未建模 | 当前用对称 SPAM |

## 版本历史

- **v1.0** (2026-04-27): 初版，4 级 × 10 经典噪声参数，纯经典（无 leakage）
