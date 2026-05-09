#!/usr/bin/env python3
"""
测试：把 L2 的 T1/T2 替换成 Willow 真值（68/89 us），其它 L2 参数不变。
看 defect 密度是否更贴近真机 Willow（5.79% @ d=3 r=10）。

输出三个对比点：
  - L2 原版（T1=270, T2=200）
  - L2 + Willow T1/T2（T1=68, T2=89）
  - Real Willow Sycamore d=3 r=10 (从你的 dataset 直接读)
"""
import json
import sys
import os
from pathlib import Path
import numpy as np

PAEMS_SC = r"C:\PAEMS-Precise_and_Adaptive_Error_Model_for_superconducting_Quantum_Processors\Surface_Code_Simulation"
sys.path.insert(0, PAEMS_SC)
from inject_basic_noise import inject_surface_code_noise
from surface_code_generate_circuits import generate_surface_code_circuit

import stim

HERE = Path(__file__).parent
SPEC = HERE / "level_params_spec.json"


def make_modified_spec(t1_mean, t1_std, t2_mean, t2_std, base_level='L2'):
    """复制 base_level，只改 T1/T2。"""
    with open(SPEC) as f:
        spec = json.load(f)
    new = json.loads(json.dumps(spec[base_level]))  # deep copy
    new['qubit']['t1']['mean'] = t1_mean
    new['qubit']['t1']['std'] = t1_std
    new['qubit']['t2']['mean'] = t2_mean
    new['qubit']['t2']['std'] = t2_std
    new['label'] = f"{base_level} + Willow T1/T2"
    return new


def gen_params_json(level_spec, distance, rounds, basis, seed, out_path):
    """复用 gen_level_params 里的逻辑，单独跑一份。"""
    sys.path.insert(0, str(HERE))
    from gen_level_params import sample_qubit, sample_cx, get_topology_from_paems

    topo = get_topology_from_paems(distance, rounds, basis)
    all_q, cx_pairs, data_q, x_stab, z_stab = topo

    rng = np.random.default_rng(seed)
    qubits = sample_qubit(rng, level_spec['qubit'], n_qubits=len(all_q))
    cx_gates = sample_cx(rng, level_spec['cx'], cx_pairs)

    out = {
        "qubits": qubits,
        "cx_gates": cx_gates,
        "_metadata": {
            "label": level_spec['label'],
            "distance": distance, "rounds": rounds, "basis": basis, "seed": seed,
        },
    }
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)


def run_defect(distance, rounds, shots, params_file, basis='z'):
    circuit, data_q, x_stab, z_stab, cx_gates = generate_surface_code_circuit(
        distance, rounds, basis)
    nc = inject_surface_code_noise(circuit, data_q, x_stab, z_stab, cx_gates,
                                   params_file)
    dets = nc.compile_detector_sampler().sample(shots=shots)
    return dets


def real_willow_defect(d, r, basis='Z'):
    """读真机 Willow 数据。"""
    base = (r"C:\Users\10124\Desktop\google dataset\google_105Q_surface_code_d3_d5_d7"
            rf"\d{d}_at_q4_5\{basis}\r{r}")
    cir = stim.Circuit.from_file(base + r"\circuit_noisy_si1000.stim")
    dets = stim.read_shot_data_file(
        path=base + r"\detection_events.b8", format='b8',
        num_detectors=cir.num_detectors, bit_packed=False)
    return dets


def main():
    # 配置
    DISTANCE = 3
    ROUNDS = [3, 10]   # 都跑，对比
    SHOTS = 50000
    SEED = 42

    print(f"d={DISTANCE} basis=z  shots={SHOTS}  seed={SEED}\n")

    # 准备两个 spec
    with open(SPEC) as f:
        spec_full = json.load(f)
    L2_orig = spec_full['L2']
    L2_willow = make_modified_spec(
        t1_mean=68e-6, t1_std=13e-6,
        t2_mean=89e-6, t2_std=15e-6,
    )

    print('Spec diffs:')
    print(f"  L2 original :  T1={L2_orig['qubit']['t1']['mean']*1e6:.0f}+/-{L2_orig['qubit']['t1']['std']*1e6:.0f} us, "
          f"T2={L2_orig['qubit']['t2']['mean']*1e6:.0f}+/-{L2_orig['qubit']['t2']['std']*1e6:.0f} us")
    print(f"  L2 + Willow :  T1={L2_willow['qubit']['t1']['mean']*1e6:.0f}+/-{L2_willow['qubit']['t1']['std']*1e6:.0f} us, "
          f"T2={L2_willow['qubit']['t2']['mean']*1e6:.0f}+/-{L2_willow['qubit']['t2']['std']*1e6:.0f} us")
    print()

    rows = []
    for R in ROUNDS:
        # L2 原版（直接用现有 sample 文件）
        sample_file = HERE / 'param_examples' / f'd{DISTANCE}_level2.json'
        if R == 3:
            l2_orig_file = str(sample_file)
        else:
            # r=10 重新生成
            l2_orig_file = str(HERE / f'.tmp_l2_r{R}.json')
            gen_params_json(L2_orig, DISTANCE, R, 'z', SEED, l2_orig_file)

        # L2 + Willow T1/T2
        l2_willow_file = str(HERE / f'.tmp_l2willow_r{R}.json')
        gen_params_json(L2_willow, DISTANCE, R, 'z', SEED, l2_willow_file)

        # 跑两个 + 真机
        d_orig = run_defect(DISTANCE, R, SHOTS, l2_orig_file)
        d_will = run_defect(DISTANCE, R, SHOTS, l2_willow_file)
        try:
            d_real = real_willow_defect(DISTANCE, R)
            real_density = float(d_real.mean())
            real_perdef = float(d_real.sum(axis=1).mean())
            real_dets = d_real.shape[1]
        except (ValueError, FileNotFoundError):
            real_density = float('nan')
            real_perdef = float('nan')
            real_dets = d_orig.shape[1]

        rows.append({
            'r': R,
            'l2_orig': float(d_orig.mean()),
            'l2_willow': float(d_will.mean()),
            'real_willow': real_density,
            'l2_orig_perdef_per_shot': float(d_orig.sum(axis=1).mean()),
            'l2_willow_perdef_per_shot': float(d_will.sum(axis=1).mean()),
            'real_willow_perdef_per_shot': real_perdef,
            'detectors': real_dets,
        })

    # 汇总
    print(f"{'rounds':<8} {'detectors':>10} {'L2 orig':>12} {'L2+Willow T1T2':>16} {'Real Willow':>14} {'Real-L2orig gap':>16}")
    print('-' * 90)
    for r in rows:
        gap = (r['real_willow'] - r['l2_orig']) * 100
        print(f"r={r['r']:<6} {r['detectors']:>10} "
              f"{r['l2_orig']*100:>11.3f}% {r['l2_willow']*100:>15.3f}% "
              f"{r['real_willow']*100:>13.3f}% {gap:>15.3f}pp")

    print()
    print('每 shot 平均 defect 数:')
    print(f"{'rounds':<8} {'L2 orig':>12} {'L2+Willow':>12} {'Real':>12}")
    for r in rows:
        print(f"r={r['r']:<6} {r['l2_orig_perdef_per_shot']:>12.2f} "
              f"{r['l2_willow_perdef_per_shot']:>12.2f} "
              f"{r['real_willow_perdef_per_shot']:>12.2f}")


if __name__ == '__main__':
    main()
