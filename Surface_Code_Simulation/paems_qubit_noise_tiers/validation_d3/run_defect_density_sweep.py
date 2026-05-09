#!/usr/bin/env python3
"""
跑 PAEMS run_sampling 4 个 level，统计 defect (detection event) 密度。
不解码，只看 syndrome 分布。

输出：
  - syndrome_data/d{d}_r{r}_level{N}_shots{K}.npy   (每个 level 一份 syndrome 矩阵)
  - 终端打印 defect 密度对比表
"""
import argparse
import os
import sys
import json
import numpy as np
from pathlib import Path

# 把 PAEMS Surface_Code_Simulation 加进 path
PAEMS_SC = r"C:\PAEMS-Precise_and_Adaptive_Error_Model_for_superconducting_Quantum_Processors\Surface_Code_Simulation"
if PAEMS_SC not in sys.path:
    sys.path.insert(0, PAEMS_SC)

from inject_basic_noise import inject_surface_code_noise
from surface_code_generate_circuits import generate_surface_code_circuit


def run_one_level(distance, rounds, shots, params_file, basis='z'):
    """生成噪声电路 + 用 detector_sampler 直接拿 detection events。"""
    circuit, data_q, x_stab, z_stab, cx_gates = generate_surface_code_circuit(
        distance, rounds, basis
    )
    noisy_circuit = inject_surface_code_noise(
        circuit, data_q, x_stab, z_stab, cx_gates, params_file
    )
    # detection events: (shots, num_detectors) bool
    detector_sampler = noisy_circuit.compile_detector_sampler()
    dets = detector_sampler.sample(shots=shots)
    return dets, noisy_circuit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--distance', type=int, default=3)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--shots', type=int, default=10000)
    ap.add_argument('--basis', default='z')
    ap.add_argument('--levels', nargs='+', type=int, default=[1, 2, 3, 4])
    ap.add_argument('--param_dir', default='param_examples')
    ap.add_argument('--out_dir', default='syndrome_data')
    args = ap.parse_args()

    here = Path(__file__).parent
    pdir = here / args.param_dir
    odir = here / args.out_dir
    odir.mkdir(parents=True, exist_ok=True)

    print(f'Sweep: d={args.distance} r={args.rounds} shots={args.shots} basis={args.basis}')
    print('=' * 80)

    rows = []
    for L in args.levels:
        param_file = pdir / f'd{args.distance}_level{L}.json'
        if not param_file.exists():
            print(f'[skip] L{L}: {param_file} not found')
            continue

        dets, nc = run_one_level(args.distance, args.rounds, args.shots,
                                 str(param_file), args.basis)
        n_det_total = dets.size
        n_det_fired = int(dets.sum())
        density = n_det_fired / n_det_total
        per_shot_count = dets.sum(axis=1)
        per_det_rate = dets.mean(axis=0)

        out_npy = odir / f'd{args.distance}_r{args.rounds}_level{L}_shots{args.shots}.npy'
        np.save(out_npy, dets.astype(np.uint8))

        rows.append({
            'level': L,
            'shape': dets.shape,
            'density': density,
            'mean_defects_per_shot': per_shot_count.mean(),
            'std_defects_per_shot': per_shot_count.std(),
            'max_per_det_rate': per_det_rate.max(),
            'min_per_det_rate': per_det_rate.min(),
            'out': out_npy.name,
        })

    hdr = ('Lvl', 'shape', 'density', 'defects/shot mean', 'std', 'per-det max', 'per-det min')
    print()
    print(f'{hdr[0]:<5} {hdr[1]:>15} {hdr[2]:>10} {hdr[3]:>18} {hdr[4]:>7} {hdr[5]:>12} {hdr[6]:>12}')
    print('-' * 90)
    for r in rows:
        print(f"L{r['level']:<4} {str(r['shape']):>15} "
              f"{r['density']*100:>9.3f}% "
              f"{r['mean_defects_per_shot']:>18.2f} "
              f"{r['std_defects_per_shot']:>7.2f} "
              f"{r['max_per_det_rate']*100:>11.3f}% "
              f"{r['min_per_det_rate']*100:>11.3f}%")
    print()
    print(f'Saved {len(rows)} syndrome arrays to {odir}/')


if __name__ == '__main__':
    main()
