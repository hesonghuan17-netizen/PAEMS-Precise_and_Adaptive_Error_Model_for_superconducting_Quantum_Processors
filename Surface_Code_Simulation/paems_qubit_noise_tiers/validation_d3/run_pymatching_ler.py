#!/usr/bin/env python3
"""
跑 PyMatching 对 4 个 level 解码，输出 LER vs Level 表。

每个 level：
  1. 生成 PAEMS noisy_circuit
  2. circuit.compile_detector_sampler() with separate_observables=True
  3. 采样 (dets, obs)
  4. circuit.detector_error_model() -> pymatching.Matching
  5. predicted_obs = matcher.decode_batch(dets)
  6. LER = mean(predicted != true)

输出：
  - syndrome_data/decode_d{d}_r{r}_shots{K}.json  (汇总表)
  - 终端打印
"""
import argparse
import os
import sys
import json
import numpy as np
from pathlib import Path
import time

PAEMS_SC = r"C:\PAEMS-Precise_and_Adaptive_Error_Model_for_superconducting_Quantum_Processors\Surface_Code_Simulation"
if PAEMS_SC not in sys.path:
    sys.path.insert(0, PAEMS_SC)

from inject_basic_noise import inject_surface_code_noise
from surface_code_generate_circuits import generate_surface_code_circuit

import pymatching


def decode_one_level(distance, rounds, shots, params_file, basis='z'):
    """生成噪声电路 -> 采样 -> DEM -> PyMatching 解码 -> LER。"""
    # 1. 噪声电路
    circuit, data_q, x_stab, z_stab, cx_gates = generate_surface_code_circuit(
        distance, rounds, basis
    )
    noisy_circuit = inject_surface_code_noise(
        circuit, data_q, x_stab, z_stab, cx_gates, params_file
    )

    # 2. 采样 dets + obs
    sampler = noisy_circuit.compile_detector_sampler()
    t0 = time.perf_counter()
    dets, obs = sampler.sample(shots=shots, separate_observables=True)
    sample_time = time.perf_counter() - t0

    # 3. DEM
    dem = noisy_circuit.detector_error_model(
        decompose_errors=True, approximate_disjoint_errors=True
    )

    # 4. PyMatching
    matcher = pymatching.Matching.from_detector_error_model(dem)
    t0 = time.perf_counter()
    pred_obs = matcher.decode_batch(dets)
    decode_time = time.perf_counter() - t0

    # 5. LER
    pred_obs = np.asarray(pred_obs, dtype=np.uint8).reshape(obs.shape)
    obs = np.asarray(obs, dtype=np.uint8)
    errors_per_shot = np.any(pred_obs != obs, axis=1)
    ler = float(errors_per_shot.mean())
    n_errors = int(errors_per_shot.sum())

    # defect 密度（顺便算）
    defect_density = float(dets.mean())
    defects_per_shot = dets.sum(axis=1).mean()

    return {
        'shots': shots,
        'n_detectors': dets.shape[1],
        'n_observables': obs.shape[1],
        'defect_density': defect_density,
        'mean_defects_per_shot': float(defects_per_shot),
        'ler': ler,
        'n_logical_errors': n_errors,
        'sample_time_s': sample_time,
        'decode_time_s': decode_time,
    }


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

    print(f'PyMatching LER sweep: d={args.distance} r={args.rounds} '
          f'shots={args.shots} basis={args.basis}')
    print('=' * 90)

    rows = []
    for L in args.levels:
        param_file = pdir / f'd{args.distance}_level{L}.json'
        if not param_file.exists():
            print(f'[skip] L{L}: {param_file.name} not found')
            continue
        print(f'  Running L{L}...', flush=True)
        r = decode_one_level(args.distance, args.rounds, args.shots,
                            str(param_file), args.basis)
        r['level'] = L
        rows.append(r)

    # 汇总表
    print()
    hdr = ('Lvl', 'defect%', 'def/shot', 'LER', 'n_err', 'sample(s)', 'decode(s)')
    print(f"{hdr[0]:<5} {hdr[1]:>9} {hdr[2]:>9} {hdr[3]:>10} {hdr[4]:>7} {hdr[5]:>10} {hdr[6]:>10}")
    print('-' * 75)
    for r in rows:
        print(f"L{r['level']:<4} "
              f"{r['defect_density']*100:>8.3f}% "
              f"{r['mean_defects_per_shot']:>9.2f} "
              f"{r['ler']*100:>9.3f}% "
              f"{r['n_logical_errors']:>7} "
              f"{r['sample_time_s']:>10.2f} "
              f"{r['decode_time_s']:>10.3f}")

    # 保存 JSON
    out_json = odir / f'decode_d{args.distance}_r{args.rounds}_shots{args.shots}.json'
    with open(out_json, 'w') as f:
        json.dump({
            'config': {
                'distance': args.distance,
                'rounds': args.rounds,
                'shots': args.shots,
                'basis': args.basis,
            },
            'results': rows,
        }, f, indent=2)
    print(f'\nSaved summary: {out_json}')


if __name__ == '__main__':
    main()
