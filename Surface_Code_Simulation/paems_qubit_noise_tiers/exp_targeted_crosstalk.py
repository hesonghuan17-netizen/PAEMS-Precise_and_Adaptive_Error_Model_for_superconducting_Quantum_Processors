#!/usr/bin/env python3
"""
靶向串扰实验：手动给 (q_a, q_b) 一对设强串扰，看 detector firing rate 是否
特异性升高在那两个 qubit 关联的 detector 上。

比较：
  Baseline    : chi_default=0, no pairs (no crosstalk)
  Test        : chi_default=0, chi_pairs = {f"{q_a}-{q_b}": 0.01}

输出：
  - 每个 detector firing rate delta
  - Top-K 上升最多的 detector 反查到的物理 qubit 坐标
  - 验证这些 qubit 是否就是 (q_a, q_b)
"""
import sys, os, json, tempfile
from pathlib import Path
import numpy as np

PAEMS_SC = r"C:\PAEMS-Precise_and_Adaptive_Error_Model_for_superconducting_Quantum_Processors\Surface_Code_Simulation"
sys.path.insert(0, PAEMS_SC)

from inject_basic_noise import inject_surface_code_noise
from surface_code_generate_circuits import generate_surface_code_circuit

import stim

HERE = Path(__file__).parent
SAMPLE_JSON = HERE / "validation_d3" / "param_examples" / "d3_level2.json"


def make_test_json(out_path, chi_default=0.0, chi_pairs=None):
    with open(SAMPLE_JSON) as f:
        d = json.load(f)
    d['crosstalk_global'] = {"default_strength": float(chi_default)}
    d['crosstalk_pairs'] = chi_pairs or {}
    with open(out_path, 'w') as f:
        json.dump(d, f, indent=2)


def build_and_sample(chi_default, chi_pairs, distance, rounds, shots, basis='z'):
    circuit, data_q, x_stab, z_stab, cx_gates = generate_surface_code_circuit(
        distance, rounds, basis)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        tmp = f.name
    try:
        make_test_json(tmp, chi_default, chi_pairs)
        nc = inject_surface_code_noise(circuit, data_q, x_stab, z_stab, cx_gates, tmp)
    finally:
        os.unlink(tmp)
    dets = nc.compile_detector_sampler().sample(shots=shots)
    return dets, nc, (data_q, x_stab, z_stab)


def get_qubit_coords(circuit_with_coords):
    """从原始 surface code 电路读 QUBIT_COORDS。"""
    coords = {}
    for inst in circuit_with_coords:
        if inst.name == "QUBIT_COORDS":
            qid = inst.targets_copy()[0].value
            xy = inst.gate_args_copy()
            coords[qid] = tuple(xy)
    return coords


def get_detector_coords(noisy_circuit):
    """对每个 detector 拿坐标 (x,y,t)。Stim API。"""
    return noisy_circuit.get_detector_coordinates()


def match_detector_to_qubit(detector_coords, qubit_coords, tol=0.5):
    """对每个 detector，找最近的 qubit (按 xy 距离)。"""
    det2q = {}
    for did, dxyt in detector_coords.items():
        dx, dy = dxyt[0], dxyt[1] if len(dxyt) >= 2 else 0
        best_q, best_d = None, float('inf')
        for q, qxy in qubit_coords.items():
            qx, qy = qxy[0], qxy[1] if len(qxy) >= 2 else 0
            d = (dx - qx) ** 2 + (dy - qy) ** 2
            if d < best_d:
                best_d, best_q = d, q
        det2q[did] = (best_q, best_d ** 0.5)
    return det2q


# ============================================================
# 主实验
# ============================================================
DISTANCE = 3
ROUNDS = 3
SHOTS = 100000
CHI_HEAVY = 0.01     # 1% per gate — 很强

# 拓扑
sc_circuit, data_q, x_stab, z_stab, cx_gates = generate_surface_code_circuit(
    DISTANCE, ROUNDS, 'z')
qubit_coords = get_qubit_coords(sc_circuit)
print(f'Topology d={DISTANCE} r={ROUNDS}:')
print(f'  data qubits ({len(data_q)}): {data_q}')
print(f'  x stab     ({len(x_stab)}): {x_stab}')
print(f'  z stab     ({len(z_stab)}): {z_stab}')
print()

# 选一对 ancilla（避开 CX 直连边，避免 validator 报警）
cx_pairs_set = set()
for gid, (c, t) in cx_gates:
    cx_pairs_set.add((c, t))
    cx_pairs_set.add((t, c))

# 找两个 ancilla：一个 X stab + 一个 Z stab，且不直接 CX 相连
pair = None
for a in x_stab:
    for b in z_stab:
        if (a, b) not in cx_pairs_set and a != b:
            pair = (a, b)
            break
    if pair: break

q_a, q_b = pair
print(f'>>> Test pair: q_a={q_a} (X-stab @ {qubit_coords[q_a]}), '
      f'q_b={q_b} (Z-stab @ {qubit_coords[q_b]})')
print(f'>>> Chi for this pair: {CHI_HEAVY:.0e}, all others: 0')
print()

# Run baseline
print('Running BASELINE (no crosstalk)...')
dets_base, nc_base, _ = build_and_sample(0.0, {}, DISTANCE, ROUNDS, SHOTS)
rate_base = dets_base.mean(axis=0)
print(f'  baseline density = {dets_base.mean()*100:.3f}%')

# Run test
print(f'Running TEST  (chi[{q_a}-{q_b}]={CHI_HEAVY})...')
chi_pairs = {f'{q_a}-{q_b}': CHI_HEAVY}
dets_test, nc_test, _ = build_and_sample(0.0, chi_pairs, DISTANCE, ROUNDS, SHOTS)
rate_test = dets_test.mean(axis=0)
print(f'  test density     = {dets_test.mean()*100:.3f}%')
print()

# 比较
delta = rate_test - rate_base
det_coords = nc_base.get_detector_coordinates()
det2q = match_detector_to_qubit(det_coords, qubit_coords)

print(f'{"det_id":>7} {"coord (x,y,t)":>20} {"nearest qubit":>14} {"baseline":>10} {"test":>10} {"delta":>10} {"hit?":>6}')
print('-' * 90)

# 排序：delta 最大的在前
order = np.argsort(-delta)
hit_count = 0
for did in order[:15]:
    coord = det_coords[did]
    q, d = det2q[did]
    is_hit = q in (q_a, q_b)
    if is_hit: hit_count += 1
    mark = '*** YES' if is_hit else ''
    print(f'{did:>7} {str(coord):>20} {q:>14} {rate_base[did]*100:>9.3f}% '
          f'{rate_test[did]*100:>9.3f}% {delta[did]*100:>+9.3f}pp  {mark}')

print()
print(f'Top-15 detectors with biggest delta: {hit_count}/15 belong to ({q_a}, {q_b})')

# 也算所有归属到 (q_a, q_b) 的 detector 的平均 delta vs 其他 detector 的平均 delta
target_dets = [did for did, (q, _) in det2q.items() if q in (q_a, q_b)]
other_dets = [did for did, (q, _) in det2q.items() if q not in (q_a, q_b)]
print()
print(f'Detector partition:')
print(f'  belong to ({q_a}, {q_b}): {len(target_dets)} detectors, '
      f'mean delta = {delta[target_dets].mean()*100:+.4f}pp')
print(f'  others                : {len(other_dets)} detectors, '
      f'mean delta = {delta[other_dets].mean()*100:+.4f}pp')

ratio = delta[target_dets].mean() / max(abs(delta[other_dets].mean()), 1e-10)
print(f'  ratio (target / other): {ratio:.1f}x')
