#!/usr/bin/env python3
"""
直接看串扰影响的差分图。

3 个面板（直接对比 vs baseline）：
  (a) PAEMS L2+X2 - PAEMS L2 pure  ← 模型给出的"串扰特征"
  (b) Real Willow - PAEMS L2 pure  ← 真机比 pure 模型多出的（含真实串扰+leakage+...）
  (c) Real Willow - PAEMS L2+X2    ← 模型还没捕获的剩余偏差

红色 = 这一对 detector 真机比模型关联强
蓝色 = 模型比真机关联强
"""
import sys, os, numpy as np, subprocess, json
import matplotlib.pyplot as plt

PAEMS_SC = r"C:\PAEMS-Precise_and_Adaptive_Error_Model_for_superconducting_Quantum_Processors\Surface_Code_Simulation"
sys.path.insert(0, PAEMS_SC)
from inject_basic_noise import inject_surface_code_noise
from surface_code_generate_circuits import generate_surface_code_circuit
import stim


def compute_pij_spitz(dets):
    dets = dets.astype(np.float64)
    p_x = dets.mean(axis=0)
    p_xy = (dets.T @ dets) / dets.shape[0]
    xi = p_x[:, None]; xj = p_x[None, :]
    denom = (1 - 2 * xi) * (1 - 2 * xj)
    numer = p_xy - xi * xj
    with np.errstate(divide='ignore', invalid='ignore'):
        p = np.where(np.abs(denom) > 1e-12, numer / denom, 0.0)
    np.fill_diagonal(p, 0.0)
    return p   # 保留正负


def detector_to_qubit_order(circuit):
    det_coords = circuit.get_detector_coordinates()
    qubit_coords = {}
    for inst in circuit:
        if inst.name == "QUBIT_COORDS":
            qid = inst.targets_copy()[0].value
            qubit_coords[qid] = tuple(inst.gate_args_copy())
    det_to_qubit = {}
    for did, coord in det_coords.items():
        dx, dy = coord[0], coord[1] if len(coord) >= 2 else 0
        t = coord[2] if len(coord) > 2 else 0
        best_q, best_d = None, float('inf')
        for q, (qx, qy) in qubit_coords.items():
            d = (dx - qx) ** 2 + (dy - qy) ** 2
            if d < best_d:
                best_d, best_q = d, q
        det_to_qubit[did] = (best_q, t)
    sorted_did = sorted(det_coords.keys(), key=lambda d: (det_to_qubit[d][0], det_to_qubit[d][1]))
    return np.array(sorted_did), np.array([det_to_qubit[d][0] for d in sorted_did])


def gen_config(level, xtalk_preset, distance, rounds, out_path, here):
    class_path = os.path.join(here, 'tmp_test', f'tmp_class_L{level}_d{distance}_r{rounds}.json')
    subprocess.run([sys.executable, os.path.join(here, 'gen_level_params.py'),
                    '--level', str(level), '--distance', str(distance),
                    '--rounds', str(rounds), '--seed', '42', '--out', class_path],
                   capture_output=True, check=True)
    if xtalk_preset:
        subprocess.run([sys.executable, os.path.join(here, 'gen_pair_overrides.py'),
                        '--in', class_path, '--out', out_path, '--merge',
                        '--crosstalk-config', os.path.join(here, 'crosstalk_presets', xtalk_preset)],
                       capture_output=True, check=True)
    else:
        with open(class_path, encoding='utf-8') as f:
            d = json.load(f)
        d['crosstalk_pairs'] = {}
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(d, f, indent=2)


import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--distance', type=int, default=5)
ap.add_argument('--rounds', type=int, default=30)
ap.add_argument('--shots', type=int, default=50000)
ap.add_argument('--patch', type=str, default='q4_7')
args = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, 'tmp_test'), exist_ok=True)

paems_pure = os.path.join(HERE, 'tmp_test', f'p_pure_d{args.distance}_r{args.rounds}.json')
paems_x2 = os.path.join(HERE, 'tmp_test', f'p_L2X2_d{args.distance}_r{args.rounds}.json')
gen_config(2, None, args.distance, args.rounds, paems_pure, HERE)
gen_config(2, 'crosstalk_X2.json', args.distance, args.rounds, paems_x2, HERE)

print(f"Sampling PAEMS d={args.distance} r={args.rounds} ...")
c, dq, xs, zs, cx = generate_surface_code_circuit(args.distance, args.rounds, 'z')
nc_p = inject_surface_code_noise(c, dq, xs, zs, cx, paems_pure)
nc_x = inject_surface_code_noise(c, dq, xs, zs, cx, paems_x2)
dets_pure = nc_p.compile_detector_sampler().sample(shots=args.shots)
dets_x2 = nc_x.compile_detector_sampler().sample(shots=args.shots)

print(f"Loading real Willow...")
base = (rf"C:\Users\10124\Desktop\google dataset\google_105Q_surface_code_d3_d5_d7"
        rf"\d{args.distance}_at_{args.patch}\Z\r{args.rounds}")
cir_real = stim.Circuit.from_file(os.path.join(base, "circuit_noisy_si1000.stim"))
dets_real = stim.read_shot_data_file(
    path=os.path.join(base, "detection_events.b8"),
    format='b8', num_detectors=cir_real.num_detectors, bit_packed=False)

# Reorder by qubit
sort_p, qpd_p = detector_to_qubit_order(nc_p)
sort_r, qpd_r = detector_to_qubit_order(cir_real)
dets_pure = dets_pure[:, sort_p]
dets_x2 = dets_x2[:, sort_p]
dets_real = dets_real[:, sort_r]

# IMPORTANT: PAEMS and Real may have different detector counts.
# Need to align them — use min size or skip if mismatched.
n_pure = dets_pure.shape[1]
n_real = dets_real.shape[1]
print(f"PAEMS dets: {n_pure}  Real dets: {n_real}")
if n_pure != n_real:
    n = min(n_pure, n_real)
    print(f"  → truncating to {n}")
    dets_pure = dets_pure[:, :n]
    dets_x2 = dets_x2[:, :n]
    dets_real = dets_real[:, :n]
    qpd_p = qpd_p[:n]; qpd_r = qpd_r[:n]

print("Computing matrices...")
m_pure = compute_pij_spitz(dets_pure)
m_x2 = compute_pij_spitz(dets_x2)
m_real = compute_pij_spitz(dets_real)

# Diffs
diff_xtalk_only = m_x2 - m_pure          # PAEMS 串扰单独贡献
diff_real_vs_pure = m_real - m_pure      # 真机比 pure 多多少
diff_real_vs_full = m_real - m_x2        # PAEMS L2+X2 还没捕获的

# Plot 2x3
fig, axes = plt.subplots(2, 3, figsize=(20, 13))

# Top row: original matrices (positive only, Reds)
vmax_pos = max(m_pure.max(), m_x2.max(), m_real.max()) * 0.7
configs_top = [
    (m_pure, "PAEMS L2 pure (no crosstalk)", qpd_p),
    (m_x2,   "PAEMS L2 + X2",                qpd_p),
    (m_real, f"Real Willow {args.patch}",    qpd_r),
]
for ax, (mat, title, qpd) in zip(axes[0], configs_top):
    im = ax.matshow(np.clip(mat, 0, None), cmap='Reds', vmin=0, vmax=vmax_pos)
    ax.set_title(title, fontsize=11)
    boundaries = [i - 0.5 for i in range(1, len(qpd)) if qpd[i] != qpd[i-1]]
    for b in boundaries:
        ax.axhline(b, color='gray', linewidth=0.2, alpha=0.5)
        ax.axvline(b, color='gray', linewidth=0.2, alpha=0.5)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046)

# Bottom row: differences (RdBu_r diverging)
vmax_diff = max(abs(diff_xtalk_only).max(), abs(diff_real_vs_pure).max()) * 0.6
configs_bot = [
    (diff_xtalk_only,   "PAEMS X2 - pure  (modeled crosstalk effect)",     qpd_p),
    (diff_real_vs_pure, "Real - PAEMS pure  (real-chip extra beyond classical)", qpd_r),
    (diff_real_vs_full, "Real - PAEMS X2  (residual gap, mostly leakage)",  qpd_r),
]
for ax, (mat, title, qpd) in zip(axes[1], configs_bot):
    im = ax.matshow(mat, cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff)
    ax.set_title(title, fontsize=11)
    boundaries = [i - 0.5 for i in range(1, len(qpd)) if qpd[i] != qpd[i-1]]
    for b in boundaries:
        ax.axhline(b, color='gray', linewidth=0.2, alpha=0.5)
        ax.axvline(b, color='gray', linewidth=0.2, alpha=0.5)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
out_png = os.path.join(HERE, f'corrmat_diff_d{args.distance}_r{args.rounds}.png')
plt.savefig(out_png, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out_png}")

print()
print("Diff stats:")
for mat, name in [(diff_xtalk_only, "PAEMS X2-pure"),
                  (diff_real_vs_pure, "Real-pure"),
                  (diff_real_vs_full, "Real-X2 (residual)")]:
    pos = mat[mat > 0]
    neg = mat[mat < 0]
    print(f"  {name:<25} pos: mean={pos.mean():.5f} max={pos.max():.5f}  "
          f"neg: mean={neg.mean() if len(neg) else 0:.5f}  abs_total={np.abs(mat).sum():.3f}")
