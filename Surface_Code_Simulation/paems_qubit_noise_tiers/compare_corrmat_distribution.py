#!/usr/bin/env python3
"""
统计分布对比 (分布形状 + 强度量级，不在乎位置匹配)。

3 个对比维度：
  (1) p_ij 强度分布直方图 — 整体强度量级是否一致
  (2) Sorted top-N 比较 — 最强的 N 个 pair 数值是否对得上
  (3) 块结构分析 (block diagonal vs off-block) — 串扰特征区分
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
    return np.clip(p, 0, None)


def detector_to_qubit_order(circuit):
    det_coords = circuit.get_detector_coordinates()
    qubit_coords = {}
    for inst in circuit:
        if inst.name == "QUBIT_COORDS":
            qid = inst.targets_copy()[0].value
            qubit_coords[qid] = tuple(inst.gate_args_copy())
    det_to_qubit = {}
    for did, coord in det_coords.items():
        dx, dy = coord[0], coord[1]
        t = coord[2] if len(coord) > 2 else 0
        best_q, best_d = None, float('inf')
        for q, (qx, qy) in qubit_coords.items():
            d = (dx - qx) ** 2 + (dy - qy) ** 2
            if d < best_d:
                best_d, best_q = d, q
        det_to_qubit[did] = (best_q, t)
    sorted_did = sorted(det_coords.keys(), key=lambda d: (det_to_qubit[d][0], det_to_qubit[d][1]))
    return np.array(sorted_did), np.array([det_to_qubit[d][0] for d in sorted_did])


def gen_config(level, xtalk, distance, rounds, out_path, here):
    class_p = os.path.join(here, 'tmp_test', f'_class_L{level}_d{distance}r{rounds}.json')
    subprocess.run([sys.executable, os.path.join(here, 'gen_level_params.py'),
                    '--level', str(level), '--distance', str(distance),
                    '--rounds', str(rounds), '--seed', '42', '--out', class_p],
                   capture_output=True, check=True)
    if xtalk:
        subprocess.run([sys.executable, os.path.join(here, 'gen_pair_overrides.py'),
                        '--in', class_p, '--out', out_path, '--merge',
                        '--crosstalk-config', os.path.join(here, 'crosstalk_presets', xtalk)],
                       capture_output=True, check=True)
    else:
        with open(class_p, encoding='utf-8') as f:
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

# Generate
paems_pure = os.path.join(HERE, 'tmp_test', f'_p_pure_d{args.distance}r{args.rounds}.json')
paems_x2 = os.path.join(HERE, 'tmp_test', f'_p_X2_d{args.distance}r{args.rounds}.json')
gen_config(2, None, args.distance, args.rounds, paems_pure, HERE)
gen_config(2, 'crosstalk_X2.json', args.distance, args.rounds, paems_x2, HERE)

print(f"Sampling d={args.distance} r={args.rounds} shots={args.shots} ...")
c, dq, xs, zs, cx = generate_surface_code_circuit(args.distance, args.rounds, 'z')
nc_p = inject_surface_code_noise(c, dq, xs, zs, cx, paems_pure)
nc_x = inject_surface_code_noise(c, dq, xs, zs, cx, paems_x2)
dets_pure = nc_p.compile_detector_sampler().sample(shots=args.shots)
dets_x2 = nc_x.compile_detector_sampler().sample(shots=args.shots)

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

# Align
n = min(dets_pure.shape[1], dets_real.shape[1])
dets_pure = dets_pure[:, :n]
dets_x2 = dets_x2[:, :n]
dets_real = dets_real[:, :n]
qpd_p = qpd_p[:n]; qpd_r = qpd_r[:n]

print("Computing matrices...")
m_pure = compute_pij_spitz(dets_pure)
m_x2 = compute_pij_spitz(dets_x2)
m_real = compute_pij_spitz(dets_real)

# Get upper-triangular (exclude self-pairs)
def utri(m):
    return m[np.triu_indices_from(m, k=1)]

p_pure = utri(m_pure)
p_x2 = utri(m_x2)
p_real = utri(m_real)

# Block masks
def block_masks(qpd):
    """Return upper-triangular masks for (within_block, cross_block)."""
    n = len(qpd)
    same = (qpd[:, None] == qpd[None, :])
    iu = np.triu_indices(n, k=1)
    same_iu = same[iu]
    return same_iu  # True if same qubit

mask_p = block_masks(qpd_p)  # upper-tri, True=same qubit
mask_r = block_masks(qpd_r)

within_p = p_pure[mask_p]; cross_p = p_pure[~mask_p]
within_x = p_x2[mask_p];   cross_x = p_x2[~mask_p]
within_r = p_real[mask_r]; cross_r = p_real[~mask_r]

# === Plot ===
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# (a) p_ij histogram (log y)
ax = axes[0, 0]
bins = np.logspace(-6, 0, 50)
ax.hist(np.clip(p_pure, 1e-7, None), bins=bins, alpha=0.5, label='PAEMS L2 pure', color='C0')
ax.hist(np.clip(p_x2, 1e-7, None), bins=bins, alpha=0.5, label='PAEMS L2+X2', color='C1')
ax.hist(np.clip(p_real, 1e-7, None), bins=bins, alpha=0.5, label='Real Willow', color='C2')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('p_ij value'); ax.set_ylabel('count (log)')
ax.set_title('(a) Distribution of all p_ij values')
ax.legend()
ax.grid(alpha=0.3)

# (b) sorted top-N strongest
ax = axes[0, 1]
N = 200
ax.plot(np.sort(p_pure)[-N:][::-1], 'o-', label='PAEMS L2 pure', markersize=3, color='C0')
ax.plot(np.sort(p_x2)[-N:][::-1], 'o-', label='PAEMS L2+X2', markersize=3, color='C1')
ax.plot(np.sort(p_real)[-N:][::-1], 'o-', label='Real Willow', markersize=3, color='C2')
ax.set_yscale('log')
ax.set_xlabel(f'rank (top {N} strongest pairs)')
ax.set_ylabel('p_ij')
ax.set_title(f'(b) Top-{N} strongest pair correlations')
ax.legend()
ax.grid(alpha=0.3)

# (c) within-block vs cross-block (within = round-to-round same qubit)
ax = axes[1, 0]
bins = np.logspace(-6, 0, 50)
labels = ['PAEMS L2+X2 within', 'PAEMS L2+X2 cross', 'Real within', 'Real cross']
data = [np.clip(within_x, 1e-7, None), np.clip(cross_x, 1e-7, None),
        np.clip(within_r, 1e-7, None), np.clip(cross_r, 1e-7, None)]
colors = ['C1', 'C1', 'C2', 'C2']
linestyles = ['-', '--', '-', '--']
for d, lab, c, ls in zip(data, labels, colors, linestyles):
    h, _ = np.histogram(d, bins=bins)
    ax.step(bins[:-1], h + 0.1, label=lab, color=c, linestyle=ls, where='post')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('p_ij'); ax.set_ylabel('count')
ax.set_title('(c) Within-qubit (round-round) vs Cross-qubit pairs')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# (d) Stat summary table as text
ax = axes[1, 1]
ax.axis('off')
text = []
text.append(f"{'':<22} {'PAEMS pure':>11} {'PAEMS L2+X2':>11} {'Real Willow':>11}")
text.append('-' * 60)
def stat(arr): return f"{arr.mean():.5f} ± {arr.std():.5f}"
text.append(f"{'all p_ij mean ± std':<22} {p_pure.mean():>10.5f}  {p_x2.mean():>10.5f}  {p_real.mean():>10.5f}")
text.append(f"{'all p_ij max':<22} {p_pure.max():>10.5f}  {p_x2.max():>10.5f}  {p_real.max():>10.5f}")
text.append(f"{'90th percentile':<22} {np.percentile(p_pure,90):>10.5f}  {np.percentile(p_x2,90):>10.5f}  {np.percentile(p_real,90):>10.5f}")
text.append(f"{'99th percentile':<22} {np.percentile(p_pure,99):>10.5f}  {np.percentile(p_x2,99):>10.5f}  {np.percentile(p_real,99):>10.5f}")
text.append(f"{'top-10 mean':<22} {np.sort(p_pure)[-10:].mean():>10.5f}  {np.sort(p_x2)[-10:].mean():>10.5f}  {np.sort(p_real)[-10:].mean():>10.5f}")
text.append(f"{'top-100 mean':<22} {np.sort(p_pure)[-100:].mean():>10.5f}  {np.sort(p_x2)[-100:].mean():>10.5f}  {np.sort(p_real)[-100:].mean():>10.5f}")
text.append('')
text.append('Within-block (same qubit):')
text.append(f"  {'mean':<20} {within_p.mean():>10.5f}  {within_x.mean():>10.5f}  {within_r.mean():>10.5f}")
text.append(f"  {'max':<20} {within_p.max():>10.5f}  {within_x.max():>10.5f}  {within_r.max():>10.5f}")
text.append('Cross-block (different qubit):')
text.append(f"  {'mean':<20} {cross_p.mean():>10.5f}  {cross_x.mean():>10.5f}  {cross_r.mean():>10.5f}")
text.append(f"  {'max':<20} {cross_p.max():>10.5f}  {cross_x.max():>10.5f}  {cross_r.max():>10.5f}")
text.append(f"  {'fraction > 1e-3':<20} {(cross_p>1e-3).mean()*100:>9.2f}%  {(cross_x>1e-3).mean()*100:>9.2f}%  {(cross_r>1e-3).mean()*100:>9.2f}%")

ax.text(0.0, 1.0, '\n'.join(text), va='top', family='monospace', fontsize=9)
ax.set_title('(d) Statistical summary')

plt.tight_layout()
out_png = os.path.join(HERE, f'corrmat_distribution_d{args.distance}_r{args.rounds}.png')
plt.savefig(out_png, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out_png}")

# Print summary
print()
print("Cross-block (off-diagonal — where crosstalk lives) statistics:")
print(f"  PAEMS pure  : mean={cross_p.mean():.5f}  max={cross_p.max():.5f}  >1e-3: {(cross_p>1e-3).sum()}")
print(f"  PAEMS L2+X2 : mean={cross_x.mean():.5f}  max={cross_x.max():.5f}  >1e-3: {(cross_x>1e-3).sum()}")
print(f"  Real Willow : mean={cross_r.mean():.5f}  max={cross_r.max():.5f}  >1e-3: {(cross_r>1e-3).sum()}")
