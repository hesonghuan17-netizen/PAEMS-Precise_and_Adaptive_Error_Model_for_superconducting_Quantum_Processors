#!/usr/bin/env python3
"""
纯 crosstalk 特征对比：只看 hop≥2 的 cross-qubit pair（无 CX 直邻）。

排除掉：
  - within-block (同 qubit 跨 round)         ← QEC 自然结构
  - hop=1 pair (CX 直邻)                      ← 被 CX 门噪声覆盖

只保留：
  - hop≥2 cross-qubit pair                    ← 这才是 spectator crosstalk 居住地
"""
import sys, os, numpy as np, subprocess, json
import matplotlib.pyplot as plt
import networkx as nx

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
    return np.array(sorted_did), np.array([det_to_qubit[d][0] for d in sorted_did]), qubit_coords


def extract_cx_coupling_from_circuit(circuit):
    """Extract qubit pairs that share a CX gate (1-hop neighbors)."""
    G = nx.Graph()
    for inst in circuit:
        if inst.name == "CX":
            ts = [t.value for t in inst.targets_copy()]
            for i in range(0, len(ts), 2):
                G.add_edge(ts[i], ts[i + 1])
        elif inst.name == "REPEAT":
            for sub in inst.body_copy():
                if sub.name == "CX":
                    ts = [t.value for t in sub.targets_copy()]
                    for i in range(0, len(ts), 2):
                        G.add_edge(ts[i], ts[i + 1])
    return G


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

# Reorder + extract CX coupling
sort_p, qpd_p, _ = detector_to_qubit_order(nc_p)
sort_r, qpd_r, _ = detector_to_qubit_order(cir_real)
G_p = extract_cx_coupling_from_circuit(nc_p)
G_r = extract_cx_coupling_from_circuit(cir_real)

dets_pure = dets_pure[:, sort_p]
dets_x2 = dets_x2[:, sort_p]
dets_real = dets_real[:, sort_r]
n = min(dets_pure.shape[1], dets_real.shape[1])
dets_pure, dets_x2, dets_real = dets_pure[:, :n], dets_x2[:, :n], dets_real[:, :n]
qpd_p, qpd_r = qpd_p[:n], qpd_r[:n]

print("Computing matrices + hop distances...")
m_pure = compute_pij_spitz(dets_pure)
m_x2 = compute_pij_spitz(dets_x2)
m_real = compute_pij_spitz(dets_real)

# All-pairs shortest path on each graph
sp_p = dict(nx.all_pairs_shortest_path_length(G_p))
sp_r = dict(nx.all_pairs_shortest_path_length(G_r))


def filter_by_hop(p_matrix, qpd, sp_dict, min_hop):
    """Return p_ij values for pairs (i,j) where qubit_i != qubit_j AND hop(qi,qj) >= min_hop."""
    iu = np.triu_indices_from(p_matrix, k=1)
    qi = qpd[iu[0]]
    qj = qpd[iu[1]]
    different_q = qi != qj
    hops = np.array([sp_dict.get(int(a), {}).get(int(b), 999)
                     for a, b in zip(qi, qj)])
    keep = different_q & (hops >= min_hop)
    return p_matrix[iu][keep], hops[keep]


# Three categories of pairs:
#   1. "all cross-block" — different qubits (incl. hop=1 CX neighbors)
#   2. "no-CX cross-block" — different qubits AND hop>=2 (PURE crosstalk)
print()
print("Filtering pairs by hop distance...")

pure_cb_all, _ = filter_by_hop(m_pure, qpd_p, sp_p, 1)    # all cross-block (hop>=1)
pure_xtalk, _ = filter_by_hop(m_pure, qpd_p, sp_p, 2)     # pure crosstalk (hop>=2)
x2_cb_all, _ = filter_by_hop(m_x2, qpd_p, sp_p, 1)
x2_xtalk, _ = filter_by_hop(m_x2, qpd_p, sp_p, 2)
real_cb_all, _ = filter_by_hop(m_real, qpd_r, sp_r, 1)
real_xtalk, _ = filter_by_hop(m_real, qpd_r, sp_r, 2)

# Print summary
def stats(arr, label):
    return f"{label:<30} n={len(arr)} mean={arr.mean():.5f} max={arr.max():.5f} top10={np.sort(arr)[-10:].mean():.5f} >1e-3:{(arr>1e-3).sum()}"

print()
print("=== Pure crosstalk signature (hop>=2 cross-qubit only) ===")
print(stats(pure_xtalk, "PAEMS pure"))
print(stats(x2_xtalk,   "PAEMS L2+X2"))
print(stats(real_xtalk, "Real Willow"))

# Plot
fig, axes = plt.subplots(2, 2, figsize=(15, 11))

# (a) hist of pure crosstalk pairs
ax = axes[0, 0]
bins = np.logspace(-6, -1, 50)
ax.hist(np.clip(pure_xtalk, 1e-7, None), bins=bins, alpha=0.5, label='PAEMS pure', color='C0', edgecolor='black', linewidth=0.5)
ax.hist(np.clip(x2_xtalk, 1e-7, None), bins=bins, alpha=0.5, label='PAEMS L2+X2', color='C1', edgecolor='black', linewidth=0.5)
ax.hist(np.clip(real_xtalk, 1e-7, None), bins=bins, alpha=0.5, label='Real Willow', color='C2', edgecolor='black', linewidth=0.5)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('p_ij'); ax.set_ylabel('count')
ax.set_title('(a) Pure crosstalk pairs only (hop≥2 cross-qubit)\n— direct vs真机分布对比')
ax.legend(); ax.grid(alpha=0.3)

# (b) Top-N strongest pure crosstalk pairs
ax = axes[0, 1]
N = 200
ax.plot(np.sort(pure_xtalk)[-N:][::-1], 'o-', label='PAEMS pure', markersize=3, color='C0')
ax.plot(np.sort(x2_xtalk)[-N:][::-1],   'o-', label='PAEMS L2+X2', markersize=3, color='C1')
ax.plot(np.sort(real_xtalk)[-N:][::-1], 'o-', label='Real Willow', markersize=3, color='C2')
ax.set_yscale('log')
ax.set_xlabel(f'rank (top {N} pairs)'); ax.set_ylabel('p_ij')
ax.set_title(f'(b) Top-{N} strongest pure-crosstalk pairs')
ax.legend(); ax.grid(alpha=0.3)

# (c) Comparison: ALL cross-block vs PURE crosstalk only
ax = axes[1, 0]
configs_c = [
    (real_cb_all, 'Real ALL cross-block (含 1-hop)', 'C2', '-'),
    (real_xtalk,  'Real ONLY hop≥2 (纯 crosstalk)',  'C2', '--'),
    (x2_cb_all,   'PAEMS X2 ALL cross-block',         'C1', '-'),
    (x2_xtalk,    'PAEMS X2 ONLY hop≥2',              'C1', '--'),
]
for arr, lab, c, ls in configs_c:
    h, _ = np.histogram(np.clip(arr, 1e-7, None), bins=bins)
    ax.step(bins[:-1], h + 0.1, label=lab, color=c, linestyle=ls, where='post')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('p_ij'); ax.set_ylabel('count')
ax.set_title('(c) ALL cross-block vs PURE crosstalk')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# (d) summary stats
ax = axes[1, 1]
ax.axis('off')
text = []
text.append("=== PURE crosstalk signature (hop≥2 cross-qubit) ===")
text.append("")
text.append(f"{'':<22} {'PAEMS pure':>11} {'PAEMS X2':>11} {'Real':>11}")
text.append("-" * 60)
text.append(f"{'pairs (hop≥2)':<22} {len(pure_xtalk):>11} {len(x2_xtalk):>11} {len(real_xtalk):>11}")
text.append(f"{'mean':<22} {pure_xtalk.mean():>11.5f} {x2_xtalk.mean():>11.5f} {real_xtalk.mean():>11.5f}")
text.append(f"{'max':<22} {pure_xtalk.max():>11.5f} {x2_xtalk.max():>11.5f} {real_xtalk.max():>11.5f}")
text.append(f"{'90th pctile':<22} {np.percentile(pure_xtalk,90):>11.5f} {np.percentile(x2_xtalk,90):>11.5f} {np.percentile(real_xtalk,90):>11.5f}")
text.append(f"{'99th pctile':<22} {np.percentile(pure_xtalk,99):>11.5f} {np.percentile(x2_xtalk,99):>11.5f} {np.percentile(real_xtalk,99):>11.5f}")
text.append(f"{'top-10 mean':<22} {np.sort(pure_xtalk)[-10:].mean():>11.5f} {np.sort(x2_xtalk)[-10:].mean():>11.5f} {np.sort(real_xtalk)[-10:].mean():>11.5f}")
text.append(f"{'fraction > 1e-3':<22} {(pure_xtalk>1e-3).mean()*100:>10.2f}% {(x2_xtalk>1e-3).mean()*100:>10.2f}% {(real_xtalk>1e-3).mean()*100:>10.2f}%")
text.append(f"{'fraction > 1e-2':<22} {(pure_xtalk>1e-2).mean()*100:>10.2f}% {(x2_xtalk>1e-2).mean()*100:>10.2f}% {(real_xtalk>1e-2).mean()*100:>10.2f}%")
text.append("")
text.append("X2 - pure (PAEMS xtalk addition):")
text.append(f"  X2 - pure mean = {x2_xtalk.mean() - pure_xtalk.mean():.6f}")
text.append(f"  X2 - pure max  = {x2_xtalk.max() - pure_xtalk.max():.5f}")
text.append("")
text.append("Real - pure (real chip's extra):")
text.append(f"  Real - pure mean = {real_xtalk.mean() - pure_xtalk.mean():.6f}")
text.append(f"  Real - pure max  = {real_xtalk.max() - pure_xtalk.max():.5f}")
text.append("")
text.append("Match ratio (X2 contribution / Real extra):")
match_mean = (x2_xtalk.mean() - pure_xtalk.mean()) / max(real_xtalk.mean() - pure_xtalk.mean(), 1e-10)
match_max = (x2_xtalk.max() - pure_xtalk.max()) / max(real_xtalk.max() - pure_xtalk.max(), 1e-10)
text.append(f"  by mean: {match_mean*100:.1f}%")
text.append(f"  by max : {match_max*100:.1f}%")

ax.text(0.0, 1.0, '\n'.join(text), va='top', family='monospace', fontsize=9)
ax.set_title('(d) Pure crosstalk statistics summary')

plt.tight_layout()
out_png = os.path.join(HERE, f'corrmat_xtalk_only_d{args.distance}_r{args.rounds}.png')
plt.savefig(out_png, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out_png}")
