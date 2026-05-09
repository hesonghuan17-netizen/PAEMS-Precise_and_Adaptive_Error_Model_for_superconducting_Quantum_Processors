#!/usr/bin/env python3
"""
对比 PAEMS pure / X1 / X2 / X3 / X4 / Real 的 hop≥2 cross-qubit pair 分布。
看不同强度的 crosstalk 在统计指标上是否有可观察差异。
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
    return np.array(sorted_did), np.array([det_to_qubit[d][0] for d in sorted_did])


def cx_graph(circuit):
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
ap.add_argument('--shots', type=int, default=100000)
ap.add_argument('--patch', type=str, default='q4_7')
args = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, 'tmp_test'), exist_ok=True)

# Generate all 5 PAEMS configs (pure, X1, X2, X3, X4)
print("Generating configs...")
configs = {}
configs['pure'] = os.path.join(HERE, 'tmp_test', f'_pure_d{args.distance}r{args.rounds}.json')
gen_config(2, None, args.distance, args.rounds, configs['pure'], HERE)
for X in [1, 2, 3, 4]:
    out = os.path.join(HERE, 'tmp_test', f'_X{X}_d{args.distance}r{args.rounds}.json')
    gen_config(2, f'crosstalk_X{X}.json', args.distance, args.rounds, out, HERE)
    configs[f'X{X}'] = out

# Sample
print(f"Sampling {args.shots} shots × 5 configs × d={args.distance} r={args.rounds}...")
c, dq, xs, zs, cx = generate_surface_code_circuit(args.distance, args.rounds, 'z')
all_dets = {}
for name, path in configs.items():
    nc = inject_surface_code_noise(c, dq, xs, zs, cx, path)
    all_dets[name] = nc.compile_detector_sampler().sample(shots=args.shots)
    print(f"  {name}: {all_dets[name].shape}")

# Real
nc_pure = inject_surface_code_noise(c, dq, xs, zs, cx, configs['pure'])
sort_p, qpd_p = detector_to_qubit_order(nc_pure)
G_p = cx_graph(nc_pure)

base = (rf"C:\Users\10124\Desktop\google dataset\google_105Q_surface_code_d3_d5_d7"
        rf"\d{args.distance}_at_{args.patch}\Z\r{args.rounds}")
cir_real = stim.Circuit.from_file(os.path.join(base, "circuit_noisy_si1000.stim"))
dets_real = stim.read_shot_data_file(
    path=os.path.join(base, "detection_events.b8"),
    format='b8', num_detectors=cir_real.num_detectors, bit_packed=False)
sort_r, qpd_r = detector_to_qubit_order(cir_real)
G_r = cx_graph(cir_real)

# Reorder + align
for name in all_dets:
    all_dets[name] = all_dets[name][:, sort_p]
dets_real = dets_real[:, sort_r]
n = min(min(d.shape[1] for d in all_dets.values()), dets_real.shape[1])
for name in all_dets:
    all_dets[name] = all_dets[name][:, :n]
dets_real = dets_real[:, :n]
qpd_p, qpd_r = qpd_p[:n], qpd_r[:n]

# Compute matrices + filter to hop>=2 cross-qubit
print("Computing matrices + filtering hop>=2 ...")
sp_p = dict(nx.all_pairs_shortest_path_length(G_p))
sp_r = dict(nx.all_pairs_shortest_path_length(G_r))


def filter_hop_ge2(m, qpd, sp_dict):
    iu = np.triu_indices_from(m, k=1)
    qi = qpd[iu[0]]; qj = qpd[iu[1]]
    different_q = qi != qj
    hops = np.array([sp_dict.get(int(a), {}).get(int(b), 999)
                     for a, b in zip(qi, qj)])
    return m[iu][different_q & (hops >= 2)]


paems_results = {}
for name, dets in all_dets.items():
    m = compute_pij_spitz(dets)
    paems_results[name] = filter_hop_ge2(m, qpd_p, sp_p)
m_real = compute_pij_spitz(dets_real)
real_xtalk = filter_hop_ge2(m_real, qpd_r, sp_r)

# Print summary
print()
print(f"{'config':<10} {'mean':>10} {'max':>10} {'90th':>10} {'99th':>10} {'top-10':>10} {'>1e-3':>10} {'>1e-2':>8}")
print('-' * 85)
for name in ['pure', 'X1', 'X2', 'X3', 'X4']:
    arr = paems_results[name]
    print(f"{name:<10} {arr.mean():>10.5f} {arr.max():>10.5f} "
          f"{np.percentile(arr,90):>10.5f} {np.percentile(arr,99):>10.5f} "
          f"{np.sort(arr)[-10:].mean():>10.5f} "
          f"{(arr>1e-3).mean()*100:>9.2f}% {(arr>1e-2).mean()*100:>7.2f}%")
print(f"{'Real':<10} {real_xtalk.mean():>10.5f} {real_xtalk.max():>10.5f} "
      f"{np.percentile(real_xtalk,90):>10.5f} {np.percentile(real_xtalk,99):>10.5f} "
      f"{np.sort(real_xtalk)[-10:].mean():>10.5f} "
      f"{(real_xtalk>1e-3).mean()*100:>9.2f}% {(real_xtalk>1e-2).mean()*100:>7.2f}%")

# Plot histograms
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
bins = np.logspace(-6, -1, 50)
colors = {'pure': 'gray', 'X1': 'C0', 'X2': 'C1', 'X3': 'C3', 'X4': 'C4', 'Real': 'C2'}
for name in ['pure', 'X1', 'X2', 'X3', 'X4']:
    arr = paems_results[name]
    ax[0].hist(np.clip(arr, 1e-7, None), bins=bins, alpha=0.4, label=f'PAEMS {name}',
               color=colors[name], edgecolor='black', linewidth=0.3)
ax[0].hist(np.clip(real_xtalk, 1e-7, None), bins=bins, alpha=0.4, label='Real Willow',
           color=colors['Real'], edgecolor='black', linewidth=0.3)
ax[0].set_xscale('log'); ax[0].set_yscale('log')
ax[0].set_xlabel('p_ij (hop>=2 cross-qubit only)'); ax[0].set_ylabel('count')
ax[0].set_title('Distribution of pure-crosstalk pairs by level')
ax[0].legend(); ax[0].grid(alpha=0.3)

# Top-200 sorted comparison
N = 200
for name in ['pure', 'X1', 'X2', 'X3', 'X4']:
    arr = paems_results[name]
    ax[1].plot(np.sort(arr)[-N:][::-1], '-', label=f'PAEMS {name}', color=colors[name])
ax[1].plot(np.sort(real_xtalk)[-N:][::-1], '-', label='Real Willow', color=colors['Real'], linewidth=2)
ax[1].set_yscale('log'); ax[1].set_xlabel(f'rank (top {N})'); ax[1].set_ylabel('p_ij')
ax[1].set_title(f'Top-{N} pure-crosstalk pairs')
ax[1].legend(); ax[1].grid(alpha=0.3)

plt.tight_layout()
out_png = os.path.join(HERE, f'corrmat_xtalk_levels_d{args.distance}_r{args.rounds}.png')
plt.savefig(out_png, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out_png}")
