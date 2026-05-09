#!/usr/bin/env python3
"""
Full-stack PAEMS vs Real Willow correlation comparison.

走完整后处理流程（与 run_sampling.py 一致）：
  raw measurements → leakage simulate (vectorized) → flip-affected → m2d → detectors

四种 PAEMS 配置 + Real Willow:
  (1) PAEMS L2 pure                     — classical only
  (2) PAEMS L2 + X2                     — + spectator crosstalk
  (3) PAEMS L2 + Leak2                  — + leakage post-processing
  (4) PAEMS L2 + X2 + Leak2             — full PAEMS
  (5) Real Willow                       — chip data

输出：
  - 5-config Spitz correlation matrix grid (qubit-block ordered)
  - hop>=2 cross-qubit p_ij distribution histogram + top-200 sorted
  - 文本统计表
"""
import sys, os, json, subprocess
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import argparse

PAEMS_SC = r"C:\PAEMS-Precise_and_Adaptive_Error_Model_for_superconducting_Quantum_Processors\Surface_Code_Simulation"
sys.path.insert(0, PAEMS_SC)
from inject_basic_noise import inject_surface_code_noise, load_surface_code_params
from inject_leakage_noise_vectorized import (
    simulate_surface_code_leakage_vectorized,
    extract_measurement_affected_vectorized,
)
from surface_code_generate_circuits import generate_surface_code_circuit
import stim


# ---------------------------------------------------------------------------
# Spitz pij + qubit ordering helpers (same as compare_corrmat_xtalk_levels.py)
# ---------------------------------------------------------------------------
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


def filter_hop_ge2(m, qpd, sp_dict):
    iu = np.triu_indices_from(m, k=1)
    qi = qpd[iu[0]]; qj = qpd[iu[1]]
    different_q = qi != qj
    hops = np.array([sp_dict.get(int(a), {}).get(int(b), 999)
                     for a, b in zip(qi, qj)])
    return m[iu][different_q & (hops >= 2)]


# ---------------------------------------------------------------------------
# Config builders: chain gen_level_params -> gen_pair_overrides -> gen_leakage_overrides
# ---------------------------------------------------------------------------
def run_py(here, *args):
    subprocess.run([sys.executable, *args], capture_output=True, check=True, cwd=here)


def make_config(here, distance, rounds, with_xtalk, with_leak, out_path,
                classical_mult=None, leakage_mult=None,
                q_frac=None, cx_frac=None,
                mult_min=None, mult_max=None):
    """构建一份 base→(可选 xtalk)→(可选 leakage) 的最终 JSON。"""
    tmp_dir = os.path.join(here, 'tmp_test')
    os.makedirs(tmp_dir, exist_ok=True)
    tag = f"d{distance}r{rounds}_x{int(with_xtalk)}_l{int(with_leak)}"

    # Step 1: classical L2 (with defect overlay)
    p_class = os.path.join(tmp_dir, f'_full_class_{tag}.json')
    cargs = ['--level', '2', '--distance', str(distance),
             '--rounds', str(rounds), '--seed', '42', '--defect-seed', '7',
             '--out', p_class]
    if classical_mult is not None:
        cargs += ['--defect-multiplier', str(classical_mult)]
    if q_frac is not None:
        cargs += ['--defect-q-fraction', str(q_frac)]
    if cx_frac is not None:
        cargs += ['--defect-cx-fraction', str(cx_frac)]
    if mult_min is not None:
        cargs += ['--defect-mult-min', str(mult_min)]
    if mult_max is not None:
        cargs += ['--defect-mult-max', str(mult_max)]
    run_py(here, os.path.join(here, 'gen_level_params.py'), *cargs)

    # Step 2: optional X2 overlay
    if with_xtalk:
        p_xt = os.path.join(tmp_dir, f'_full_xt_{tag}.json')
        run_py(here, os.path.join(here, 'gen_pair_overrides.py'),
               '--in', p_class, '--out', p_xt, '--merge',
               '--crosstalk-config', os.path.join(here, 'crosstalk_presets', 'crosstalk_X2.json'))
        cur = p_xt
    else:
        # Strip any crosstalk_pairs (gen_level_params writes empty dict but be explicit)
        with open(p_class, encoding='utf-8') as f:
            d = json.load(f)
        d['crosstalk_pairs'] = {}
        with open(p_class, 'w', encoding='utf-8') as f:
            json.dump(d, f, indent=2)
        cur = p_class

    # Step 3: optional Leakage L2 overlay (with defect-seed)
    if with_leak:
        p_leak = out_path
        largs = ['--in', cur, '--leakage-config',
                 os.path.join(here, 'leakage_presets', 'leakage_L2.json'),
                 '--seed', '42', '--defect-seed', '7', '--out', p_leak]
        if leakage_mult is not None:
            largs += ['--defect-multiplier', str(leakage_mult)]
        if q_frac is not None:
            largs += ['--defect-q-fraction', str(q_frac)]
        if cx_frac is not None:
            largs += ['--defect-cx-fraction', str(cx_frac)]
        if mult_min is not None:
            largs += ['--defect-mult-min', str(mult_min)]
        if mult_max is not None:
            largs += ['--defect-mult-max', str(mult_max)]
        run_py(here, os.path.join(here, 'gen_leakage_overrides.py'), *largs)
    else:
        # Just copy `cur` to out_path (leakage params already 0)
        with open(cur, encoding='utf-8') as f:
            d = json.load(f)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(d, f, indent=2)


# ---------------------------------------------------------------------------
# PAEMS sampling with optional leakage post-processing -> detector samples
# ---------------------------------------------------------------------------
def sample_paems_full(distance, rounds, params_file, shots, with_leakage,
                      basis='z', batch_size=2000):
    """Returns (det_samples, noisy_circuit) where det_samples shape (shots, n_dets).

    If with_leakage: raw measurement sample -> leakage post-process flip -> m2d.
    Else: detector_sampler() directly (faster, equivalent in this case).
    """
    base_circuit, data_q, x_stab, z_stab, cx_gates = generate_surface_code_circuit(
        distance, rounds, basis)
    nc = inject_surface_code_noise(base_circuit, data_q, x_stab, z_stab, cx_gates,
                                   params_file)

    if not with_leakage:
        dets = nc.compile_detector_sampler().sample(shots=shots)
        return dets, nc

    params = load_surface_code_params(params_file)

    # 1. Raw measurement sample
    m_results = nc.compile_sampler().sample(shots=shots)

    # 2. Leakage simulate (over the BASE circuit — leakage states propagate
    #    through CX topology, independent of injected stim noise)
    affected_timelines = simulate_surface_code_leakage_vectorized(
        base_circuit, data_q, x_stab, z_stab, cx_gates, params, shots, batch_size)

    # 3. Per-measurement affected mask
    measurement_affected = extract_measurement_affected_vectorized(
        affected_timelines, data_q, x_stab, z_stab, rounds)

    # 4. Flip with prob 0.5
    flip_prob = 0.5
    rng = np.random.default_rng(42)
    rand = rng.random(m_results.shape).astype(np.float32)
    flip_mask = (measurement_affected == 1) & (rand < flip_prob)
    processed = m_results.copy()
    processed[flip_mask] = 1 - processed[flip_mask]

    # 5. Convert measurements -> detectors via stim m2d converter
    m2d = nc.compile_m2d_converter()
    dets, _ = m2d.convert(measurements=processed.astype(np.bool_),
                          separate_observables=True)
    return dets.astype(np.uint8), nc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--distance', type=int, default=5)
    ap.add_argument('--rounds', type=int, default=30)
    ap.add_argument('--shots', type=int, default=50000)
    ap.add_argument('--patch', type=str, default='q4_7')
    ap.add_argument('--classical-mult', type=float, default=None,
                    help='Override classical defect multiplier (default = spec)')
    ap.add_argument('--leakage-mult', type=float, default=None,
                    help='Override leakage defect multiplier (default = preset)')
    ap.add_argument('--q-frac', type=float, default=None,
                    help='Override qubit defect fraction')
    ap.add_argument('--cx-frac', type=float, default=None,
                    help='Override CX defect fraction')
    ap.add_argument('--mult-min', type=float, default=None,
                    help='Override defect mult_min')
    ap.add_argument('--mult-max', type=float, default=None,
                    help='Override defect mult_max')
    ap.add_argument('--mult-scale-with-d', action='store_true',
                    help='Auto-scale mult_max(d) = max(5, 8 + (d-5)*2): d=5→8, d=7→12, d=9→16')
    ap.add_argument('--tag', type=str, default='',
                    help='Suffix for output PNG (e.g. "D1" -> corrmat_full_d5_r30_D1.png)')
    args = ap.parse_args()

    HERE = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(HERE, 'tmp_test'), exist_ok=True)

    # Build the four configs
    print("Building 4 PAEMS configs (pure, +X2, +Leak2, +X2+Leak2)...")
    configs = {}
    cfg_specs = [('pure', False, False),
                 ('X2',   True,  False),
                 ('Leak2', False, True),
                 ('X2+Leak2', True, True)]
    for name, xt, lk in cfg_specs:
        out = os.path.join(HERE, 'tmp_test',
                           f'_full_{name}_d{args.distance}r{args.rounds}{("_"+args.tag) if args.tag else ""}.json')
        # Auto-scale mult_max with code distance if requested
        eff_mult_max = args.mult_max
        if args.mult_scale_with_d and eff_mult_max is None:
            eff_mult_max = max(5.0, 8.0 + (args.distance - 5) * 2.0)
            print(f"  [mult-scale-with-d] d={args.distance} -> mult_max={eff_mult_max}")
        make_config(HERE, args.distance, args.rounds, xt, lk, out,
                    classical_mult=args.classical_mult, leakage_mult=args.leakage_mult,
                    q_frac=args.q_frac, cx_frac=args.cx_frac,
                    mult_min=args.mult_min, mult_max=eff_mult_max)
        configs[name] = (out, xt, lk)

    # Sample each
    print(f"Sampling {args.shots} shots × 4 PAEMS configs ...")
    all_dets = {}
    nc_for_order = None
    for name, (path, xt, lk) in configs.items():
        dets, nc = sample_paems_full(args.distance, args.rounds, path,
                                     args.shots, with_leakage=lk)
        all_dets[name] = dets
        if nc_for_order is None:
            nc_for_order = nc
        print(f"  {name:<10}: {dets.shape}  xtalk={xt} leakage={lk}")

    # Real Willow
    print("Loading Real Willow ...")
    base = (rf"C:\Users\10124\Desktop\google dataset\google_105Q_surface_code_d3_d5_d7"
            rf"\d{args.distance}_at_{args.patch}\Z\r{args.rounds}")
    cir_real = stim.Circuit.from_file(os.path.join(base, "circuit_noisy_si1000.stim"))
    dets_real = stim.read_shot_data_file(
        path=os.path.join(base, "detection_events.b8"),
        format='b8', num_detectors=cir_real.num_detectors, bit_packed=False)

    # Reorder by qubit-block
    print("Reordering by qubit-block + aligning detector counts...")
    sort_p, qpd_p = detector_to_qubit_order(nc_for_order)
    sort_r, qpd_r = detector_to_qubit_order(cir_real)
    for name in all_dets:
        all_dets[name] = all_dets[name][:, sort_p]
    dets_real = dets_real[:, sort_r]
    n = min(min(d.shape[1] for d in all_dets.values()), dets_real.shape[1])
    for name in all_dets:
        all_dets[name] = all_dets[name][:, :n]
    dets_real = dets_real[:, :n]
    qpd_p, qpd_r = qpd_p[:n], qpd_r[:n]

    # Spitz matrices + hop>=2 cross-qubit filter
    print("Computing Spitz matrices + hop>=2 filter ...")
    G_p = cx_graph(nc_for_order); G_r = cx_graph(cir_real)
    sp_p = dict(nx.all_pairs_shortest_path_length(G_p))
    sp_r = dict(nx.all_pairs_shortest_path_length(G_r))

    paems_results = {}; paems_mat = {}
    for name, dets in all_dets.items():
        m = compute_pij_spitz(dets)
        paems_mat[name] = m
        paems_results[name] = filter_hop_ge2(m, qpd_p, sp_p)
    m_real = compute_pij_spitz(dets_real)
    real_xtalk = filter_hop_ge2(m_real, qpd_r, sp_r)

    # ---- Print summary ----
    names = ['pure', 'X2', 'Leak2', 'X2+Leak2']
    print()
    print(f"{'config':<12} {'mean':>10} {'max':>10} {'90th':>10} {'99th':>10} "
          f"{'top-10':>10} {'>1e-3':>10} {'>1e-2':>8}  defect_rate")
    print('-' * 100)
    for name in names:
        arr = paems_results[name]
        defect_rate = all_dets[name].mean() * 100
        print(f"{name:<12} {arr.mean():>10.5f} {arr.max():>10.5f} "
              f"{np.percentile(arr,90):>10.5f} {np.percentile(arr,99):>10.5f} "
              f"{np.sort(arr)[-10:].mean():>10.5f} "
              f"{(arr>1e-3).mean()*100:>9.2f}% {(arr>1e-2).mean()*100:>7.2f}%  "
              f"{defect_rate:>5.2f}%")
    print(f"{'Real':<12} {real_xtalk.mean():>10.5f} {real_xtalk.max():>10.5f} "
          f"{np.percentile(real_xtalk,90):>10.5f} {np.percentile(real_xtalk,99):>10.5f} "
          f"{np.sort(real_xtalk)[-10:].mean():>10.5f} "
          f"{(real_xtalk>1e-3).mean()*100:>9.2f}% {(real_xtalk>1e-2).mean()*100:>7.2f}%  "
          f"{dets_real.mean()*100:>5.2f}%")

    # ---- Plot 2x3: 5 correlation matrices + 1 distribution panel ----
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    vmax = np.percentile(np.concatenate([paems_mat[n].flatten() for n in names]
                                        + [m_real.flatten()]), 99.5)

    panels = [
        ((0, 0), paems_mat['pure'],     qpd_p, "PAEMS L2 pure"),
        ((0, 1), paems_mat['X2'],       qpd_p, "PAEMS L2 + X2"),
        ((0, 2), paems_mat['Leak2'],    qpd_p, "PAEMS L2 + Leak2"),
        ((1, 0), paems_mat['X2+Leak2'], qpd_p, "PAEMS L2 + X2 + Leak2"),
        ((1, 1), m_real,                qpd_r, f"Real Willow {args.patch}"),
    ]
    for (i, j), mat, qpd, title in panels:
        ax = axes[i][j]
        im = ax.matshow(mat, cmap='Reds', vmin=0, vmax=vmax)
        boundaries = [k - 0.5 for k in range(1, len(qpd)) if qpd[k] != qpd[k-1]]
        for b in boundaries:
            ax.axhline(b, color='gray', linewidth=0.2, alpha=0.5)
            ax.axvline(b, color='gray', linewidth=0.2, alpha=0.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Bottom-right: hop>=2 cross-qubit distribution
    ax = axes[1][2]
    bins = np.logspace(-6, -1, 50)
    colors = {'pure': 'gray', 'X2': 'C0', 'Leak2': 'C1',
              'X2+Leak2': 'C3', 'Real': 'C2'}
    for name in names:
        arr = paems_results[name]
        ax.hist(np.clip(arr, 1e-7, None), bins=bins, alpha=0.4,
                label=f'PAEMS {name}', color=colors[name],
                edgecolor='black', linewidth=0.3)
    ax.hist(np.clip(real_xtalk, 1e-7, None), bins=bins, alpha=0.4,
            label='Real Willow', color=colors['Real'],
            edgecolor='black', linewidth=0.3)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('p_ij (hop>=2 cross-qubit)')
    ax.set_ylabel('count')
    ax.set_title('hop>=2 cross-qubit distribution')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    suffix = f'_{args.tag}' if args.tag else ''
    out_png = os.path.join(HERE, f'corrmat_full_d{args.distance}_r{args.rounds}{suffix}.png')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_png}")


if __name__ == '__main__':
    main()
