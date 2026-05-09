#!/usr/bin/env python3
"""
v2: detector 按 qubit 重排序 + 多轮（r=30），输出跟 Google 论文同款的相关矩阵图。

Google 图特征：
  - x/y 轴每行/列是一个 ancilla qubit（不是单个 detector）
  - 同 qubit 的多轮 detector 集中在一个 block 内
  - block 之间有 gridline 分隔
  - 强对角线（block 内 round-to-round 关联）+ off-diagonal hot dots（相关 ancilla 对）
"""
import sys, os, numpy as np, subprocess, json
import matplotlib.pyplot as plt
import argparse

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
    """For a Stim circuit, return (sort_indices, qubit_id_per_detector, qubit_xy_per_detector).
    sort_indices reorders detectors by (qubit_id, time-coord)."""
    det_coords = circuit.get_detector_coordinates()  # dict[idx -> coord]
    # Get qubit coords (from QUBIT_COORDS instructions in the original circuit)
    qubit_coords = {}
    for inst in circuit:
        if inst.name == "QUBIT_COORDS":
            qid = inst.targets_copy()[0].value
            qubit_coords[qid] = tuple(inst.gate_args_copy())

    # For each detector, find nearest qubit by (x, y) distance
    det_to_qubit = {}
    for did, coord in det_coords.items():
        if len(coord) >= 2:
            dx, dy = coord[0], coord[1]
            t = coord[2] if len(coord) > 2 else 0
        else:
            dx, dy, t = 0, 0, 0
        best_q, best_d = None, float('inf')
        for q, (qx, qy) in qubit_coords.items():
            d = (dx - qx) ** 2 + (dy - qy) ** 2
            if d < best_d:
                best_d, best_q = d, q
        det_to_qubit[did] = (best_q, t)

    # Sort by (qubit_id, time) — gives qubit-blocks
    sorted_det_ids = sorted(det_coords.keys(), key=lambda d: (det_to_qubit[d][0], det_to_qubit[d][1]))
    sort_indices = np.array(sorted_det_ids)
    qubit_per_det = np.array([det_to_qubit[d][0] for d in sorted_det_ids])
    return sort_indices, qubit_per_det


def sample_paems(d, r, params_file, shots):
    c, dq, xs, zs, cx = generate_surface_code_circuit(d, r, 'z')
    nc = inject_surface_code_noise(c, dq, xs, zs, cx, params_file)
    dets = nc.compile_detector_sampler().sample(shots=shots)
    return dets, nc


def load_real_willow(d, patch, basis, rounds):
    base = (rf"C:\Users\10124\Desktop\google dataset\google_105Q_surface_code_d3_d5_d7"
            rf"\d{d}_at_{patch}\{basis}\r{rounds}")
    cir = stim.Circuit.from_file(os.path.join(base, "circuit_noisy_si1000.stim"))
    dets = stim.read_shot_data_file(
        path=os.path.join(base, "detection_events.b8"),
        format='b8', num_detectors=cir.num_detectors, bit_packed=False)
    return dets, cir


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


def plot_with_blocks(ax, mat, qubit_per_det, title, vmax):
    im = ax.matshow(mat, cmap='Reds', vmin=0, vmax=vmax, interpolation='nearest')
    ax.set_title(title, fontsize=10)
    # Add gridlines between qubit blocks
    boundaries = []
    for i in range(1, len(qubit_per_det)):
        if qubit_per_det[i] != qubit_per_det[i - 1]:
            boundaries.append(i - 0.5)
    for b in boundaries:
        ax.axhline(b, color='gray', linewidth=0.3, alpha=0.6)
        ax.axvline(b, color='gray', linewidth=0.3, alpha=0.6)
    ax.set_xticks([]); ax.set_yticks([])
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--distance', type=int, default=5)
    ap.add_argument('--rounds', type=int, default=30)
    ap.add_argument('--shots', type=int, default=50000)
    ap.add_argument('--patch', type=str, default='q4_7')
    args = ap.parse_args()

    HERE = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(HERE, 'tmp_test'), exist_ok=True)

    # Generate PAEMS configs
    paems_pure = os.path.join(HERE, 'tmp_test', f'p_pure_d{args.distance}_r{args.rounds}.json')
    paems_x2 = os.path.join(HERE, 'tmp_test', f'p_L2X2_d{args.distance}_r{args.rounds}.json')
    paems_l4x4 = os.path.join(HERE, 'tmp_test', f'p_L4X4_d{args.distance}_r{args.rounds}.json')
    gen_config(2, None, args.distance, args.rounds, paems_pure, HERE)
    gen_config(2, 'crosstalk_X2.json', args.distance, args.rounds, paems_x2, HERE)
    gen_config(4, 'crosstalk_X4.json', args.distance, args.rounds, paems_l4x4, HERE)

    print(f"Sampling PAEMS d={args.distance} r={args.rounds} ...")
    dets_pure, nc_pure = sample_paems(args.distance, args.rounds, paems_pure, args.shots)
    dets_x2, nc_x2 = sample_paems(args.distance, args.rounds, paems_x2, args.shots)
    dets_l4x4, nc_l4x4 = sample_paems(args.distance, args.rounds, paems_l4x4, args.shots)

    print(f"Loading real Willow d={args.distance} {args.patch} r={args.rounds} ...")
    dets_real, cir_real = load_real_willow(args.distance, args.patch, 'Z', args.rounds)

    # Reorder detectors by (qubit, time)
    print("Reordering by qubit-then-time...")
    sort_idx_p, qpd_p = detector_to_qubit_order(nc_pure)
    sort_idx_r, qpd_r = detector_to_qubit_order(cir_real)

    dets_pure = dets_pure[:, sort_idx_p]
    dets_x2 = dets_x2[:, sort_idx_p]
    dets_l4x4 = dets_l4x4[:, sort_idx_p]
    dets_real = dets_real[:, sort_idx_r]

    print(f"  PAEMS detectors per qubit: ~{len(qpd_p) // len(np.unique(qpd_p))}")
    print(f"  Real  detectors per qubit: ~{len(qpd_r) // len(np.unique(qpd_r))}")

    print("Computing correlation matrices...")
    m_pure = compute_pij_spitz(dets_pure)
    m_x2 = compute_pij_spitz(dets_x2)
    m_l4x4 = compute_pij_spitz(dets_l4x4)
    m_real = compute_pij_spitz(dets_real)

    # Larger figure for detail
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    configs = [
        ((0, 0), m_pure, qpd_p, f"PAEMS L2 (pure) d={args.distance} r={args.rounds}"),
        ((0, 1), m_x2, qpd_p, f"PAEMS L2 + X2 d={args.distance} r={args.rounds}"),
        ((1, 0), m_real, qpd_r, f"Real Willow {args.patch} d={args.distance} r={args.rounds}"),
        ((1, 1), m_l4x4, qpd_p, f"PAEMS L4 + X4 d={args.distance} r={args.rounds}"),
    ]
    # Use percentile-based vmax to avoid L4+X4 dominance breaking colorbar
    vmax = np.percentile(np.concatenate([m_pure.flatten(), m_x2.flatten(), m_real.flatten()]), 99.5)
    for (i, j), mat, qpd, title in configs:
        ax = axes[i][j]
        plot_with_blocks(ax, mat, qpd, title, vmax)

    plt.tight_layout()
    out_png = os.path.join(HERE, f'correlation_matrix_v2_d{args.distance}_r{args.rounds}.png')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_png}")

    print()
    print(f"{'Config':<48} {'mean':>10} {'max':>10} {'top-10':>10}")
    print('-' * 80)
    for (_, _), mat, _, title in configs:
        nz = mat[mat > 0]
        top10 = np.sort(mat.flatten())[-10:].mean()
        print(f"{title:<48} {nz.mean():>10.5f} {mat.max():>10.5f} {top10:>10.5f}")


if __name__ == '__main__':
    main()
