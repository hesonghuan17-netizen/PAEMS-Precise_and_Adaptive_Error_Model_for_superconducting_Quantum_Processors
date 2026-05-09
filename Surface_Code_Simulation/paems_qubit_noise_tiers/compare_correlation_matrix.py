#!/usr/bin/env python3
"""
对比 PAEMS L2+X2 跟真机 Willow 的 detector 相关矩阵。

用 Spitz 公式（PAEMS 同款）：
    p_ij = (P(x_i × x_j) - P(x_i)×P(x_j)) / ((1-2 P(x_i))(1-2 P(x_j)))

生成 4 张相关矩阵图：
  1. PAEMS L2 (pure, no crosstalk)
  2. PAEMS L2 + X2 (with SOTA crosstalk)
  3. Real Willow d=3 r=10 q4_5 patch
  4. PAEMS L4 + X4 (extreme worst — 应该看到强关联)
"""
import sys, os, numpy as np
import matplotlib.pyplot as plt
import matplotlib

PAEMS_SC = r"C:\PAEMS-Precise_and_Adaptive_Error_Model_for_superconducting_Quantum_Processors\Surface_Code_Simulation"
sys.path.insert(0, PAEMS_SC)

from inject_basic_noise import inject_surface_code_noise
from surface_code_generate_circuits import generate_surface_code_circuit
import stim


def compute_pij_spitz(dets):
    """Spitz pairwise correlation matrix from binary detection events.

    dets: shape (shots, n_dets)
    Returns: (n_dets, n_dets) symmetric, diagonal zeroed
    """
    dets = dets.astype(np.float64)
    n_dets = dets.shape[1]
    p_x = dets.mean(axis=0)
    p_xy = (dets.T @ dets) / dets.shape[0]   # (n_dets, n_dets)
    xi = p_x[:, None]
    xj = p_x[None, :]
    denom = (1 - 2 * xi) * (1 - 2 * xj)
    numer = p_xy - xi * xj
    with np.errstate(divide='ignore', invalid='ignore'):
        p_ij = np.where(np.abs(denom) > 1e-12, numer / denom, 0.0)
    np.fill_diagonal(p_ij, 0.0)
    return np.clip(p_ij, 0, None)   # show only positive correlations


def sample_paems(d, r, params_file, shots):
    c, dq, xs, zs, cx = generate_surface_code_circuit(d, r, 'z')
    nc = inject_surface_code_noise(c, dq, xs, zs, cx, params_file)
    return nc.compile_detector_sampler().sample(shots=shots)


def load_real_willow(d, patch, basis='Z', rounds=10):
    base = (rf"C:\Users\10124\Desktop\google dataset\google_105Q_surface_code_d3_d5_d7"
            rf"\d{d}_at_{patch}\{basis}\r{rounds}")
    cir = stim.Circuit.from_file(os.path.join(base, "circuit_noisy_si1000.stim"))
    return stim.read_shot_data_file(
        path=os.path.join(base, "detection_events.b8"),
        format='b8', num_detectors=cir.num_detectors, bit_packed=False)


# Configs (parametric)
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--distance', type=int, default=3)
ap.add_argument('--rounds', type=int, default=10)
ap.add_argument('--shots', type=int, default=50000)
ap.add_argument('--patch', type=str, default='q4_5',
                help='Real Willow patch (d3: q4_5/q10_7/...; d5: q4_7/q6_5/q6_9/q8_7; d7: q6_7)')
args = ap.parse_args()
DISTANCE = args.distance
ROUNDS = args.rounds
SHOTS = args.shots
HERE = os.path.dirname(os.path.abspath(__file__))

# Generate PAEMS configs first
import subprocess, json
def gen(level, xtalk_preset, out):
    class_path = os.path.join(HERE, 'tmp_test', f'tmp_class_L{level}.json')
    subprocess.run([sys.executable, os.path.join(HERE, 'gen_level_params.py'),
                    '--level', str(level), '--distance', str(DISTANCE),
                    '--rounds', str(ROUNDS), '--seed', '42', '--out', class_path],
                   capture_output=True, check=True)
    if xtalk_preset:
        subprocess.run([sys.executable, os.path.join(HERE, 'gen_pair_overrides.py'),
                        '--in', class_path, '--out', out, '--merge',
                        '--crosstalk-config', os.path.join(HERE, 'crosstalk_presets', xtalk_preset)],
                       capture_output=True, check=True)
    else:
        # Just copy with empty crosstalk
        with open(class_path, encoding='utf-8') as f:
            d = json.load(f)
        d['crosstalk_pairs'] = {}
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(d, f, indent=2)


os.makedirs(os.path.join(HERE, 'tmp_test'), exist_ok=True)

paems_pure = os.path.join(HERE, 'tmp_test', 'p_L2_pure.json')
paems_x2 = os.path.join(HERE, 'tmp_test', 'p_L2_X2.json')
paems_l4x4 = os.path.join(HERE, 'tmp_test', 'p_L4_X4.json')
gen(2, None, paems_pure)
gen(2, 'crosstalk_X2.json', paems_x2)
gen(4, 'crosstalk_X4.json', paems_l4x4)

print("Sampling PAEMS configurations...")
dets_pure = sample_paems(DISTANCE, ROUNDS, paems_pure, SHOTS)
dets_x2 = sample_paems(DISTANCE, ROUNDS, paems_x2, SHOTS)
dets_l4x4 = sample_paems(DISTANCE, ROUNDS, paems_l4x4, SHOTS)
print(f"Loading real Willow d={DISTANCE} {args.patch}...")
dets_real = load_real_willow(DISTANCE, args.patch, 'Z', ROUNDS)

print(f"All shapes: PAEMS_pure {dets_pure.shape}  X2 {dets_x2.shape}  L4X4 {dets_l4x4.shape}  real {dets_real.shape}")

print("Computing correlation matrices...")
m_pure = compute_pij_spitz(dets_pure)
m_x2 = compute_pij_spitz(dets_x2)
m_l4x4 = compute_pij_spitz(dets_l4x4)
m_real = compute_pij_spitz(dets_real)

# Plot 4 matrices side by side
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
configs = [
    (m_pure, f"PAEMS L2 (pure, no crosstalk) d={DISTANCE}"),
    (m_x2, f"PAEMS L2 + X2 (SOTA crosstalk) d={DISTANCE}"),
    (m_real, f"Real Willow d={DISTANCE} r={ROUNDS} {args.patch}"),
    (m_l4x4, f"PAEMS L4 + X4 (worst) d={DISTANCE}"),
]
vmax = max(m.max() for m, _ in configs[:3])  # exclude L4+X4 from vmax to keep first 3 on same scale
for ax, (m, title) in zip(axes, configs):
    im = ax.matshow(m, cmap='Reds', vmin=0, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('detector index')
    ax.set_ylabel('detector index')
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
out_png = os.path.join(HERE, f'correlation_matrix_compare_d{DISTANCE}.png')
plt.savefig(out_png, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {out_png}")

# Also print summary stats
print()
print(f"{'Config':<40} {'mean p_ij':>12} {'max p_ij':>12} {'top-10 p_ij':>14}")
print('-' * 80)
for m, title in configs:
    nonzero = m[m > 0]
    top10 = np.sort(m.flatten())[-10:].mean()
    print(f"{title:<40} {nonzero.mean():>12.5f} {m.max():>12.5f} {top10:>14.5f}")
