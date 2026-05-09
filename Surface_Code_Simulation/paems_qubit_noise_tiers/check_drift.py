#!/usr/bin/env python3
"""
检查 Willow 真机数据是否有时间 drift：
  (A) 一个 shot 内 rounds 越往后是不是越差（intra-shot）
  (B) 50K shots 中越往后的 shot 是不是越差（inter-shot）

用 d=7 r=250 q6_7 Z 数据（最长 round count）。
"""
import os
import numpy as np
import stim
import matplotlib.pyplot as plt

base = r"C:\Users\10124\Desktop\google dataset\google_105Q_surface_code_d3_d5_d7\d7_at_q6_7\Z\r250"
cir = stim.Circuit.from_file(os.path.join(base, "circuit_noisy_si1000.stim"))
N_DET = cir.num_detectors
print(f"Loading {N_DET} detectors x 50000 shots...")

dets = stim.read_shot_data_file(
    path=os.path.join(base, "detection_events.b8"),
    format='b8', num_detectors=N_DET, bit_packed=False)
print(f"shape={dets.shape}, mean defect rate={dets.mean()*100:.3f}%")

# Detector layout: surface code d=7 Z basis r=250
#   per round: typically (d^2 - 1) / 2 = 24 ancilla detectors
#   total = 24 * 250 + final-layer detectors
# Use detector coordinates to bin by round (z time index)
det_coords = cir.get_detector_coordinates()
rounds_of_det = np.array([
    int(det_coords[d][2]) if len(det_coords[d]) > 2 else 0
    for d in range(N_DET)
])
n_rounds_max = rounds_of_det.max() + 1
print(f"detector round indices: 0..{n_rounds_max-1}")

# ---- (A) Intra-shot: defect rate vs round number ----
print("\n=== (A) Intra-shot drift: defect rate vs round number ===")
defect_per_round = []
for r in range(n_rounds_max):
    mask = (rounds_of_det == r)
    if mask.sum() == 0:
        defect_per_round.append(np.nan)
    else:
        defect_per_round.append(dets[:, mask].mean() * 100)
defect_per_round = np.array(defect_per_round)
print(f"  round 0       : {defect_per_round[0]:.3f}%   ({(rounds_of_det == 0).sum()} dets)")
print(f"  round 50      : {defect_per_round[50]:.3f}%")
print(f"  round 125     : {defect_per_round[125]:.3f}%")
print(f"  round 200     : {defect_per_round[200]:.3f}%")
print(f"  round {n_rounds_max-1:<7}: {defect_per_round[-1]:.3f}%   ({(rounds_of_det == n_rounds_max-1).sum()} dets, last layer is data-qubit final M)")
print(f"  rounds 1-249 mean: {defect_per_round[1:n_rounds_max-1].mean():.3f}%")
# Linear fit slope
mid = defect_per_round[1:n_rounds_max-1]
slope_intra, _ = np.polyfit(np.arange(len(mid)), mid, 1)
print(f"  linear slope (mid rounds): {slope_intra:+.5f} %/round")

# ---- (B) Inter-shot: defect rate vs shot index ----
print("\n=== (B) Inter-shot drift: defect rate vs shot index ===")
# Group every 1000 shots, get mean defect rate
SHOT_GROUP = 1000
n_groups = len(dets) // SHOT_GROUP
defect_per_group = []
for g in range(n_groups):
    s, e = g * SHOT_GROUP, (g + 1) * SHOT_GROUP
    defect_per_group.append(dets[s:e].mean() * 100)
defect_per_group = np.array(defect_per_group)
print(f"  shots 0-1k   : {defect_per_group[0]:.3f}%")
print(f"  shots 24-25k : {defect_per_group[24]:.3f}%")
print(f"  shots 49-50k : {defect_per_group[-1]:.3f}%")
print(f"  full mean    : {defect_per_group.mean():.3f}%   std={defect_per_group.std():.3f}%")
slope_inter, _ = np.polyfit(np.arange(n_groups), defect_per_group, 1)
print(f"  linear slope: {slope_inter:+.5f} %/1k_shots  (over 50 groups)")
total_change = slope_inter * (n_groups - 1)
print(f"  total drift over 50K shots: {total_change:+.3f}pp")

# ---- (C) Compare per-qubit defect rate stability ----
# If chip drifted, some qubits' defect rate would change between first 10K and last 10K
print("\n=== (C) Per-qubit drift: top-10 qubits with biggest first-10K vs last-10K change ===")
first10k = dets[:10000]
last10k = dets[-10000:]
det_drift = last10k.mean(axis=0) - first10k.mean(axis=0)
top10_idx = np.argsort(np.abs(det_drift))[-10:][::-1]
for i in top10_idx:
    print(f"  detector {i:4d} (round {rounds_of_det[i]:3d}): "
          f"first10K={first10k[:, i].mean()*100:.2f}%  "
          f"last10K={last10k[:, i].mean()*100:.2f}%  "
          f"drift={det_drift[i]*100:+.2f}pp")

# ---- Plot ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(np.arange(n_rounds_max), defect_per_round, 'o-', alpha=0.4, markersize=3)
ax.axhline(defect_per_round[1:n_rounds_max-1].mean(), color='gray', linestyle='--', alpha=0.5,
           label=f'mid-rounds mean={defect_per_round[1:n_rounds_max-1].mean():.2f}%')
ax.set_xlabel('round (within shot)')
ax.set_ylabel('defect rate (%)')
ax.set_title(f'(A) Intra-shot drift: slope={slope_intra:+.5f} %/round')
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(np.arange(n_groups) * SHOT_GROUP, defect_per_group, 'o-', alpha=0.6)
fit_y = slope_inter * np.arange(n_groups) + defect_per_group.mean() - slope_inter * (n_groups - 1) / 2
ax.plot(np.arange(n_groups) * SHOT_GROUP, fit_y, 'r--', alpha=0.6,
        label=f'linear fit: {slope_inter:+.4f}%/1k_shots')
ax.set_xlabel('shot index')
ax.set_ylabel(f'defect rate (% per {SHOT_GROUP}-shot group)')
ax.set_title(f'(B) Inter-shot drift: total {total_change:+.3f}pp over 50K')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'willow_drift_d7_r250.png')
plt.savefig(out_png, dpi=120, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out_png}")
