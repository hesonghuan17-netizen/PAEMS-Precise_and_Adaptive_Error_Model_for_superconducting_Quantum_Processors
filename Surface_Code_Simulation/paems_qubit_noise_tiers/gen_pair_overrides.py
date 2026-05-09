#!/usr/bin/env python3
"""
独立的 crosstalk overlay 工具 — 跟 L1-L4 classical noise 解耦。

输入：classical PAEMS JSON（无 frequency_mhz, 无 crosstalk_pairs）
输出：JSON 多了 frequency_mhz per qubit + crosstalk_pairs

统一 χ 公式：
    χ(i, j) = A × max(Lorentzian(Δf, W), zz_floor) × spatial_factor(hop)
    Lorentzian(Δf, W) = 1 / (1 + (Δf/W)²)

频率分配：cumsum 模型
    gaps = N(mean_gap_mhz, std_gap_mhz),  i=1..n-1
    positions = cumsum(gaps), centered at chip middle
    Auto-scale: if total_spread > max_total_spread_mhz, all gaps shrunk to fit.

参数全部从 --crosstalk-config preset.json 或 CLI 提供。
不传任何串扰参数 → 输出空 crosstalk_pairs（无串扰）。

用法示例：
  # 用 preset
  python gen_pair_overrides.py --in class.json --out full.json --merge \\
      --crosstalk-config crosstalk_X2.json

  # 全 CLI
  python gen_pair_overrides.py --in class.json --out full.json --merge \\
      --mean-gap 15 --std-gap 5 --max-spread 1000 \\
      --A 5e-3 --W 5 --zz-floor 0.01 \\
      --spatial-factors '{"2":1.0,"3":0.3,"4":0.1,"5":0.03}'

  # 不带任何 crosstalk 参数 → 无串扰
  python gen_pair_overrides.py --in class.json --out same.json --merge

Preset JSON 格式：
  {
    "mean_gap_mhz": 15, "std_gap_mhz": 5, "max_total_spread_mhz": 1000,
    "A_peak": 5e-3, "W_mhz": 5, "zz_floor": 0.01,
    "spatial_factors": {"2": 1.0, "3": 0.3, "4": 0.1, "5": 0.03}
  }
"""
import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np


# ---------- Frequency assignment (cumsum + auto-scale + bipartite split) -----

def assign_frequencies_cumsum(qubits_ids, cx_gates, mean_gap, std_gap,
                              max_total_spread, center_freq_mhz, rng):
    """Sample n-1 gaps ~ N(mean_gap, std_gap), cumsum to get sorted positions,
    then split positions into 2 halves: bipartite color 0 → lower half,
    color 1 → upper half. This guarantees 1-hop CX-direct neighbors are always
    in different freq halves (avoids physically-close + freq-close pathology).

    Auto-scale: if total spread > max_total_spread, gaps are shrunk to fit.

    Returns: dict {qubit_id: freq_mhz}, info_dict
    """
    n = len(qubits_ids)
    if n < 2:
        return {qubits_ids[0]: center_freq_mhz} if n == 1 else {}, {}

    # --- Step 1: bipartite coloring on CX-adjacency graph ---
    G = nx.Graph()
    G.add_nodes_from(qubits_ids)
    for g in cx_gates.values():
        G.add_edge(int(g['control']), int(g['target']))

    if nx.is_bipartite(G):
        color = nx.bipartite.color(G)
        color_0 = [q for q in qubits_ids if color.get(q, 0) == 0]
        color_1 = [q for q in qubits_ids if color.get(q, 0) == 1]
        is_bipartite = True
    else:
        # Non-bipartite: fall back to greedy + alternate-bucket assignment
        gc = nx.coloring.greedy_color(G, strategy='smallest_last')
        # Sort by color, alternate halves
        ordered = sorted(qubits_ids, key=lambda q: gc.get(q, 0))
        color_0 = ordered[::2]
        color_1 = ordered[1::2]
        is_bipartite = False

    # --- Step 2: sample gaps + auto-scale ---
    gaps = rng.normal(mean_gap, std_gap, size=n - 1)
    gaps = np.clip(gaps, 0.1, None)
    total = float(gaps.sum())

    info = {
        'mean_gap_mhz_target': mean_gap,
        'mean_gap_mhz_realized': float(np.mean(gaps)),
        'total_spread_mhz_pre_scale': total,
        'scaled': False,
        'scale_factor': 1.0,
        'bipartite': is_bipartite,
        'color_0_count': len(color_0),
        'color_1_count': len(color_1),
    }
    if total > max_total_spread:
        scale = max_total_spread / total
        gaps *= scale
        info['scaled'] = True
        info['scale_factor'] = scale
        info['mean_gap_mhz_realized'] = float(np.mean(gaps))
        info['total_spread_mhz_post_scale'] = float(gaps.sum())

    positions = np.concatenate(([0.0], np.cumsum(gaps)))
    positions = positions - positions.mean() + center_freq_mhz

    # --- Step 3: assign positions, color 0 → lower half, color 1 → upper half ---
    n_color_0 = len(color_0)
    rng.shuffle(color_0)
    rng.shuffle(color_1)

    freqs = {}
    for i, q in enumerate(color_0):
        freqs[q] = float(positions[i])              # lower-freq half
    for i, q in enumerate(color_1):
        freqs[q] = float(positions[n_color_0 + i])  # upper-freq half

    return freqs, info


# ---------------- Crosstalk classification ------------------

def chi_unified(freq_diff_mhz, hop, cfg):
    """Unified χ formula:  χ = A × max(Lorentzian, zz_floor) × spatial_factor(hop).

    Returns float χ or None if pair contributes nothing.
    """
    if hop == 1:
        return None  # CX-direct, skipped
    s = cfg['spatial_factors'].get(int(hop), cfg['spatial_factors'].get(str(hop)))
    if s is None or s <= 0:
        return None
    if freq_diff_mhz is None:
        # No frequency info — only ZZ floor contributes
        lorentz_or_floor = cfg['zz_floor']
    else:
        lorentz = 1.0 / (1.0 + (freq_diff_mhz / cfg['W_mhz']) ** 2)
        lorentz_or_floor = max(lorentz, cfg['zz_floor'])
    chi = cfg['A_peak'] * lorentz_or_floor * s
    if chi <= 0:
        return None
    return chi


def classify_label(freq_diff_mhz, hop, chi, cfg):
    """Coarse human-readable label for analysis."""
    if freq_diff_mhz is None:
        return f"zz_only_hop{hop}"
    lorentz = 1.0 / (1.0 + (freq_diff_mhz / cfg['W_mhz']) ** 2)
    if lorentz < cfg['zz_floor']:
        return f"zz_floor_hop{hop}"          # floor dominates
    if freq_diff_mhz < 1.0:
        return f"freq_severe_hop{hop}"
    if freq_diff_mhz < 10.0:
        return f"freq_collision_hop{hop}"
    return f"freq_close_hop{hop}"


# ---------------- Pair generation ------------------

def generate_pairs(qubits, cx_gates, cfg):
    """Compute χ for all qubit pairs using unified formula."""
    G = nx.Graph()
    for qid_str in qubits:
        G.add_node(int(qid_str))
    for g in cx_gates.values():
        G.add_edge(int(g['control']), int(g['target']))
    sp = dict(nx.all_pairs_shortest_path_length(G))

    qids = sorted(int(k) for k in qubits.keys())
    pairs = {}
    n_skipped = 0

    for ii, qi in enumerate(qids):
        fi = qubits[str(qi)].get('frequency_mhz')
        for qj in qids[ii + 1:]:
            fj = qubits[str(qj)].get('frequency_mhz')

            hop = sp.get(qi, {}).get(qj, float('inf'))
            if hop == 1:
                n_skipped += 1
                continue
            if hop == float('inf'):
                # Disconnected pair (e.g. XZZX boundary aux qubits not in
                # any 2-qubit gate) — no crosstalk path, skip
                n_skipped += 1
                continue

            diff = abs(fi - fj) if (fi is not None and fj is not None) else None
            chi = chi_unified(diff, hop, cfg)
            if chi is None:
                n_skipped += 1
                continue

            label = classify_label(diff, hop, chi, cfg)
            entry = {
                "strength": float(chi),
                "type": label,
                "hop_distance": int(hop) if hop != float('inf') else None,
            }
            if diff is not None:
                entry["freq_diff_mhz"] = round(diff, 4)
            pairs[f"{qi}-{qj}"] = entry

    return pairs, n_skipped


def summarize(pairs):
    by_type = {}
    for v in pairs.values():
        by_type.setdefault(v['type'], []).append(v['strength'])
    print(f"  By type:")
    print(f"    {'type':<28} {'count':>6} {'chi range':>26}")
    print(f"    {'-' * 60}")
    for t in sorted(by_type):
        vals = by_type[t]
        if len(vals) == 1:
            print(f"    {t:<28} {len(vals):>6}   {vals[0]:.2e}")
        else:
            print(f"    {t:<28} {len(vals):>6}   {min(vals):.2e} — {max(vals):.2e}")
    print(f"    {'TOTAL':<28} {len(pairs):>6}")


# ---------------- CLI ------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--in', dest='in_file', required=True)
    ap.add_argument('--out', dest='out_file', required=True)
    ap.add_argument('--merge', action='store_true',
                    help='Merge into input JSON')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--crosstalk-config', type=str, default=None,
                    help='Preset JSON; CLI args below override its values')
    ap.add_argument('--center-freq', type=float, default=5500.0,
                    help='Chip center frequency (MHz, default 5500)')
    # Frequency model
    ap.add_argument('--mean-gap', type=float, default=None)
    ap.add_argument('--std-gap', type=float, default=None)
    ap.add_argument('--max-spread', type=float, default=None)
    # Drive Lorentzian + ZZ floor
    ap.add_argument('--A', type=float, default=None,
                    help='Peak chi (Lorentzian amplitude)')
    ap.add_argument('--W', type=float, default=None,
                    help='Lorentzian half-width (MHz)')
    ap.add_argument('--zz-floor', type=float, default=None,
                    help='Always-on floor ratio (multiplied by A)')
    # Spatial factor (passed as JSON string)
    ap.add_argument('--spatial-factors', type=str, default=None,
                    help="JSON like '{\"2\":1.0,\"3\":0.3,\"4\":0.1,\"5\":0.03}'")
    args = ap.parse_args()

    cfg_required = {
        'mean_gap_mhz': args.mean_gap,
        'std_gap_mhz': args.std_gap,
        'max_total_spread_mhz': args.max_spread,
        'A_peak': args.A,
        'W_mhz': args.W,
        'zz_floor': args.zz_floor,
        'spatial_factors': (json.loads(args.spatial_factors)
                            if args.spatial_factors else None),
    }

    if args.crosstalk_config:
        with open(args.crosstalk_config, encoding='utf-8') as f:
            preset = json.load(f)
        for k, v in preset.items():
            if k in cfg_required and cfg_required[k] is None:
                cfg_required[k] = v

    # Normalize spatial_factors keys to int
    if cfg_required['spatial_factors'] is not None:
        sf = cfg_required['spatial_factors']
        cfg_required['spatial_factors'] = {int(k): float(v) for k, v in sf.items()}

    n_provided = sum(1 for v in cfg_required.values() if v is not None)
    crosstalk_off = (n_provided == 0)
    if not crosstalk_off and n_provided < len(cfg_required):
        missing = [k for k, v in cfg_required.items() if v is None]
        ap.error(
            f"Partial crosstalk spec — missing: {missing}. "
            "Specify ALL or NONE."
        )

    cfg = cfg_required

    with open(args.in_file, encoding='utf-8') as f:
        data = json.load(f)
    qubits = data.get('qubits', {})
    cx_gates = data.get('cx_gates', {})
    if not qubits or not cx_gates:
        raise SystemExit("ERROR: input JSON missing qubits or cx_gates")

    print(f"=== Crosstalk overlay ===")
    print(f"Input: {args.in_file} (n_qubits={len(qubits)}, n_cx={len(cx_gates)})")

    if crosstalk_off:
        print(f"\n[No crosstalk parameters specified] — emitting empty crosstalk_pairs.")
        if args.merge:
            data['crosstalk_pairs'] = {}
            out_data = data
        else:
            out_data = {"crosstalk_pairs": {}}
        with open(args.out_file, 'w', encoding='utf-8') as f:
            json.dump(out_data, f, indent=2)
        print(f"Wrote: {args.out_file}")
        return

    rng = np.random.default_rng(args.seed)

    # Step 1: assign frequencies (bipartite-aware cumsum)
    qids = sorted(int(k) for k in qubits.keys())
    freqs, info = assign_frequencies_cumsum(
        qids, cx_gates, cfg['mean_gap_mhz'], cfg['std_gap_mhz'],
        cfg['max_total_spread_mhz'], args.center_freq, rng
    )
    for qid, f in freqs.items():
        qubits[str(qid)]['frequency_mhz'] = f

    fs = sorted(freqs.values())
    print(f"\nFrequency model: cumsum + auto-scale + bipartite split")
    print(f"  target mean_gap={cfg['mean_gap_mhz']} std_gap={cfg['std_gap_mhz']} MHz")
    print(f"  realized mean_gap={info['mean_gap_mhz_realized']:.2f} MHz")
    print(f"  total spread={fs[-1] - fs[0]:.0f} MHz, range [{fs[0]:.0f}, {fs[-1]:.0f}]")
    print(f"  bipartite={info['bipartite']}  color_0={info['color_0_count']}  color_1={info['color_1_count']}")
    if info['scaled']:
        print(f"  [Auto-scaled by {info['scale_factor']:.3f}x to fit max {cfg['max_total_spread_mhz']} MHz]")

    # Step 2: generate pairs
    print(f"\nχ formula: A × max(Lorentzian, zz_floor) × spatial_factor(hop)")
    print(f"  A={cfg['A_peak']:.0e}, W={cfg['W_mhz']} MHz, zz_floor={cfg['zz_floor']}")
    print(f"  spatial_factors={cfg['spatial_factors']}")
    pairs, n_skipped = generate_pairs(qubits, cx_gates, cfg)

    n_total = len(qubits) * (len(qubits) - 1) // 2
    print(f"\nResult:")
    print(f"  total possible pairs: {n_total}")
    print(f"  skipped (1-hop or beyond cutoff): {n_skipped}")
    print(f"  pairs in output: {len(pairs)}")
    print()
    summarize(pairs)
    print()

    if args.merge:
        data['qubits'] = qubits
        data['crosstalk_pairs'] = pairs
        out = data
    else:
        out = {"qubits_with_freq": qubits, "crosstalk_pairs": pairs}

    with open(args.out_file, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f"Wrote: {args.out_file}")


if __name__ == '__main__':
    main()
