#!/usr/bin/env python3
"""
按 4-tier 噪声等级生成 PAEMS 表面码参数 JSON。

用法：
    python gen_level_params.py --level 2 --distance 3 --rounds 3 --seed 42 \
        --out samples/d3_level2.json

支持 level: 1, 2, 3, 4
依赖: numpy, stim
（若需自动确定 cx_gates 拓扑会用到 PAEMS 的 surface_code_generate_circuits.py
 否则用纯 Stim 默认 d×r 表面码电路自己解析）
"""
import argparse
import json
import os
import sys
import numpy as np


def load_spec(spec_path):
    with open(spec_path, encoding='utf-8') as f:
        return json.load(f)


def sample_frequencies_random(rng, freq_spec, n_qubits, max_attempts=10000):
    """Random Gaussian sampling with optional min_gap enforcement (L4 Wukong-style)."""
    mean = freq_spec.get('mean_mhz', 5500.0)
    std = freq_spec.get('std_mhz', 200.0)
    min_gap = freq_spec.get('min_gap_mhz', 0.0)

    if min_gap <= 0:
        return list(rng.normal(mean, std, size=n_qubits))

    freqs, attempts = [], 0
    while len(freqs) < n_qubits and attempts < max_attempts:
        cand = float(rng.normal(mean, std))
        if all(abs(cand - f) >= min_gap for f in freqs):
            freqs.append(cand)
        attempts += 1
    if len(freqs) < n_qubits:
        print(f"[WARN] sample_frequencies_random: could only place {len(freqs)}/{n_qubits} "
              f"with min_gap={min_gap}; filling rest without constraint")
        freqs.extend(rng.normal(mean, std, size=n_qubits - len(freqs)).tolist())
    return freqs


def sample_frequencies_groups(rng, freq_spec, all_qubits, cx_pairs):
    """Frequency-group / graph-coloring model (Sycamore / Heron-style).

    1. Build CX adjacency graph
    2. Color it (bipartite-2 if possible, else greedy). 1-hop neighbors
       are guaranteed to have different colors → no severe drive collision
       on CX-direct pairs.
    3. If spec.n_groups > chromatic, subdivide each color randomly so the
       total number of frequency tiles equals n_groups (more diversity).
    4. Each tile gets a base frequency uniformly spread in [base_min, base_max].
    5. Add per-qubit jitter ~N(0, jitter_std).

    Returns: (freqs_dict {qubit_id: freq_mhz}, color_dict {qubit_id: tile_id})
    """
    import networkx as nx
    n_groups_spec = int(freq_spec.get('n_groups', 0))  # 0 = use chromatic only
    base_min = float(freq_spec.get('base_min_mhz', 5300.0))
    base_max = float(freq_spec.get('base_max_mhz', 5700.0))
    jitter_std = float(freq_spec.get('jitter_std_mhz', 20.0))

    G = nx.Graph()
    G.add_nodes_from(all_qubits)
    for c, t in cx_pairs:
        G.add_edge(c, t)

    if nx.is_bipartite(G):
        color = nx.bipartite.color(G)              # always 2 colors, balanced
    else:
        color = nx.coloring.greedy_color(G, strategy='smallest_last')
    chromatic = max(color.values()) + 1 if color else 1

    # If user wants more tiles than chromatic, subdivide each color randomly.
    # Each color → groups_per_color sub-tiles. CX pairs (different color) still
    # see different tiles, freq diff guaranteed by spacing.
    n_use = max(chromatic, n_groups_spec) if n_groups_spec > 0 else chromatic
    if n_use > chromatic:
        groups_per_color = (n_use + chromatic - 1) // chromatic  # ceil
        n_use = groups_per_color * chromatic                      # round up
        new_color = {}
        for q, c in color.items():
            sub = int(rng.integers(0, groups_per_color))
            new_color[q] = c * groups_per_color + sub
        color = new_color

    if n_use == 1:
        base_freqs = [(base_min + base_max) / 2]
    else:
        base_freqs = [base_min + i * (base_max - base_min) / (n_use - 1)
                      for i in range(n_use)]

    # Within each tile: distribute qubits UNIFORMLY across +/- tile_width/2,
    # then add tiny residual_jitter for realistic imperfection.
    # tile_width = total intra-tile spread; jitter_std = small residual.
    tile_width = float(freq_spec.get('tile_width_mhz', 60.0))
    residual_jitter = float(freq_spec.get('residual_jitter_mhz', jitter_std * 0.1))

    # Group qubits by color
    groups = {}
    for q in all_qubits:
        groups.setdefault(color.get(q, 0), []).append(q)

    freqs_dict = {}
    for tile_id, qs_in_tile in groups.items():
        base = base_freqs[tile_id]
        n = len(qs_in_tile)
        if n == 1:
            offsets = [0.0]
        else:
            # Uniform spacing across [-tile_width/2, +tile_width/2]
            offsets = [(i - (n - 1) / 2) * tile_width / n for i in range(n)]
        # Randomize which qubit gets which slot (cosmetic, doesn't change distribution)
        rng.shuffle(qs_in_tile)
        for q, off in zip(qs_in_tile, offsets):
            freqs_dict[q] = float(base + off + rng.normal(0, residual_jitter))
    return freqs_dict, color


def sample_qubit(rng, spec_q, n_qubits, freq_spec=None,
                 all_qubits=None, cx_pairs=None, hot_q_mults=None):
    """对 n_qubits 个 qubit 各采一组参数；返回 dict[str, dict].

    hot_q_mults (dict[int, dict[str, float]]): {qubit_id_1based: {param_name: mult}}.
    Each hot qubit has its OWN per-param multipliers (heterogeneous defect overlay).
    Params not in dict are not degraded.
    """
    if hot_q_mults is None:
        hot_q_mults = {}
    freqs_list = [None] * n_qubits
    out = {}
    for q in range(1, n_qubits + 1):
        # When all_qubits passed, use it for frequency lookup; else 1-based index
        freq_for_this = freqs_list[q - 1] if (q - 1) < len(freqs_list) else None
        _ = freq_for_this  # placed for clarity, used below
        # 时间相关：normal + floor at min
        t1 = max(spec_q['t1']['min'],
                 rng.normal(spec_q['t1']['mean'], spec_q['t1']['std']))
        # T2 must be <= 2*T1 if rule says so
        t2_raw = max(spec_q['t2']['min'],
                     rng.normal(spec_q['t2']['mean'], spec_q['t2']['std']))
        if spec_q['t2'].get('max_rule') == '2*t1':
            t2 = min(t2_raw, 2 * t1)
        else:
            t2 = t2_raw

        sqg_length = clip_norm(rng, spec_q['sqg_length'])
        rd_length = clip_norm(rng, spec_q['rd_length'])

        # fid: sample on error rate, convert to fid
        sqg_err = clip_norm(rng, spec_q['sqg_fid_err'])

        # measurement / init errors: clipped normal/lognormal
        data_init = clip_norm(rng, spec_q['data_init_error'])
        data_meas = clip_norm(rng, spec_q['data_measurement_error'])
        meas_spam = clip_norm(rng, spec_q['measurement_spam_rate'])

        # ---- Heterogeneous defect overlay (per-param) ----
        mults = hot_q_mults.get(q, {})
        if 'sqg_fid_err' in mults:
            sqg_err = min(spec_q['sqg_fid_err'].get('max', 0.5), sqg_err * mults['sqg_fid_err'])
        if 'data_init_error' in mults:
            data_init = min(spec_q['data_init_error'].get('max', 1.0), data_init * mults['data_init_error'])
        if 'data_measurement_error' in mults:
            data_meas = min(spec_q['data_measurement_error'].get('max', 1.0), data_meas * mults['data_measurement_error'])
        if 'measurement_spam_rate' in mults:
            meas_spam = min(spec_q['measurement_spam_rate'].get('max', 1.0), meas_spam * mults['measurement_spam_rate'])

        sqg_fid = 1.0 - sqg_err

        # leakage params (默认 disabled, 写 0)
        lp = float(spec_q['lp']['mean'])
        sp = float(spec_q['sp']['mean'])

        out[str(q)] = {
            "t1": float(t1),
            "t2": float(t2),
            "sqg_length": float(sqg_length),
            "sqg_fid": float(sqg_fid),
            "rd_length": float(rd_length),
            "data_init_error": float(data_init),
            "data_measurement_error": float(data_meas),
            "measurement_spam_rate": float(meas_spam),
            "lp": float(lp),
            "sp": float(sp),
        }
    return out


CANON_QUBIT_PARAMS = ['sqg_fid_err', 'data_init_error', 'data_measurement_error',
                      'measurement_spam_rate', 'lp']
CANON_CX_PARAMS = ['cx_fid_err', 'lp_propagation_prob']


def pick_hot_params(sub_rng, candidates, k_min, k_max, mult_min, mult_max):
    """Given a sub-RNG (deterministic per hot point), pick K params + multipliers.

    Both gen_level_params.py and gen_leakage_overrides.py call this with the SAME
    sub_rng + same candidates (canonical order) -> same dict in both scripts.
    Returns: dict[param_name] = multiplier_float.
    """
    n_pick = int(sub_rng.integers(k_min, k_max + 1))
    n_pick = min(n_pick, len(candidates))
    picked_idx = sorted(sub_rng.choice(len(candidates), n_pick, replace=False).tolist())
    out = {}
    # IMPORTANT: iterate canonical order so multiplier draw order matches across scripts
    for i, name in enumerate(candidates):
        if i in picked_idx:
            out[name] = float(sub_rng.uniform(mult_min, mult_max))
    return out


def clip_norm(rng, spec):
    """Sample N(mean, std) and clip to [min, max] (max may be None).

    Auto-dispatch: if spec contains 'log_std', use LogNormal instead (mean-preserving):
        val = mean * exp(N(-log_std**2/2, log_std))
    Right-skewed -- preferred for error rates / probabilities physically bounded at 0.
    """
    if 'log_std' in spec:
        return clip_lognormal(rng, spec)
    val = rng.normal(spec['mean'], spec['std'])
    val = max(spec.get('min', 0.0), val)
    if spec.get('max') is not None:
        val = min(spec['max'], val)
    return val


def clip_lognormal(rng, spec):
    """Sample mean-preserving LogNormal: val = mean * exp(N(-log_std**2/2, log_std)).

    E[val] = mean exactly; median = mean * exp(-log_std**2/2) < mean.
    log_std=0.5 -> ~5% of samples >= 2*mean, ~1% >= 5*mean.
    Right-skewed; reproduces 'bad qubit / bad CX' outliers found in real chip data.
    """
    mean = float(spec['mean'])
    log_std = float(spec['log_std'])
    if log_std <= 0:
        val = mean
    else:
        val = mean * np.exp(rng.normal(-log_std ** 2 / 2, log_std))
    val = max(spec.get('min', 0.0), val)
    if spec.get('max') is not None:
        val = min(spec['max'], val)
    return val


def sample_cx(rng, spec_cx, cx_pairs, hot_cx_mults=None):
    """cx_pairs: list of (control, target) integer ID tuples (1-based).

    hot_cx_mults (dict[int, dict[str, float]]): per-CX per-param multipliers.
    """
    if hot_cx_mults is None:
        hot_cx_mults = {}
    out = {}
    for idx, (c, t) in enumerate(cx_pairs, start=1):
        cx_err = clip_norm(rng, spec_cx['cx_fid_err'])
        mults = hot_cx_mults.get(idx, {})
        if 'cx_fid_err' in mults:
            cx_err = min(spec_cx['cx_fid_err'].get('max', 0.5), cx_err * mults['cx_fid_err'])
        cx_fid = 1.0 - cx_err
        cx_length = clip_norm(rng, spec_cx['cx_length'])
        lp_prop = float(spec_cx['lp_propagation_prob']['mean'])
        out[str(idx)] = {
            "control": int(c),
            "target": int(t),
            "cx_fid": float(cx_fid),
            "cx_length": float(cx_length),
            "lp_propagation_prob": float(lp_prop),
        }
    return out


def get_topology_from_paems(distance, rounds, basis):
    """优先用 PAEMS 已有函数生成拓扑（含 1-based qubit 编号）。"""
    try:
        from surface_code_generate_circuits import generate_surface_code_circuit
        _, data_q, x_stab, z_stab, cx_gates = generate_surface_code_circuit(
            distance, rounds, basis
        )
        all_q = sorted(set(data_q + x_stab + z_stab))
        cx_pairs = [(c, t) for _, (c, t) in cx_gates]
        # 去重（PAEMS cx_gates 是按出现顺序给的，可能含双向重复）
        seen = set()
        unique = []
        for c, t in cx_pairs:
            if (c, t) not in seen:
                seen.add((c, t))
                unique.append((c, t))
        return all_q, unique, data_q, x_stab, z_stab
    except ImportError:
        return None


def get_topology_fallback(distance, rounds, basis):
    """没有 PAEMS 模块时，直接用 Stim 解析。"""
    import stim
    code_type = ('surface_code:rotated_memory_z' if basis.lower() == 'z'
                 else 'surface_code:rotated_memory_x')
    circuit = stim.Circuit.generated(code_type, distance=distance, rounds=rounds)

    # 解析 qubit 编号
    coords = {}
    for inst in circuit:
        if inst.name == "QUBIT_COORDS":
            qid = inst.targets_copy()[0].value
            coords[qid] = inst.gate_args_copy()
    all_q_orig = sorted(coords.keys())
    # 1-based 重编号
    orig_to_new = {q: i + 1 for i, q in enumerate(all_q_orig)}
    all_q = [orig_to_new[q] for q in all_q_orig]

    # 找数据比特：最后一条 M 指令的 targets
    data_q = []
    for inst in reversed(circuit):
        if inst.name in ("M", "MX"):
            data_q = [orig_to_new[t.value] for t in inst.targets_copy()]
            break

    # 找 X stab：第一条 H 指令的 targets
    x_stab = []
    for inst in circuit:
        if inst.name == "H":
            x_stab = [orig_to_new[t.value] for t in inst.targets_copy()]
            break

    z_stab = sorted(set(all_q) - set(data_q) - set(x_stab))

    # 提 CX pair（去重）
    seen, cx_pairs = set(), []
    for inst in circuit:
        if inst.name == "CX":
            ts = inst.targets_copy()
            for j in range(0, len(ts), 2):
                c = orig_to_new[ts[j].value]
                t = orig_to_new[ts[j + 1].value]
                if (c, t) not in seen:
                    seen.add((c, t))
                    cx_pairs.append((c, t))
        elif inst.name == "REPEAT":
            for sub in inst.body_copy():
                if sub.name == "CX":
                    ts = sub.targets_copy()
                    for j in range(0, len(ts), 2):
                        c = orig_to_new[ts[j].value]
                        t = orig_to_new[ts[j + 1].value]
                        if (c, t) not in seen:
                            seen.add((c, t))
                            cx_pairs.append((c, t))

    return all_q, cx_pairs, data_q, x_stab, z_stab


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--level', type=int, required=True, choices=[1, 2, 3, 4])
    ap.add_argument('--distance', type=int, required=True)
    ap.add_argument('--rounds', type=int, required=True)
    ap.add_argument('--basis', default='z', choices=['z', 'x', 'Z', 'X'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--defect-seed', type=int, default=None,
                    help='RNG seed for defect overlay. None = no hot-spot injection.')
    ap.add_argument('--defect-multiplier', type=float, default=None,
                    help='Collapse mult_min/mult_max to single value (back-compat)')
    ap.add_argument('--defect-mult-min', type=float, default=None,
                    help='Override mult_min from spec')
    ap.add_argument('--defect-mult-max', type=float, default=None,
                    help='Override mult_max from spec')
    ap.add_argument('--defect-q-fraction', type=float, default=None,
                    help='Override qubit_fraction from spec (None = use spec value)')
    ap.add_argument('--defect-cx-fraction', type=float, default=None,
                    help='Override cx_fraction from spec (None = use spec value)')
    ap.add_argument('--spec', default=None,
                    help='Path to level_params_spec.json (default: same dir as script)')
    ap.add_argument('--out', required=True, help='Output JSON path')
    args = ap.parse_args()

    spec_path = args.spec or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'level_params_spec.json')
    spec = load_spec(spec_path)
    level_key = f'L{args.level}'
    if level_key not in spec:
        raise SystemExit(f'Spec missing {level_key}')
    level_spec = spec[level_key]

    # 拓扑（优先 PAEMS，fallback 到 Stim）
    topo = get_topology_from_paems(args.distance, args.rounds, args.basis)
    if topo is None:
        topo = get_topology_fallback(args.distance, args.rounds, args.basis)
    all_q, cx_pairs, data_q, x_stab, z_stab = topo

    rng = np.random.default_rng(args.seed)
    freq_spec = level_spec.get('frequency')

    # ---- Heterogeneous defect overlay ----
    hot_q_mults = {}; hot_cx_mults = {}; overlay = level_spec.get('defect_overlay')
    if args.defect_seed is not None and overlay is not None:
        master = np.random.default_rng(args.defect_seed)
        q_frac = float(args.defect_q_fraction if args.defect_q_fraction is not None
                       else overlay.get('qubit_fraction', 0.0))
        cx_frac = float(args.defect_cx_fraction if args.defect_cx_fraction is not None
                        else overlay.get('cx_fraction', 0.0))
        k_q_min, k_q_max = overlay.get('params_per_qubit', [1, 3])
        k_cx_min, k_cx_max = overlay.get('params_per_cx', [1, 2])
        # mult range: priority = single-value override > min/max overrides > spec
        if args.defect_multiplier is not None:
            mult_min = mult_max = float(args.defect_multiplier)
        else:
            mult_min = float(args.defect_mult_min if args.defect_mult_min is not None
                             else overlay.get('mult_min', 4.0))
            mult_max = float(args.defect_mult_max if args.defect_mult_max is not None
                             else overlay.get('mult_max', 10.0))
        n_hot_q = int(round(len(all_q) * q_frac))
        n_hot_cx = int(round(len(cx_pairs) * cx_frac))
        if q_frac > 0 and n_hot_q == 0: n_hot_q = 1
        if cx_frac > 0 and n_hot_cx == 0: n_hot_cx = 1
        # Master picks indices + sub-seeds (same logic in leakage script)
        hot_q_idx = master.choice(len(all_q), n_hot_q, replace=False) if n_hot_q > 0 else np.array([], dtype=int)
        hot_cx_idx = master.choice(len(cx_pairs), n_hot_cx, replace=False) if n_hot_cx > 0 else np.array([], dtype=int)
        sub_seeds_q = master.integers(0, 2**31, size=n_hot_q) if n_hot_q > 0 else np.array([], dtype=int)
        sub_seeds_cx = master.integers(0, 2**31, size=n_hot_cx) if n_hot_cx > 0 else np.array([], dtype=int)
        # Per-hot-point: pick params + multipliers via sub-RNG
        for j, qi in enumerate(hot_q_idx):
            sub = np.random.default_rng(int(sub_seeds_q[j]))
            picks = pick_hot_params(sub, CANON_QUBIT_PARAMS, k_q_min, k_q_max, mult_min, mult_max)
            # Filter to params owned by classical script (drop 'lp' which is leakage's)
            hot_q_mults[int(qi) + 1] = {k: v for k, v in picks.items() if k != 'lp'}
        for j, ci in enumerate(hot_cx_idx):
            sub = np.random.default_rng(int(sub_seeds_cx[j]))
            picks = pick_hot_params(sub, CANON_CX_PARAMS, k_cx_min, k_cx_max, mult_min, mult_max)
            hot_cx_mults[int(ci) + 1] = {k: v for k, v in picks.items() if k != 'lp_propagation_prob'}

    qubits = sample_qubit(rng, level_spec['qubit'], n_qubits=len(all_q),
                          freq_spec=freq_spec, all_qubits=all_q, cx_pairs=cx_pairs,
                          hot_q_mults=hot_q_mults)
    cx_gates = sample_cx(rng, level_spec['cx'], cx_pairs, hot_cx_mults=hot_cx_mults)

    # ---- Crosstalk: empty per-pair list (no global default in v2) ----
    # Crosstalk is now ONLY introduced via crosstalk_pairs.
    # Run gen_pair_overrides.py to populate based on freq + distance,
    # or write crosstalk_pairs manually before running PAEMS.
    crosstalk_pairs = {}

    out = {
        "qubits": qubits,
        "cx_gates": cx_gates,
        "crosstalk_pairs": crosstalk_pairs,
        "_metadata": {
            "level": args.level,
            "level_label": level_spec['label'],
            "distance": args.distance,
            "rounds": args.rounds,
            "basis": args.basis.lower(),
            "seed": args.seed,
            "n_qubits": len(all_q),
            "n_cx_pairs": len(cx_pairs),
            "data_qubits": data_q,
            "x_stabilizers": x_stab,
            "z_stabilizers": z_stab,
            "spec_source": os.path.basename(spec_path),
            "defect_overlay": overlay,
            "defect_seed": args.defect_seed,
            "hot_qubits_mults": hot_q_mults,
            "hot_cx_mults": hot_cx_mults,
            "leakage_disabled": True,
            "crosstalk_pairs_count": len(crosstalk_pairs),
            "note": ("Classical noise + crosstalk via pairs only. "
                     "lp/sp/lp_propagation_prob all set to 0. "
                     "crosstalk_pairs is empty by default — run gen_pair_overrides.py "
                     "to populate from frequency + topology, or write pairs manually.")
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)

    # 简短摘要
    import statistics as st
    t1s = [q['t1'] * 1e6 for q in qubits.values()]
    t2s = [q['t2'] * 1e6 for q in qubits.values()]
    sqg_fids = [q['sqg_fid'] for q in qubits.values()]
    spams = [q['measurement_spam_rate'] for q in qubits.values()]
    cx_fids = [g['cx_fid'] for g in cx_gates.values()]
    print(f"[OK] Wrote {args.out}")
    print(f"  level={args.level} ({level_spec['label']})")
    print(f"  distance={args.distance}  rounds={args.rounds}  basis={args.basis}")
    print(f"  qubits={len(qubits)}  cx_pairs={len(cx_gates)}")
    if args.defect_seed is not None and overlay is not None:
        print(f"  defect: {len(hot_q_mults)} hot qubits, {len(hot_cx_mults)} hot CX  (seed={args.defect_seed})")
        for q, m in hot_q_mults.items():
            print(f"    hot qubit q{q}: {', '.join(f'{k}×{v:.1f}' for k,v in m.items()) or '(no classical params picked)'}")
        for c, m in hot_cx_mults.items():
            print(f"    hot CX  c{c}: {', '.join(f'{k}×{v:.1f}' for k,v in m.items()) or '(no classical params picked)'}")
    print(f"  T1 (us)        mean={st.mean(t1s):.1f}  std={st.stdev(t1s):.1f}")
    print(f"  T2 (us)        mean={st.mean(t2s):.1f}  std={st.stdev(t2s):.1f}")
    print(f"  sqg_fid        mean={st.mean(sqg_fids):.5f}")
    print(f"  meas_spam      mean={st.mean(spams):.4f}")
    print(f"  cx_fid         mean={st.mean(cx_fids):.5f}")


if __name__ == '__main__':
    main()
