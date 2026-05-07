#!/usr/bin/env python3
"""
读取一个已有的 PAEMS params JSON（含 qubits + cx_gates），
按 leakage_presets/leakage_LX.json 把每个 qubit 的 lp/sp 和
每条 cx_gate 的 lp_propagation_prob 覆盖写入，输出新 JSON。

支持两种 preset 字段格式：
  scalar:           "lp": 4e-5
  LogNormal dict:   "lp": {"mean": 4e-5, "log_std": 0.7, "min": 1e-8, "max": 5e-3}

LogNormal 采样保留真实均值: val = mean * exp(N(-log_std**2/2, log_std))
即 E[val] = mean, 但分布右偏长尾。

与 gen_pair_overrides.py 平行：负责泄漏注入参数；与 L1-L4 经典级和
X1-X4 串扰级完全解耦，可任意组合。

用法：
    python gen_leakage_overrides.py --in base.json \
        --leakage-config leakage_presets/leakage_L2.json --out out.json --seed 42
"""
import argparse
import json
import os
import sys
import numpy as np


def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


CANON_QUBIT_PARAMS = ['sqg_fid_err', 'data_init_error', 'data_measurement_error',
                      'measurement_spam_rate', 'lp']
CANON_CX_PARAMS = ['cx_fid_err', 'lp_propagation_prob']


def pick_hot_params(sub_rng, candidates, k_min, k_max, mult_min, mult_max):
    """Same canonical picker as gen_level_params.py — both must match exactly."""
    n_pick = int(sub_rng.integers(k_min, k_max + 1))
    n_pick = min(n_pick, len(candidates))
    picked_idx = sorted(sub_rng.choice(len(candidates), n_pick, replace=False).tolist())
    out = {}
    for i, name in enumerate(candidates):
        if i in picked_idx:
            out[name] = float(sub_rng.uniform(mult_min, mult_max))
    return out


def sample_field(rng, spec, n):
    """spec 是标量或 {mean, log_std, min?, max?}; 返回长度 n 的数组。"""
    if isinstance(spec, (int, float)):
        return np.full(n, float(spec))
    if not isinstance(spec, dict):
        raise ValueError(f"Unsupported spec: {spec!r}")
    mean = float(spec['mean'])
    log_std = float(spec.get('log_std', 0.0))
    if log_std == 0.0 or n == 0:
        vals = np.full(n, mean)
    else:
        # E[val] = mean (mean-preserving lognormal)
        vals = mean * np.exp(rng.normal(-log_std**2 / 2, log_std, size=n))
    if 'max' in spec:
        vals = np.minimum(vals, float(spec['max']))
    if 'min' in spec:
        vals = np.maximum(vals, float(spec['min']))
    return vals


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--in', dest='inp', required=True,
                    help='Base params JSON (from gen_level_params or gen_pair_overrides)')
    ap.add_argument('--leakage-config', required=True,
                    help='Path to leakage_LX.json preset')
    ap.add_argument('--out', required=True, help='Output JSON path')
    ap.add_argument('--seed', type=int, default=42, help='RNG seed for LogNormal sampling')
    ap.add_argument('--defect-seed', type=int, default=None,
                    help='RNG seed for defect overlay (hot-spot index pick). None = disabled.')
    ap.add_argument('--defect-multiplier', type=float, default=None,
                    help='Collapse mult_min/mult_max to single value (back-compat)')
    ap.add_argument('--defect-mult-min', type=float, default=None,
                    help='Override mult_min from preset')
    ap.add_argument('--defect-mult-max', type=float, default=None,
                    help='Override mult_max from preset')
    ap.add_argument('--defect-q-fraction', type=float, default=None,
                    help='Override qubit_fraction from preset (None = use preset value)')
    ap.add_argument('--defect-cx-fraction', type=float, default=None,
                    help='Override cx_fraction from preset (None = use preset value)')
    args = ap.parse_args()

    base = load_json(args.inp)
    leak = load_json(args.leakage_config)
    rng = np.random.default_rng(args.seed)

    label = leak.get('label', os.path.basename(args.leakage_config))
    qubits = base.get('qubits', {})
    cx_gates = base.get('cx_gates', {})
    n_q = len(qubits); n_cx = len(cx_gates)

    # Sample per-qubit / per-CX
    lp_arr = sample_field(rng, leak['lp'], n_q)
    sp_arr = sample_field(rng, leak['sp'], n_q)
    prop_arr = sample_field(rng, leak['lp_propagation_prob'], n_cx)

    # ---- Heterogeneous defect overlay (lp / lp_propagation_prob only) ----
    hot_q_lp_mults = {}; hot_cx_prop_mults = {}
    overlay = leak.get('defect_overlay')
    if args.defect_seed is not None and overlay is not None:
        master = np.random.default_rng(args.defect_seed)
        q_frac = float(args.defect_q_fraction if args.defect_q_fraction is not None
                       else overlay.get('qubit_fraction', 0.0))
        cx_frac = float(args.defect_cx_fraction if args.defect_cx_fraction is not None
                        else overlay.get('cx_fraction', 0.0))
        k_q_min, k_q_max = overlay.get('params_per_qubit', [1, 3])
        k_cx_min, k_cx_max = overlay.get('params_per_cx', [1, 2])
        if args.defect_multiplier is not None:
            mult_min = mult_max = float(args.defect_multiplier)
        else:
            mult_min = float(args.defect_mult_min if args.defect_mult_min is not None
                             else overlay.get('mult_min', 4.0))
            mult_max = float(args.defect_mult_max if args.defect_mult_max is not None
                             else overlay.get('mult_max', 10.0))
        n_hot_q = int(round(n_q * q_frac))
        n_hot_cx = int(round(n_cx * cx_frac))
        if q_frac > 0 and n_hot_q == 0: n_hot_q = 1
        if cx_frac > 0 and n_hot_cx == 0: n_hot_cx = 1
        # Same master sequence as gen_level_params.py -> same indices + sub-seeds
        hot_q_idx = master.choice(n_q, n_hot_q, replace=False) if n_hot_q > 0 else np.array([], dtype=int)
        hot_cx_idx = master.choice(n_cx, n_hot_cx, replace=False) if n_hot_cx > 0 else np.array([], dtype=int)
        sub_seeds_q = master.integers(0, 2**31, size=n_hot_q) if n_hot_q > 0 else np.array([], dtype=int)
        sub_seeds_cx = master.integers(0, 2**31, size=n_hot_cx) if n_hot_cx > 0 else np.array([], dtype=int)
        # Per-hot-point: same sub-RNG sequence as classical; we only consume 'lp' / 'lp_propagation_prob' picks
        for j, qi in enumerate(hot_q_idx):
            sub = np.random.default_rng(int(sub_seeds_q[j]))
            picks = pick_hot_params(sub, CANON_QUBIT_PARAMS, k_q_min, k_q_max, mult_min, mult_max)
            if 'lp' in picks:
                idx = int(qi)
                lp_arr[idx] = lp_arr[idx] * picks['lp']
                hot_q_lp_mults[idx + 1] = picks['lp']
        for j, ci in enumerate(hot_cx_idx):
            sub = np.random.default_rng(int(sub_seeds_cx[j]))
            picks = pick_hot_params(sub, CANON_CX_PARAMS, k_cx_min, k_cx_max, mult_min, mult_max)
            if 'lp_propagation_prob' in picks:
                idx = int(ci)
                prop_arr[idx] = prop_arr[idx] * picks['lp_propagation_prob']
                hot_cx_prop_mults[idx + 1] = picks['lp_propagation_prob']
        # Final clip to max
        if isinstance(leak['lp'], dict) and 'max' in leak['lp']:
            lp_arr = np.minimum(lp_arr, float(leak['lp']['max']))
        if isinstance(leak['lp_propagation_prob'], dict) and 'max' in leak['lp_propagation_prob']:
            prop_arr = np.minimum(prop_arr, float(leak['lp_propagation_prob']['max']))

    # Apply (qubits/cx_gates dicts iterate in insertion order in modern Python)
    for i, (qid, qdict) in enumerate(qubits.items()):
        qdict['lp'] = float(lp_arr[i])
        qdict['sp'] = float(sp_arr[i])
    for i, (gid, gdict) in enumerate(cx_gates.items()):
        gdict['lp_propagation_prob'] = float(prop_arr[i])

    # Stamp metadata
    meta = base.setdefault('_metadata', {})
    meta['leakage_disabled'] = False
    meta['leakage_preset'] = {
        'label': label,
        'lp_spec': leak['lp'],
        'sp_spec': leak['sp'],
        'lp_propagation_prob_spec': leak['lp_propagation_prob'],
        'defect_overlay': overlay,
        'seed': args.seed,
        'defect_seed': args.defect_seed,
        'hot_qubits_lp_mults': hot_q_lp_mults,
        'hot_cx_prop_mults': hot_cx_prop_mults,
        'source': os.path.abspath(args.leakage_config),
    }
    notes = meta.get('note', '')
    if 'leakage' not in notes.lower():
        meta['note'] = (notes + ' | ' if notes else '') + f'Leakage applied via {label} (LogNormal per-qubit).'

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(base, f, indent=2)

    # Summary
    def pct(arr, p): return float(np.percentile(arr, p))
    print(f"[OK] Wrote {args.out}")
    print(f"  preset:  {label}")
    print(f"  qubits:  {n_q}    cx_gates: {n_cx}    seed: {args.seed}    defect_seed: {args.defect_seed}")
    if args.defect_seed is not None and overlay is not None:
        print(f"  lp_picked: {len(hot_q_lp_mults)} hot qubits had lp×, prop_picked: {len(hot_cx_prop_mults)} hot CX")
        for q, m in hot_q_lp_mults.items():
            print(f"    hot qubit q{q}: lp×{m:.1f}")
        for c, m in hot_cx_prop_mults.items():
            print(f"    hot CX  c{c}: prop×{m:.1f}")
    print(f"  lp        median={np.median(lp_arr):.3e}  mean={lp_arr.mean():.3e}  "
          f"max={lp_arr.max():.3e}  p99={pct(lp_arr,99):.3e}")
    print(f"  sp        median={np.median(sp_arr):.3e}  mean={sp_arr.mean():.3e}")
    print(f"  prop      median={np.median(prop_arr):.3f}    mean={prop_arr.mean():.3f}    "
          f"max={prop_arr.max():.3f}    p99={pct(prop_arr,99):.3f}")


if __name__ == '__main__':
    main()
