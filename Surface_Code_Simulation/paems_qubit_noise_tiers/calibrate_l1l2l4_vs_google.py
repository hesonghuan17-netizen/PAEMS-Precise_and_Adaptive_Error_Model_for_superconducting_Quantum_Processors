#!/usr/bin/env python3
"""
PAEMS L1+L2+L4 (no L3 leakage) calibration against Google real-chip syndrome.

L3 (leakage post-processing) is intentionally excluded because it cannot live
inside a stim Circuit / DEM, so any sampler driven from PAEMS-injected stim
circuits inherits only L1+L2+L4. We need to fit L1+L2+L4 alone to the Google
chip defect-rate / per-detector distribution / Spitz pij correlation so that
the syndrome distribution from PAEMS sampling matches the real chip.

Usage:
    python calibrate_l1l2l4_vs_google.py \
        --distance 7 --rounds 250 --patch q6_7 --shots 2000 \
        --level 2 --xtalk X2 --defect-seed 7 --mult-scale-with-d
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import stim

PAEMS_SC = r"C:\PAEMS-Precise_and_Adaptive_Error_Model_for_superconducting_Quantum_Processors\Surface_Code_Simulation"
sys.path.insert(0, PAEMS_SC)
from inject_basic_noise import inject_surface_code_noise  # noqa: E402
from surface_code_generate_circuits import generate_surface_code_circuit  # noqa: E402

GOOGLE_ROOT_TMPL = (r"C:\Users\10124\Desktop\google dataset"
                    r"\google_105Q_surface_code_d3_d5_d7"
                    r"\d{d}_at_{patch}\Z\r{r}")


# ---------------- metrics ----------------
def per_detector_rate(dets):
    """dets shape (shots, n_dets). returns array of per-detector firing rate."""
    return dets.mean(axis=0)


def per_round_rate(dets, circuit):
    """Group detectors by their t coord and average within each round."""
    coords = circuit.get_detector_coordinates()
    ts = np.array([coords[i][2] if len(coords[i]) > 2 else 0
                   for i in range(circuit.num_detectors)])
    out = []
    for t in sorted(set(int(x) for x in ts)):
        out.append(dets[:, ts == t].mean())
    return np.array(out), np.array(sorted(set(int(x) for x in ts)))


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
        best_q, best_d = None, float('inf')
        for q, (qx, qy) in qubit_coords.items():
            d = (dx - qx) ** 2 + (dy - qy) ** 2
            if d < best_d:
                best_d, best_q = d, q
        det_to_qubit[did] = (best_q, coord[2] if len(coord) > 2 else 0)
    sort_did = sorted(det_coords.keys(),
                      key=lambda d: (det_to_qubit[d][0], det_to_qubit[d][1]))
    return (np.array(sort_did),
            np.array([det_to_qubit[d][0] for d in sort_did]))


def cx_graph(circuit):
    import networkx as nx
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


def summary_block(name, dets, circuit, qpd, sp):
    rate = dets.mean() * 100
    per_det = per_detector_rate(dets) * 100
    per_round, ts = per_round_rate(dets, circuit)
    pij = compute_pij_spitz(dets)
    cross = filter_hop_ge2(pij, qpd, sp)

    print(f"\n=== {name} ===")
    print(f"  shots: {dets.shape[0]}, n_detectors: {dets.shape[1]}")
    print(f"  overall defect rate: {rate:.3f}%")
    print(f"  per-detector rate %: mean={per_det.mean():.3f} "
          f"std={per_det.std():.3f} "
          f"p10={np.percentile(per_det,10):.3f} "
          f"p50={np.percentile(per_det,50):.3f} "
          f"p90={np.percentile(per_det,90):.3f} "
          f"p99={np.percentile(per_det,99):.3f} "
          f"max={per_det.max():.3f}")
    print(f"  per-round rate %: first3={per_round[:3]*100} "
          f"last3={per_round[-3:]*100} "
          f"mid_mean={per_round[len(per_round)//4:-len(per_round)//4].mean()*100:.3f}")
    print(f"  Spitz pij hop>=2 cross-qubit: count={len(cross)} "
          f"mean={cross.mean():.5f} "
          f"max={cross.max():.5f} "
          f"p90={np.percentile(cross,90):.5f} "
          f"p99={np.percentile(cross,99):.5f} "
          f">1e-3:{(cross>1e-3).mean()*100:.2f}% "
          f">1e-2:{(cross>1e-2).mean()*100:.2f}%")
    return {"name": name, "rate": rate, "per_det": per_det,
            "per_round": per_round, "pij_cross": cross}


# ---------------- PAEMS config build ----------------
def make_l1l2l4_config(here, distance, rounds, level, xtalk_json,
                       defect_seed, q_frac, cx_frac, mult_min, mult_max,
                       defect_mult, out_path,
                       code_variant='css', xzzx_template=None):
    """Build PAEMS classical L<level> + crosstalk + defect-overlay config.
    Skips leakage entirely (no gen_leakage_overrides.py call).
    code_variant='xzzx' loads topology from external noiseless .stim template
    (e.g. Google circuit_ideal.stim) instead of stim's CSS generator.
    """
    tmp_dir = os.path.join(here, 'tmp_test')
    os.makedirs(tmp_dir, exist_ok=True)
    xt_tag = (os.path.splitext(os.path.basename(xtalk_json))[0]
              if xtalk_json is not None else 'noxt')
    tag = f"l{level}_x{xt_tag}_{code_variant}_d{distance}r{rounds}"

    # Step 1: classical (L<level>) + defect overlay
    p_class = os.path.join(tmp_dir, f'_cal_class_{tag}.json')
    cargs = ['--level', str(level), '--distance', str(distance),
             '--rounds', str(rounds), '--seed', '42',
             '--defect-seed', str(defect_seed),
             '--out', p_class,
             '--code-variant', code_variant]
    if code_variant == 'xzzx':
        if xzzx_template is None:
            raise ValueError('code_variant=xzzx requires xzzx_template')
        cargs += ['--xzzx-template', xzzx_template]
    if defect_mult is not None:
        cargs += ['--defect-multiplier', str(defect_mult)]
    if q_frac is not None:
        cargs += ['--defect-q-fraction', str(q_frac)]
    if cx_frac is not None:
        cargs += ['--defect-cx-fraction', str(cx_frac)]
    if mult_min is not None:
        cargs += ['--defect-mult-min', str(mult_min)]
    if mult_max is not None:
        cargs += ['--defect-mult-max', str(mult_max)]
    subprocess.run([sys.executable, os.path.join(here, 'gen_level_params.py'),
                    *cargs], check=True, capture_output=True, cwd=here)

    # Step 2: merge crosstalk
    if xtalk_json is None:
        # No crosstalk
        with open(p_class, encoding='utf-8') as f:
            d = json.load(f)
        d['crosstalk_pairs'] = {}
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(d, f, indent=2)
    else:
        subprocess.run([sys.executable,
                        os.path.join(here, 'gen_pair_overrides.py'),
                        '--in', p_class, '--out', out_path, '--merge',
                        '--crosstalk-config', xtalk_json],
                       check=True, capture_output=True, cwd=here)


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--distance', type=int, default=7)
    ap.add_argument('--rounds', type=int, default=250)
    ap.add_argument('--patch', type=str, default='q6_7')
    ap.add_argument('--shots', type=int, default=2000)
    ap.add_argument('--google-shots', type=int, default=5000,
                    help='Number of Google shots to load for reference (cap)')
    ap.add_argument('--level', type=int, default=2, choices=[1, 2, 3, 4],
                    help='PAEMS classical level (1=next-gen ... 4=poor)')
    ap.add_argument('--xtalk', type=str, default='X2',
                    choices=['none', 'X1', 'X2', 'X3', 'X4'],
                    help='Crosstalk preset (none = skip crosstalk)')
    ap.add_argument('--defect-seed', type=int, default=7)
    ap.add_argument('--q-frac', type=float, default=None)
    ap.add_argument('--cx-frac', type=float, default=None)
    ap.add_argument('--mult-min', type=float, default=None)
    ap.add_argument('--mult-max', type=float, default=None)
    ap.add_argument('--defect-mult', type=float, default=None,
                    help='Single fixed multiplier (overrides mult-min/max)')
    ap.add_argument('--mult-scale-with-d', action='store_true',
                    help='auto mult_max(d) = max(5, 8 + (d-5)*2): d=5→8 d=7→12 d=9→16')
    ap.add_argument('--tag', type=str, default='',
                    help='Tag for tmp config file')
    ap.add_argument('--code-variant', default='css', choices=['css', 'xzzx'],
                    help='Surface code variant (default css = standard rotated). '
                         'xzzx requires --xzzx-template.')
    ap.add_argument('--xzzx-template', default=None,
                    help='Noiseless XZZX .stim template path. If omitted with '
                         'code-variant=xzzx, defaults to Google chip ideal circuit '
                         'at the patch path.')
    args = ap.parse_args()

    # Auto-derive XZZX template path from Google patch if not given
    if args.code_variant == 'xzzx' and args.xzzx_template is None:
        args.xzzx_template = GOOGLE_ROOT_TMPL.format(
            d=args.distance, patch=args.patch, r=args.rounds) + '\\circuit_ideal.stim'

    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(here, 'tmp_test'), exist_ok=True)

    eff_mult_max = args.mult_max
    if args.mult_scale_with_d and eff_mult_max is None:
        eff_mult_max = max(5.0, 8.0 + (args.distance - 5) * 2.0)
        print(f"[mult-scale-with-d] d={args.distance} -> mult_max={eff_mult_max}")

    xt_json = (None if args.xtalk == 'none'
               else os.path.join(here, 'crosstalk_presets', f'crosstalk_{args.xtalk}.json'))

    cfg_tag = (f"L{args.level}_{args.xtalk}_{args.code_variant}_seed{args.defect_seed}"
               f"_qf{args.q_frac}_cxf{args.cx_frac}"
               f"_mm{args.mult_min}_{eff_mult_max}"
               f"_d{args.distance}r{args.rounds}{('_'+args.tag) if args.tag else ''}")
    out_cfg = os.path.join(here, 'tmp_test', f'_cal_final_{cfg_tag}.json')

    print(f"=== PAEMS L1+L2+L4 calibration vs Google d={args.distance} r={args.rounds} {args.patch} ===")
    print(f"  code_variant={args.code_variant}"
          + (f" xzzx_template={args.xzzx_template}" if args.code_variant == 'xzzx' else ""))
    print(f"  L-level={args.level}, xtalk={args.xtalk}, "
          f"defect-seed={args.defect_seed} q_frac={args.q_frac} "
          f"cx_frac={args.cx_frac} mult_min={args.mult_min} mult_max={eff_mult_max} "
          f"defect_mult={args.defect_mult}")
    print(f"  shots(PAEMS)={args.shots}  shots(Google)={args.google_shots}")
    print(f"  out_cfg: {out_cfg}")

    # ---- Build PAEMS config + sample ----
    print("\n[1/3] Building PAEMS L1+L2+L4 config ...")
    t0 = time.time()
    make_l1l2l4_config(here, args.distance, args.rounds,
                       level=args.level,
                       xtalk_json=xt_json,
                       defect_seed=args.defect_seed,
                       q_frac=args.q_frac, cx_frac=args.cx_frac,
                       mult_min=args.mult_min, mult_max=eff_mult_max,
                       defect_mult=args.defect_mult,
                       out_path=out_cfg,
                       code_variant=args.code_variant,
                       xzzx_template=args.xzzx_template)
    print(f"  built in {time.time()-t0:.1f}s")

    print("\n[2/3] Building noisy stim circuit + sampling shots ...")
    t0 = time.time()
    base_circ, dq, xs, zs, cx = generate_surface_code_circuit(
        args.distance, args.rounds, 'z',
        code_variant=args.code_variant, xzzx_template=args.xzzx_template)
    nc = inject_surface_code_noise(base_circ, dq, xs, zs, cx, out_cfg)
    print(f"  build noisy circuit: {time.time()-t0:.1f}s, num_detectors={nc.num_detectors}")
    t0 = time.time()
    dets_paems = nc.compile_detector_sampler().sample(shots=args.shots)
    print(f"  sample {args.shots} shots: {time.time()-t0:.1f}s")
    dets_paems = dets_paems.astype(np.uint8)

    # ---- Load Google reference ----
    print("\n[3/3] Loading Google reference + computing metrics ...")
    base = GOOGLE_ROOT_TMPL.format(d=args.distance, patch=args.patch, r=args.rounds)
    cir_real = stim.Circuit.from_file(os.path.join(base, "circuit_noisy_si1000.stim"))
    dets_real = stim.read_shot_data_file(
        path=os.path.join(base, "detection_events.b8"),
        format='b8', num_detectors=cir_real.num_detectors, bit_packed=False)
    if dets_real.shape[0] > args.google_shots:
        dets_real = dets_real[:args.google_shots]
    dets_real = dets_real.astype(np.uint8)

    # Reorder by qubit-block for both
    sort_p, qpd_p = detector_to_qubit_order(nc)
    sort_r, qpd_r = detector_to_qubit_order(cir_real)
    dets_paems = dets_paems[:, sort_p]
    dets_real = dets_real[:, sort_r]
    n = min(dets_paems.shape[1], dets_real.shape[1])
    dets_paems = dets_paems[:, :n]; dets_real = dets_real[:, :n]
    qpd_p, qpd_r = qpd_p[:n], qpd_r[:n]

    import networkx as nx
    Gp = cx_graph(nc); Gr = cx_graph(cir_real)
    sp_p = dict(nx.all_pairs_shortest_path_length(Gp))
    sp_r = dict(nx.all_pairs_shortest_path_length(Gr))

    res_p = summary_block(f"PAEMS L{args.level}+{args.xtalk}+L4", dets_paems, nc, qpd_p, sp_p)
    res_r = summary_block(f"Google d={args.distance} r={args.rounds} {args.patch}",
                          dets_real, cir_real, qpd_r, sp_r)

    # ---- Side-by-side gap report ----
    print("\n=== GAP (PAEMS - Google), positive = PAEMS too high ===")
    print(f"  defect_rate: {res_p['rate']-res_r['rate']:+.3f}pp  "
          f"(P={res_p['rate']:.3f}%, G={res_r['rate']:.3f}%)")
    p10p, p90p = np.percentile(res_p['per_det'], [10, 90])
    p10r, p90r = np.percentile(res_r['per_det'], [10, 90])
    print(f"  per-det p10:  {p10p-p10r:+.3f}pp  (P={p10p:.3f}%, G={p10r:.3f}%)")
    print(f"  per-det p90:  {p90p-p90r:+.3f}pp  (P={p90p:.3f}%, G={p90r:.3f}%)")
    print(f"  per-det std:  {res_p['per_det'].std()-res_r['per_det'].std():+.3f}pp")
    print(f"  pij hop>=2 mean: {res_p['pij_cross'].mean()-res_r['pij_cross'].mean():+.5f}")
    print(f"  pij hop>=2 >1e-3 frac: "
          f"{(res_p['pij_cross']>1e-3).mean()*100 - (res_r['pij_cross']>1e-3).mean()*100:+.2f}pp")


if __name__ == '__main__':
    main()
