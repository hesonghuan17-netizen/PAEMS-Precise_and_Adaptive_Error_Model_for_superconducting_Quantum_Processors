#!/usr/bin/env python3
"""
Export a PAEMS-noisy stim Circuit to a .stim file so that it can be fed
into the NLDU dataset generator (gen_dataset_stream.py --circuit-file).

Usage:
    python export_paems_circuit.py \
        --params tmp_test/_cal_final_L2_X2_seed7_qfNone_cxfNone_mm5.0_20.0_d7r250_E_mult5_20.json \
        --distance 7 --rounds 250 --basis z \
        --out paems_d7_r250_E.stim
"""
import argparse
import os
import sys

PAEMS_SC = os.environ.get(
    'PAEMS_SC',
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PAEMS_SC)
from inject_basic_noise import inject_surface_code_noise  # noqa: E402
from surface_code_generate_circuits import generate_surface_code_circuit  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--params', required=True, help='PAEMS params JSON path')
    ap.add_argument('--distance', type=int, required=True)
    ap.add_argument('--rounds', type=int, required=True)
    ap.add_argument('--basis', default='z', choices=['z', 'x', 'Z', 'X'])
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    print(f"Building d={args.distance} r={args.rounds} basis={args.basis} from {args.params}")
    base_circ, dq, xs, zs, cx = generate_surface_code_circuit(
        args.distance, args.rounds, args.basis.lower())
    nc = inject_surface_code_noise(base_circ, dq, xs, zs, cx, args.params)
    print(f"  num_qubits: {nc.num_qubits}, num_detectors: {nc.num_detectors}, "
          f"num_observables: {nc.num_observables}")

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(str(nc))
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"Wrote {out_path} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
