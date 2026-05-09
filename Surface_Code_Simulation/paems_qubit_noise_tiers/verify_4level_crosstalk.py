#!/usr/bin/env python3
"""
4 级 χ 值定案后的最终验证：每个 level 跑 with-crosstalk vs without-crosstalk
看 density 增量是否符合预期阶梯。
"""
import sys, os, json, tempfile, subprocess
from pathlib import Path

PAEMS_SC = r"C:\PAEMS-Precise_and_Adaptive_Error_Model_for_superconducting_Quantum_Processors\Surface_Code_Simulation"
sys.path.insert(0, PAEMS_SC)

from inject_basic_noise import inject_surface_code_noise
from surface_code_generate_circuits import generate_surface_code_circuit

HERE = Path(__file__).parent
PYTHON = sys.executable


def gen_param(level, out_path, seed=42):
    """Re-generate sample using current spec."""
    subprocess.run([PYTHON, str(HERE / 'gen_level_params.py'),
                    '--level', str(level), '--distance', '3', '--rounds', '3',
                    '--seed', str(seed), '--out', out_path],
                   check=True, capture_output=True)


def make_no_xtalk_copy(in_path, out_path):
    """Copy params, set crosstalk_global to 0."""
    with open(in_path) as f:
        d = json.load(f)
    d['crosstalk_global'] = {"default_strength": 0.0}
    d['crosstalk_pairs'] = {}
    with open(out_path, 'w') as f:
        json.dump(d, f, indent=2)


def measure_density(params_file, distance=3, rounds=3, shots=50000):
    circuit, dq, xs, zs, cxs = generate_surface_code_circuit(distance, rounds, 'z')
    nc = inject_surface_code_noise(circuit, dq, xs, zs, cxs, params_file)
    dets = nc.compile_detector_sampler().sample(shots=shots)
    return float(dets.mean())


print(f"{'Level':<6} {'chi_default':>12} {'no-xtalk':>10} {'with-xtalk':>12} {'delta(pp)':>10} {'rel(%)':>8}")
print("-" * 70)

with tempfile.TemporaryDirectory() as td:
    for L in [1, 2, 3, 4]:
        full_path = os.path.join(td, f'L{L}_full.json')
        no_path = os.path.join(td, f'L{L}_no.json')
        gen_param(L, full_path)
        make_no_xtalk_copy(full_path, no_path)

        with open(full_path) as f:
            chi = json.load(f)['crosstalk_global']['default_strength']
        density_no = measure_density(no_path)
        density_with = measure_density(full_path)
        delta = density_with - density_no
        rel = delta / density_no * 100 if density_no > 0 else 0
        print(f"L{L:<5} {chi:>12.0e} {density_no*100:>9.3f}% {density_with*100:>11.3f}% "
              f"{delta*100:>+9.3f}pp {rel:>+7.1f}%")
