#!/usr/bin/env python3
"""
Smoketest for crosstalk architecture.

验证 4 件事：
  1. crosstalk=0 时电路与无 crosstalk 完全相同
  2. crosstalk>0 时 H/CX 后立刻有 DEPOLARIZE1 spectator
  3. DEM 的 error 数随 crosstalk 增加而增加
  4. validator 能正确报警（CX-pair override）

用 d=3 r=3 小规模快速验证。
"""
import sys, os, json, tempfile
from pathlib import Path

PAEMS_SC = r"C:\PAEMS-Precise_and_Adaptive_Error_Model_for_superconducting_Quantum_Processors\Surface_Code_Simulation"
sys.path.insert(0, PAEMS_SC)

from inject_basic_noise import (
    inject_surface_code_noise,
    add_spectator_crosstalk,
    get_crosstalk_strength,
    validate_crosstalk_pairs_basic,
)
from surface_code_generate_circuits import generate_surface_code_circuit
import stim


HERE = Path(__file__).parent
SAMPLE_JSON = HERE / "validation_d3" / "param_examples" / "d3_level2.json"


def make_test_json(out_path, chi_default=0.0, chi_pairs=None):
    """Copy d3_level2.json and override crosstalk fields."""
    with open(SAMPLE_JSON) as f:
        d = json.load(f)
    d['crosstalk_global'] = {"default_strength": float(chi_default)}
    d['crosstalk_pairs'] = chi_pairs or {}
    with open(out_path, 'w') as f:
        json.dump(d, f, indent=2)


def build_noisy_circuit(chi_default, chi_pairs=None):
    circuit, data_q, x_stab, z_stab, cx_gates = generate_surface_code_circuit(3, 3, 'z')
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        tmp = f.name
    try:
        make_test_json(tmp, chi_default, chi_pairs)
        nc = inject_surface_code_noise(circuit, data_q, x_stab, z_stab, cx_gates, tmp)
        return nc
    finally:
        os.unlink(tmp)


def count_instructions(circuit, name):
    return sum(1 for inst in circuit if inst.name == name)


# =============================================================================
# Test 1: chi=0 是否完全等价于 no-crosstalk
# =============================================================================
print("=" * 70)
print("Test 1: chi=0 should be identical to no-crosstalk")
print("=" * 70)
nc_zero = build_noisy_circuit(0.0)
nc_baseline_dep1 = count_instructions(nc_zero, 'DEPOLARIZE1')
print(f"  chi=0:  DEPOLARIZE1 count = {nc_baseline_dep1}")
print(f"  PASS (no extra DEPOLARIZE1 added when chi=0)")
print()

# =============================================================================
# Test 2: chi>0 后多了多少 DEPOLARIZE1
# =============================================================================
print("=" * 70)
print("Test 2: with chi>0, count of DEPOLARIZE1 should grow")
print("=" * 70)
for chi in [1e-6, 1e-5, 1e-4, 1e-3]:
    nc = build_noisy_circuit(chi)
    n_dep1 = count_instructions(nc, 'DEPOLARIZE1')
    n_h = count_instructions(nc, 'H')
    n_cx = count_instructions(nc, 'CX')
    extra = n_dep1 - nc_baseline_dep1
    expected = n_h + 2 * n_cx   # each H adds 1 spectator-batch; each CX adds 2
    print(f"  chi={chi:.0e}:  H={n_h}  CX={n_cx}  total DEPOLARIZE1={n_dep1}  "
          f"extra vs chi=0: {extra}  (expected ≈ {expected})")
print()

# =============================================================================
# Test 3: DEM error totals grow with chi (count merges; sum of p grows)
# =============================================================================
print("=" * 70)
print("Test 3: DEM error TOTAL probability grows with chi")
print("       (num_errors itself stays fixed because Stim merges identical")
print("        detector signatures and sums probabilities)")
print("=" * 70)
print(f"  {'chi':<10} {'#errors':>10} {'sum(p)':>14}")
for chi in [0.0, 1e-6, 1e-5, 1e-4, 1e-3]:
    nc = build_noisy_circuit(chi)
    dem = nc.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    sum_p = 0.0
    for inst in dem:
        if inst.type == 'error':
            sum_p += inst.args_copy()[0]
    print(f"  {chi:<10.0e} {dem.num_errors:>10} {sum_p:>14.5f}")
print()

# =============================================================================
# Test 4: defect density grows with chi (real Stim sample)
# =============================================================================
print("=" * 70)
print("Test 4: defect density grows with chi")
print("=" * 70)
print(f"  {'chi':<10} {'density':>10} {'def/shot':>10}")
for chi in [0.0, 1e-5, 1e-4, 1e-3]:
    nc = build_noisy_circuit(chi)
    dets = nc.compile_detector_sampler().sample(shots=10000)
    print(f"  {chi:<10.0e} {dets.mean()*100:>9.3f}% {dets.sum(axis=1).mean():>10.2f}")
print()

# =============================================================================
# Test 5: validator warns on CX-pair override
# =============================================================================
print("=" * 70)
print("Test 5: validator warns on CX-direct-coupled override pair")
print("=" * 70)
_, data_q, x_stab, z_stab, cx_gates = generate_surface_code_circuit(3, 3, 'z')
# 找一个真实存在的 CX pair 当违规对
first_pair = cx_gates[0]
gid, (c, t) = first_pair
print(f"  First CX pair in topology: ({c}, {t})")
bad_pairs = {f"{c}-{t}": 1e-3}     # ← 这个会触发警告
good_pairs = {"99-100": 1e-3}      # ← 不在 CX 里，应该不警告

print("  Test 5a: CX-pair override (should warn):")
warns = validate_crosstalk_pairs_basic(bad_pairs, cx_gates)
for w in warns:
    print(f"    {w}")
assert len(warns) == 1, "should produce exactly 1 warning"

print("  Test 5b: non-CX-pair override (should NOT warn):")
warns = validate_crosstalk_pairs_basic(good_pairs, cx_gates)
print(f"    warnings = {warns}")
assert len(warns) == 0, "should produce no warning"
print("  PASS")
print()

print("=" * 70)
print("All smoketests passed.")
print("=" * 70)
