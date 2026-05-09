# Surface Code Variants in PAEMS

`Surface_Code_Simulation/surface_code_generate_circuits.py` provides a single
entry point — `generate_surface_code_circuit(...)` — that returns a noiseless
stim `Circuit` plus the structural information PAEMS needs for noise
injection (`data_qubits`, `x_stabilizers`, `z_stabilizers`, `gate_pairs`).

Three surface code variants are supported through one parameter,
`code_variant`:

| Variant | `code_variant` | Topology source | Use case |
|---|---|---|---|
| **Standard rotated CSS** | `'css'` (default) | stim's built-in `surface_code:rotated_memory_z\|x` | Most QEC research; matches the bulk of stim-based experiments |
| **XZZX (any size)** | `'xzzx'` | Pure-Python builder, any `(distance, rounds)` | Distance scaling, biased-noise experiments, Willow-style memory tests |
| **XZZX from chip template** | `'xzzx'` + `xzzx_template=...` | Loaded from an external noiseless `.stim` file (e.g. Google `circuit_ideal.stim`) | Matching real-chip qubit numbering / boundary auxiliary qubits |
| **Custom .stim** | `'xzzx'` + `xzzx_template=any.stim` | Any well-formed noiseless `.stim` file | Importing other code variants (heavy-hex, hexagonal color code, etc.) provided the input contains `QUBIT_COORDS`, gate, and `DETECTOR`/`OBSERVABLE_INCLUDE` instructions |

All four paths return the same 5-tuple, so PAEMS noise injection is
oblivious to the underlying code:

```python
circuit, data_qubits, x_stabilizers, z_stabilizers, gate_pairs = \
    generate_surface_code_circuit(distance, rounds, basis,
                                   code_variant=..., xzzx_template=...)
```

The `gate_pairs` list contains every distinct two-qubit gate `(control, target)`
pair (CX **and** CZ), each assigned a unique gate id. The PAEMS noise model
treats CX and CZ identically (superconducting CX/CZ have similar fidelity and
length on real hardware) — `inject_basic_noise.py` injects
`add_two_gate_noise` for both.

---

## 1. Standard rotated CSS — `code_variant='css'` (default)

Wraps stim's `surface_code:rotated_memory_z` (basis `'z'`) or
`surface_code:rotated_memory_x` (basis `'x'`). Stabilizers are XXXX
(X-checks) and ZZZZ (Z-checks); CX is the entangling gate.

```python
from surface_code_generate_circuits import generate_surface_code_circuit

circuit, data_q, x_stab, z_stab, cx_gates = \
    generate_surface_code_circuit(distance=7, rounds=10, basis='z')
# code_variant='css' is the default — no need to pass it
```

Qubits are renumbered to `1..N` for PAEMS convention.

## 2. XZZX (any size) — `code_variant='xzzx'`, no template

Builds an XZZX rotated surface code memory circuit from scratch via the
CZ-only compilation. Any `distance >= 2` and `rounds >= 2` are supported.

```python
circuit, data_q, x_stab, z_stab, gate_pairs = \
    generate_surface_code_circuit(distance=11, rounds=11, basis='z',
                                   code_variant='xzzx')
```

Each stabilizer measures an alternating `XZZX` Pauli string around its
plaquette. The compilation uses `H(syndrome) → CZ → H(data) → CZ → CZ →
H(data) → CZ → H(syndrome) → MZ(syndrome)` per round; data qubits get an
intermediate H twist between CZ layers.

`x_stabilizers` is empty — XZZX has uniform stabilizers (no separate X / Z
basis groups), and PAEMS's noise model treats them identically. All
syndrome qubits go into `z_stabilizers`.

**Algorithm credit**:

- CZ-only compilation ported from
  [`jetxezarreta/qec-two-level-qubits-circuit-noise-bias`](https://github.com/jetxezarreta/qec-two-level-qubits-circuit-noise-bias)
  (`circuits/CZcompilation_XZZX_surface_code_HybridBiasCLN.py`). All noise
  injection from that source has been stripped — PAEMS injects its own
  physically-motivated noise model on the noiseless skeleton.
- XZZX surface code definition and tailoring properties:
  Bonilla Ataides, Tuckett, Bartlett, Flammia, Brown.
  *The XZZX surface code*. Nature Communications 12, 2172 (2021).
  [arXiv:2009.07851](https://arxiv.org/abs/2009.07851).

## 3. XZZX from real-chip template — `code_variant='xzzx'` + `xzzx_template=path`

For matching a specific chip's qubit layout / boundary auxiliary qubits
exactly (e.g. Google's Willow d=7 publication data):

```python
circuit, data_q, x_stab, z_stab, gate_pairs = \
    generate_surface_code_circuit(
        distance=7, rounds=250, basis='z',
        code_variant='xzzx',
        xzzx_template='/path/to/google/d7_at_q6_7/Z/r250/circuit_ideal.stim')
```

The template must be a **noiseless** `.stim` file. The loader:

1. Renumbers all qubits 1..N (sorted by original stim index)
2. Identifies data vs syndrome qubits by measurement count (data: measured
   exactly once at end; syndrome: measured each round)
3. Extracts every distinct two-qubit gate pair (skipping classical
   sweep_bit / measurement_record controls used for per-shot Pauli-frame
   randomization)
4. Folds qubits with `QUBIT_COORDS` but no measurements (boundary aux /
   dynamical decoupling spectators) into `z_stabilizers` so PAEMS still
   emits per-qubit single-gate noise params for them

Topological equivalence with the synthetic builder has been verified for
Google d=7 r=250: identical syndrome–data adjacency distribution
(`{2: 12, 4: 36}`), identical detector annotation structure, identical
two-qubit gate pair count (168). The only difference is 4 extra
boundary-auxiliary qubits in the Google template that do not participate
in any stabilizer measurement.

## 4. Custom `.stim` — `code_variant='xzzx'` + `xzzx_template=any.stim`

The template loader does **not** assume the input is XZZX — it works on any
well-formed noiseless `.stim` file that contains:

- `QUBIT_COORDS` instructions assigning a coordinate to every qubit
- Gates: any combination of `R / RX / H / CX / CZ / M / MR / MX`
- `DETECTOR` and `OBSERVABLE_INCLUDE` annotations
- (Optional) `REPEAT` blocks with bodies

This means you can drop in an unrotated surface code, a heavy-hex code, a
color code, or any other QEC circuit and get PAEMS-noise injection for free.

```python
circuit, data_q, x_stab, z_stab, gate_pairs = \
    generate_surface_code_circuit(
        distance=None, rounds=None, basis='z',
        code_variant='xzzx',
        xzzx_template='/path/to/your_custom_circuit.stim')
```

If `distance` / `rounds` are passed, the loader emits a warning when the
data-qubit count does not equal `distance ** 2`.

**Limitation**: `inject_basic_noise.py` currently only injects gate noise
on `R / RX / H / CX / CZ / M / MR / MX`. Other instructions (e.g. `S`,
`SQRT_X`, `MY`, `MPP`) pass through unchanged — they remain in the
circuit but receive no PAEMS noise. Extend the dispatch table in
`inject_basic_noise.py` if you need noise on non-Clifford or
multi-qubit-Pauli-measurement gates.

---

## End-to-end pipeline (any variant)

```python
from surface_code_generate_circuits import generate_surface_code_circuit
from inject_basic_noise import inject_surface_code_noise

# Step 1: get a noiseless circuit + structure (any variant)
c, data_q, x_stab, z_stab, gate_pairs = generate_surface_code_circuit(
    distance=7, rounds=10, basis='z', code_variant='xzzx')

# Step 2: inject PAEMS noise from a JSON params file
nc = inject_surface_code_noise(c, data_q, x_stab, z_stab, gate_pairs,
                                params_file='paems_params.json')

# Step 3: sample
det = nc.compile_detector_sampler().sample(shots=10_000)
print(f'defect rate: {det.mean()*100:.2f}%')
```

Generate `paems_params.json` with
`paems_qubit_noise_tiers/gen_level_params.py` (passes through
`--code-variant` / `--xzzx-template` to keep qubit numbering consistent
with the circuit you will sample from).
