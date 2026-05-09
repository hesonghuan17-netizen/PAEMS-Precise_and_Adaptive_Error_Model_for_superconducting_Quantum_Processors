import math
import stim
import sys
import os
from typing import Dict, List, Set
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)


# ---------------------------------------------------------------------------
# XZZX surface code support
# ---------------------------------------------------------------------------
# CSS rotated surface code: XXXX / ZZZZ stabilizers, CX entangling.
# XZZX surface code: alternating XZZX stabilizers, CZ entangling + intermediate
# H on data qubits to switch basis. Stim cannot generate XZZX natively.
#
# Two ways to build XZZX in PAEMS:
#   (1) generate_noiseless_xzzx(d, r, basis) — pure-Python builder for any
#       distance/rounds. CZ-only compilation, ported from
#       jetxezarreta/qec-two-level-qubits-circuit-noise-bias
#       (circuits/CZcompilation_XZZX_surface_code_HybridBiasCLN.py); algorithm
#       described in Bonilla Ataides et al., Nature Comms 12, 2172 (2021),
#       arXiv:2009.07851. Noise injection stripped — use PAEMS noise model.
#   (2) generate_xzzx_circuit_from_template(stim_path) — load an external
#       noiseless .stim (e.g. Google chip ideal circuits) and strip noise.
#       Use this when the chip-specific qubit numbering matters (matching real
#       chip data); use (1) for synthetic experiments at arbitrary d/r.
#
# Returned tuple matches the CSS path: (circuit, data_qubits, x_stabilizers,
# z_stabilizers, gate_pairs). For XZZX, x_stabilizers is empty and all syndromes
# go into z_stabilizers (PAEMS noise model treats X/Z stabs identically).
# gate_pairs contains BOTH CX and CZ pairs, each with a unique gate_id — PAEMS
# noise model uses one set of params per (control, target) pair regardless of
# gate type (superconducting CX/CZ have similar fidelity/length).


def generate_noiseless_xzzx(distance: int, rounds: int, basis: str = 'z'):
    """Pure-Python noiseless XZZX rotated surface code memory circuit builder.

    Algorithm: CZ-only compilation. Each round = H(syndrome) + CZ_layer_0 +
    H(data) + CZ_layer_1 + CZ_layer_2 + H(data) + CZ_layer_3 + H(syndrome) +
    M(syndrome). Final round measures data qubits.

    Returns (circuit, data_qubits, x_stab=[], z_stab=syndromes, gate_pairs).
    Qubits 1-indexed for PAEMS convention (renumbered after the standard build).

    NOTE on basis convention (see source): the data qubit init pattern is
    INVERTED relative to the chosen-basis label, intrinsic to the CZ-only
    compilation. basis='z' uses RX on data_qubits_z and R on data_qubits_x.
    """
    if rounds < 2:
        raise ValueError("XZZX builder requires rounds >= 2")
    if distance < 2:
        raise ValueError("XZZX builder requires distance >= 2")
    is_memory_H = (basis.lower() == 'z')

    # ---- Layout (data + syndrome positions on complex grid) ----
    x_distance = z_distance = distance
    data_coords: Set[complex] = set()
    x_observable: List[complex] = []
    z_observable: List[complex] = []
    for x in [i + 0.5 for i in range(z_distance)]:
        for y in [i + 0.5 for i in range(x_distance)]:
            q = x * 2 + y * 2j
            data_coords.add(q)
            if y == 0.5:
                z_observable.append(q)
            if x == 0.5:
                x_observable.append(q)

    x_measure_coords: Set[complex] = set()
    z_measure_coords: Set[complex] = set()
    for x in range(z_distance + 1):
        for y in range(x_distance + 1):
            q = x * 2 + y * 2j
            on_b1 = (x == 0 or x == z_distance)
            on_b2 = (y == 0 or y == x_distance)
            parity = ((x % 2) != (y % 2))
            if on_b1 and parity:
                continue
            if on_b2 and not parity:
                continue
            (x_measure_coords if parity else z_measure_coords).add(q)

    def coord_to_idx(q: complex) -> int:
        # math.fmod (not %) — preserves half-integer offsets in data coords
        q = q - math.fmod(q.real, 2) * 1j
        return int(q.real + q.imag * (z_distance + 0.5))

    p2q: Dict[complex, int] = {}
    for q in data_coords:        p2q[q] = coord_to_idx(q)
    for q in x_measure_coords:   p2q[q] = coord_to_idx(q)
    for q in z_measure_coords:   p2q[q] = coord_to_idx(q)
    q2p: Dict[int, complex] = {v: k for k, v in p2q.items()}

    data_qubits = sorted(p2q[q] for q in data_coords)
    measurement_qubits = sorted([p2q[q] for q in x_measure_coords] +
                                [p2q[q] for q in z_measure_coords])

    measure_coord_to_order: Dict[complex, int] = {}
    data_coord_to_order: Dict[complex, int] = {}
    for q in data_qubits:
        data_coord_to_order[q2p[q]] = len(data_coord_to_order)
    for q in measurement_qubits:
        measure_coord_to_order[q2p[q]] = len(measure_coord_to_order)

    # XZZX twist: x_order is y-major (vs standard rotated CSS x-major).
    # This is what makes X-stabilizers pick up the rotated XZZX pattern under
    # CZ-only compilation. z_order is the standard x-major.
    z_order = [-1 - 1j, -1 + 1j, +1 - 1j, +1 + 1j]
    x_order = [-1 - 1j, +1 - 1j, -1 + 1j, +1 + 1j]

    cz_targets: List[List[int]] = [[], [], [], []]
    for k in range(4):
        for measure in sorted(x_measure_coords, key=lambda c: (c.real, c.imag)):
            data = measure + x_order[k]
            if data in p2q:
                cz_targets[k] += [p2q[measure], p2q[data]]
        for measure in sorted(z_measure_coords, key=lambda c: (c.real, c.imag)):
            data = measure + z_order[k]
            if data in p2q:
                cz_targets[k] += [p2q[measure], p2q[data]]

    if is_memory_H:
        data_qubits_x = data_qubits[::2]
        data_qubits_z = data_qubits[1::2]
    else:
        data_qubits_x = data_qubits[1::2]
        data_qubits_z = data_qubits[::2]
    chosen_basis_observable     = z_observable     if is_memory_H else x_observable
    chosen_basis_measure_coords = z_measure_coords if is_memory_H else x_measure_coords

    def append_extraction_body(circ: stim.Circuit):
        circ.append("TICK", []); circ.append("H", measurement_qubits)
        circ.append("TICK", []); circ.append("CZ", cz_targets[0])
        circ.append("TICK", []); circ.append("H", data_qubits)
        circ.append("TICK", []); circ.append("CZ", cz_targets[1])
        circ.append("TICK", []); circ.append("CZ", cz_targets[2])
        circ.append("TICK", []); circ.append("H", data_qubits)
        circ.append("TICK", []); circ.append("CZ", cz_targets[3])
        circ.append("TICK", []); circ.append("H", measurement_qubits)

    head = stim.Circuit()
    for k, v in sorted(q2p.items()):
        head.append("QUBIT_COORDS", [k], [v.real, v.imag])
    head.append("TICK", [])
    head.append("RX", data_qubits_z)
    head.append("R",  data_qubits_x)
    head.append("R",  measurement_qubits)
    append_extraction_body(head)
    head.append("TICK", []); head.append("MZ", measurement_qubits)
    for measure in sorted(chosen_basis_measure_coords, key=lambda c: (c.real, c.imag)):
        head.append("DETECTOR",
                    [stim.target_rec(-len(measurement_qubits) + measure_coord_to_order[measure])],
                    [measure.real, measure.imag, 0.0])

    body = stim.Circuit()
    body.append("TICK", []); body.append("R", measurement_qubits)
    append_extraction_body(body)
    body.append("TICK", []); body.append("MZ", measurement_qubits)
    m = len(measurement_qubits)
    body.append("SHIFT_COORDS", [], [0.0, 0.0, 1.0])
    # NOTE: emit DETECTOR for ALL stabilizers (not just chosen-basis) — both
    # X- and Z-basis stabilizers commute with the logical observable and give
    # consistent XOR detectors across rounds. Matches Google si1000 behavior.
    for m_index in measurement_qubits:
        m_coord = q2p[m_index]
        k = m - measure_coord_to_order[m_coord] - 1
        body.append("DETECTOR",
                    [stim.target_rec(-k - 1), stim.target_rec(-k - 1 - m)],
                    [m_coord.real, m_coord.imag, 0.0])

    tail = stim.Circuit()
    tail.append("TICK", []); tail.append("R", measurement_qubits)
    append_extraction_body(tail)
    tail.append("TICK", []); tail.append("MZ", measurement_qubits)
    tail.append("SHIFT_COORDS", [], [0.0, 0.0, 1.0])
    # Same as body: emit ALL stabilizer XOR detectors (full Google si1000 match).
    for m_index in measurement_qubits:
        m_coord = q2p[m_index]
        k = m - measure_coord_to_order[m_coord] - 1
        tail.append("DETECTOR",
                    [stim.target_rec(-k - 1), stim.target_rec(-k - 1 - m)],
                    [m_coord.real, m_coord.imag, 0.0])
    for q in data_qubits:
        tail.append("M" + ("ZX"[q in data_qubits_z]), [q])
    for measure in sorted(chosen_basis_measure_coords, key=lambda c: (c.real, c.imag)):
        recs: List[int] = []
        for delta in z_order:
            data = measure + delta
            if data in p2q:
                recs.append(-len(data_qubits) + data_coord_to_order[data])
        recs.append(-len(data_qubits) - len(measurement_qubits) + measure_coord_to_order[measure])
        recs.sort(reverse=True)
        tail.append("DETECTOR", [stim.target_rec(x) for x in recs],
                    [measure.real, measure.imag, 1.0])
    obs_inc = sorted([-len(data_qubits) + data_coord_to_order[q]
                      for q in chosen_basis_observable], reverse=True)
    tail.append("OBSERVABLE_INCLUDE", [stim.target_rec(x) for x in obs_inc], 0.0)

    raw_circuit = head + body * (rounds - 2) + tail

    # Renumber qubits 1..N (PAEMS convention) and reorganize into the API
    # tuple expected by inject_basic_noise.py.
    return _xzzx_renumber_and_extract(raw_circuit)


def _xzzx_renumber_and_extract(template: stim.Circuit):
    """Walk a (noiseless) XZZX stim.Circuit, renumber qubits to 1..N, identify
    data vs syndrome qubits, extract all 2-qubit gate pairs (CX + CZ), and
    return the standard 5-tuple (circuit, data_q, x_stab=[], z_stab, gate_pairs).
    Shared by template loader and pure-Python builder.
    """
    # Step 1: collect QUBIT_COORDS, renumber 1..N (sorted by orig id)
    qubit_coords = {}
    for inst in template:
        if inst.name == "QUBIT_COORDS":
            qubit_coords[inst.targets_copy()[0].value] = inst.gate_args_copy()
    sorted_orig = sorted(qubit_coords.keys())
    original_to_new = {orig: i + 1 for i, orig in enumerate(sorted_orig)}

    # Step 2: data vs syndrome via measurement-count heuristic
    meas_count = {}
    def _count_into(circ, dest):
        for inst in circ:
            if inst.name == "REPEAT":
                sub = {}
                _count_into(inst.body_copy(), sub)
                for q, n in sub.items():
                    dest[q] = dest.get(q, 0) + n * inst.repeat_count
            elif inst.name in ("M", "MR", "MX", "MY", "MZ"):
                for t in inst.targets_copy():
                    if t.is_qubit_target:
                        dest[t.value] = dest.get(t.value, 0) + 1
    _count_into(template, meas_count)

    syndrome_origs = {q for q, n in meas_count.items() if n > 1}
    data_origs = {q for q, n in meas_count.items() if n == 1}
    other_origs = set(qubit_coords.keys()) - syndrome_origs - data_origs
    syndrome_origs |= other_origs

    data_qubits = sorted(original_to_new[o] for o in data_origs if o in original_to_new)
    z_stabilizers = sorted(original_to_new[o] for o in syndrome_origs if o in original_to_new)
    x_stabilizers = []

    # Step 3: extract gate pairs (skip sweep_bit / measurement_record controls)
    gate_pairs = []
    seen_pair_to_gid = {}
    next_gid = 1
    def _collect_pairs(circ):
        nonlocal next_gid
        for inst in circ:
            if inst.name == "REPEAT":
                _collect_pairs(inst.body_copy())
            elif inst.name in ("CX", "CZ"):
                ts = inst.targets_copy()
                for j in range(0, len(ts), 2):
                    if not (ts[j].is_qubit_target and ts[j + 1].is_qubit_target):
                        continue
                    c, t = original_to_new[ts[j].value], original_to_new[ts[j + 1].value]
                    if (c, t) not in seen_pair_to_gid:
                        seen_pair_to_gid[(c, t)] = next_gid
                        gate_pairs.append((next_gid, (c, t)))
                        next_gid += 1
    _collect_pairs(template)

    # Step 4: rebuild circuit with renumbered qubit IDs
    new_circuit = stim.Circuit()
    for nid in range(1, len(sorted_orig) + 1):
        orig = sorted_orig[nid - 1]
        x, y = qubit_coords[orig]
        new_circuit.append("QUBIT_COORDS", [nid], [x, y])

    def _renum_targets(inst):
        out = []
        for t in inst.targets_copy():
            if t.is_qubit_target and t.value in original_to_new:
                out.append(original_to_new[t.value])
            else:
                out.append(t)
        return out

    def _copy_circ(src, dst):
        for inst in src:
            if inst.name == "QUBIT_COORDS":
                continue
            if inst.name == "REPEAT":
                body = stim.Circuit()
                _copy_circ(inst.body_copy(), body)
                dst.append(stim.CircuitRepeatBlock(inst.repeat_count, body))
                continue
            tgs = _renum_targets(inst)
            args = inst.gate_args_copy()
            dst.append(inst.name, tgs, args)
    _copy_circ(template, new_circuit)

    return new_circuit, data_qubits, x_stabilizers, z_stabilizers, gate_pairs

def generate_xzzx_circuit_from_template(template_file, distance=None, rounds=None):
    """Load a noiseless XZZX surface code template (e.g. Google's
    circuit_ideal.stim) and convert it into PAEMS's standard tuple format.

    Args:
    - template_file: path to noiseless .stim template
    - distance, rounds: optional; used for sanity-check warning only

    Returns:
    - new_circuit, data_qubits, x_stabilizers (=[]), z_stabilizers, gate_pairs
    """
    template = stim.Circuit.from_file(template_file)
    out = _xzzx_renumber_and_extract(template)
    if distance is not None:
        expected_data = distance * distance
        if len(out[1]) != expected_data:
            print(f"[XZZX template] WARNING: expected {expected_data} data qubits "
                  f"for d={distance}, found {len(out[1])}")
    return out


def generate_surface_code_circuit(distance, rounds, basis='z',
                                   code_variant='css', xzzx_template=None):
    """
    生成基于CX门的表面码电路，量子比特从1开始连续编号

    Args:
    - distance: 表面码距离
    - rounds: 纠错轮数
    - basis: 'z' 或 'x'，指定Z基或X基表面码

    Returns:
    - circuit: 转换后的电路
    - data_qubits: 数据比特序号列表
    - x_stabilizers: X稳定子比特序号列表
    - z_stabilizers: Z稳定子比特序号列表
    - cx_gates: CX门的映射列表，格式为[(1, (control, target)), (2, (control, target)), ...]
    """
    # XZZX variant: either load from external noiseless template (e.g. Google
    # circuit_ideal.stim, when matching real-chip qubit numbering matters) or
    # build noiseless XZZX from scratch via the pure-Python builder (any d/r).
    if code_variant == 'xzzx':
        if xzzx_template is not None:
            return generate_xzzx_circuit_from_template(xzzx_template,
                                                        distance=distance,
                                                        rounds=rounds)
        return generate_noiseless_xzzx(distance, rounds, basis=basis)
    if code_variant != 'css':
        raise ValueError(f"code_variant must be 'css' or 'xzzx', got {code_variant!r}")

    # 根据basis选择表面码类型
    if basis.lower() == 'z':
        code_type = "surface_code:rotated_memory_z"
    elif basis.lower() == 'x':
        code_type = "surface_code:rotated_memory_x"
    else:
        raise ValueError("basis must be 'z' or 'x'")

    # 生成原始电路
    base_circuit = stim.Circuit.generated(
        code_type,
        distance=distance,
        rounds=rounds,
    )

    # 解析量子比特坐标和重新编号
    qubit_coords = {}
    original_to_new = {}
    new_id = 1

    for instruction in base_circuit:
        if instruction.name == "QUBIT_COORDS":
            x, y = instruction.gate_args_copy()
            original_id = instruction.targets_copy()[0].value
            qubit_coords[new_id] = (x, y)
            original_to_new[original_id] = new_id
            new_id += 1

    # 分类量子比特
    data_qubits = []
    x_stabilizers = []
    z_stabilizers = []

    # 找到最后的测量指令获取数据比特
    for instruction in reversed(base_circuit):
        if instruction.name in ["M", "MX"]:
            data_qubits = [original_to_new[t.value] for t in instruction.targets_copy()]
            break

    # 找到第一个H门指令获取X稳定子
    for instruction in base_circuit:
        if instruction.name == "H":
            x_stabilizers = [original_to_new[t.value] for t in instruction.targets_copy()]
            break

    # 其余稳定子比特为Z稳定子
    all_qubits = set(qubit_coords.keys())
    z_stabilizers = list(all_qubits - set(data_qubits) - set(x_stabilizers))

    # 构建新电路并提取CX门
    new_circuit = stim.Circuit()
    cx_gates = []
    cx_index = 1  # CX门编号从1开始

    # 添加量子比特坐标
    for new_id, (x, y) in qubit_coords.items():
        new_circuit.append("QUBIT_COORDS", [new_id], [x, y])

    def process_cx_instruction(instruction):
        """处理CX指令并提取门信息"""
        nonlocal cx_index
        targets = instruction.targets_copy()
        new_targets = []

        # CX门是成对出现的，每两个target为一对
        for j in range(0, len(targets), 2):
            control = original_to_new[targets[j].value]
            target = original_to_new[targets[j + 1].value]

            # 检查这个CX门是否已经存在
            gate_key = (control, target)
            existing_gate_id = None
            for gate_id, (ctrl, tgt) in cx_gates:
                if ctrl == control and tgt == target:
                    existing_gate_id = gate_id
                    break

            if existing_gate_id is None:
                # 新的CX门，分配新ID
                cx_gates.append((cx_index, (control, target)))
                cx_index += 1
            else:
                # 重复的CX门，重用已有ID
                cx_gates.append((existing_gate_id, (control, target)))

            new_targets.extend([control, target])

        return new_targets

    def convert_targets(instruction):
        """转换指令的target"""
        if not instruction.targets_copy():
            return []

        new_targets = []
        for t in instruction.targets_copy():
            if t.value >= 0 and t.value in original_to_new:
                new_targets.append(original_to_new[t.value])
            else:
                new_targets.append(t)
        return new_targets

    # 转换指令
    i = 0
    while i < len(base_circuit):
        instruction = base_circuit[i]

        if instruction.name == "QUBIT_COORDS":
            i += 1
            continue

        elif instruction.name == "CX":
            new_targets = process_cx_instruction(instruction)
            new_circuit.append("CX", new_targets)


        elif instruction.name == "RX":

            # 收集RX的targets

            rx_targets = convert_targets(instruction)

            # 查找后续的R指令并合并

            all_r_targets = rx_targets.copy()

            j = i + 1

            while j < len(base_circuit) and base_circuit[j].name in ["TICK", "R"]:

                if base_circuit[j].name == "R":
                    r_targets = convert_targets(base_circuit[j])

                    all_r_targets.extend(r_targets)

                    # 跳过这个R指令

                    i = j  # 让外层循环跳过这个R指令

                    break

                j += 1

            # 添加合并后的R门和H门

            new_circuit.append("R", all_r_targets)

            new_circuit.append("H", rx_targets)

        elif instruction.name == "MX":
            # 将MX转换为H+M
            new_targets = convert_targets(instruction)
            new_circuit.append("H", new_targets)
            new_circuit.append("M", new_targets, instruction.gate_args_copy())

        elif instruction.name == "REPEAT":
            # 处理REPEAT块
            repeat_count = instruction.repeat_count
            repeat_body = instruction.body_copy()

            # 处理REPEAT体内的指令
            new_repeat_body = stim.Circuit()
            for sub_instruction in repeat_body:
                if sub_instruction.name == "CX":
                    sub_new_targets = process_cx_instruction(sub_instruction)
                    new_repeat_body.append("CX", sub_new_targets)
                else:
                    sub_new_targets = convert_targets(sub_instruction)
                    if sub_new_targets:
                        new_repeat_body.append(sub_instruction.name, sub_new_targets, sub_instruction.gate_args_copy())
                    else:
                        new_repeat_body.append(sub_instruction.name, [], sub_instruction.gate_args_copy())

            # 添加新的REPEAT块 - 手动展开重复内容
            for _ in range(repeat_count):
                new_circuit += new_repeat_body

        else:
            # 其他指令
            new_targets = convert_targets(instruction)
            if new_targets:
                new_circuit.append(instruction.name, new_targets, instruction.gate_args_copy())
            else:
                new_circuit.append(instruction.name, [], instruction.gate_args_copy())

        i += 1

    return new_circuit, data_qubits, x_stabilizers, z_stabilizers, cx_gates
'''
new_circuit, data_qubits, x_stabilizers, z_stabilizers, cx_gates = generate_surface_code_circuit(3, 3, basis='x')

print(new_circuit)
print(data_qubits)
print(x_stabilizers)
print(z_stabilizers)
print(cx_gates)'''





