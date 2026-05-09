import stim
import json
from calculate import get_single_gate_noise_from_json, get_readout_idle_noise_from_json, get_two_gate_noise_from_json


def load_surface_code_params(json_file):
    """
    加载表面码参数JSON文件

    Args:
    - json_file: JSON文件路径

    Returns:
    - params_dict: 参数字典
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        params_dict = json.load(f)
    return params_dict


def add_data_init_noise(circuit, data_qubits, params):
    """
    添加数据比特初始化噪声（R门后的X_ERROR）

    Args:
    - circuit: stim.Circuit对象
    - data_qubits: 数据比特列表
    - params: 参数字典
    """
    for qubit in data_qubits:
        qubit_params = params["qubits"][str(qubit)]
        error_rate = qubit_params["data_init_error"]
        if error_rate > 0:
            circuit.append("X_ERROR", qubit, error_rate)


def add_data_measurement_noise(circuit, data_qubits, params):
    """
    添加数据比特测量噪声（M门前的X_ERROR）

    Args:
    - circuit: stim.Circuit对象
    - data_qubits: 数据比特列表
    - params: 参数字典
    """
    for qubit in data_qubits:
        qubit_params = params["qubits"][str(qubit)]
        error_rate = qubit_params["data_measurement_error"]
        if error_rate > 0:
            circuit.append("X_ERROR", qubit, error_rate)


def add_readout_idle_noise(circuit, data_qubits, params):
    """
    添加数据比特在测量期间的空闲噪声

    Args:
    - circuit: stim.Circuit对象
    - data_qubits: 数据比特列表
    - params: 参数字典
    """

    for qubit in data_qubits:
        readout_noise = get_readout_idle_noise_from_json(qubit, params)
        circuit.append("PAULI_CHANNEL_1", qubit, readout_noise)


def add_measurement_spam_noise(circuit, measurement_qubits, params):
    """
    添加测量比特SPAM噪声（MR前的X_ERROR）

    Args:
    - circuit: stim.Circuit对象
    - measurement_qubits: 测量比特列表
    - params: 参数字典
    """
    for qubit in measurement_qubits:
        qubit_params = params["qubits"][str(qubit)]
        error_rate = qubit_params["measurement_spam_rate"]
        if error_rate > 0:
            circuit.append("X_ERROR", qubit, error_rate)


def add_single_gate_noise(circuit, qubit, params):
    """
    添加单量子比特门噪声：PAULI_CHANNEL_1 + DEPOLARIZE1

    Args:
    - circuit: stim.Circuit对象
    - qubit: 量子比特编号
    - params: 参数字典
    """


    noise = get_single_gate_noise_from_json(qubit, params)
    circuit.append("PAULI_CHANNEL_1", qubit, noise['px_py_pz'])
    circuit.append("DEPOLARIZE1", qubit, noise['p1'])


def add_two_gate_noise(circuit, cx_gate_id, params):
    """
    添加双量子比特门噪声：PAULI_CHANNEL_1（每个量子比特） + DEPOLARIZE2

    Args:
    - circuit: stim.Circuit对象
    - cx_gate_id: CX门ID（字符串）
    - params: 参数字典
    """

    noise = get_two_gate_noise_from_json(cx_gate_id, params)

    # 获取CX门的控制和目标量子比特
    cx_params = params["cx_gates"][cx_gate_id]
    control = cx_params["control"]
    target = cx_params["target"]

    circuit.append("PAULI_CHANNEL_1", control, noise['control_pauli'])
    circuit.append("PAULI_CHANNEL_1", target, noise['target_pauli'])
    circuit.append("DEPOLARIZE2", [control, target], noise['p2'])


# =============================================================================
# Crosstalk noise injection (added 2026-04-29)
# =============================================================================
# Model: when qubit `active_q` fires a gate (H or CX), every other qubit
# `spec_q` experiences a depolarizing event with probability χ(active, spec).
# χ is per-gate per-pair. Stim handles Pauli propagation through subsequent
# CX gates automatically; the resulting hyperedges are absorbed into the DEM.
#
# Specification in params JSON:
#   "crosstalk_global": { "default_strength": 1e-5 }
#   "crosstalk_pairs":  { "5-12": {"strength": 1e-4, "type": "physical_proximity"},
#                         "23-89": {"strength": 8e-5, "type": "freq_collision"} }
# `crosstalk_pairs` is optional. Strength can be a bare float instead of a dict.


def build_crosstalk_lookup(chi_pairs):
    """Precompute lookup: for each active qubit, list of (spectator, chi).

    Both directions of each pair are recorded so iteration is symmetric.
    Bare-float strength values are accepted (auto-wrapped).
    Returns: dict[int, list[tuple[int, float]]]
    """
    lookup = {}
    for key, spec in (chi_pairs or {}).items():
        try:
            i, j = (int(x) for x in key.split('-'))
        except ValueError:
            continue
        if isinstance(spec, dict):
            chi = float(spec.get('strength', 0.0))
        else:
            chi = float(spec)
        if chi <= 0:
            continue
        lookup.setdefault(i, []).append((j, chi))
        lookup.setdefault(j, []).append((i, chi))
    return lookup


def add_spectator_crosstalk(circuit, active_q, pair_lookup):
    """Append DEPOLARIZE1 spectator events for qubits paired with active_q.

    Only qubits explicitly listed in crosstalk_pairs receive an event;
    qubits NOT in any pair entry experience no crosstalk (no global default).
    Same-chi spectators are grouped into one DEPOLARIZE1 (Stim auto-merges).
    """
    entries = pair_lookup.get(active_q)
    if not entries:
        return
    by_chi = {}
    for spec_q, chi in entries:
        by_chi.setdefault(chi, []).append(spec_q)
    for chi, qs in by_chi.items():
        if len(qs) == 1:
            circuit.append("DEPOLARIZE1", qs[0], chi)
        else:
            circuit.append("DEPOLARIZE1", qs, chi)


def validate_crosstalk_pairs_basic(chi_pairs, cx_gates_list):
    """Basic semantic check on crosstalk_pairs entries.

    Currently warns when an override pair is also a direct CX-coupled pair
    (because gate noise already covers strong coupling there).
    Returns list of warning strings (empty if all clean).
    """
    cx_set = set()
    for _gid, (c, t) in cx_gates_list:
        cx_set.add((c, t))
        cx_set.add((t, c))
    warnings = []
    for key in chi_pairs:
        try:
            i, j = (int(x) for x in key.split('-'))
        except ValueError:
            warnings.append(f"[CROSSTALK WARN] crosstalk_pairs key '{key}' is malformed (expect 'i-j')")
            continue
        if (i, j) in cx_set:
            warnings.append(
                f"[CROSSTALK WARN] crosstalk_pairs '{key}': directly connected by CX, "
                "spectator crosstalk overlaps with CX gate noise (consider removing)"
            )
    return warnings


def inject_surface_code_noise(base_circuit, data_qubits, x_stabilizers, z_stabilizers,
                              cx_gates, params_file):
    """
    为表面码电路注入噪声通道（不包含泄漏噪声）

    Args:
    - base_circuit: 基础表面码电路
    - data_qubits: 数据比特列表
    - x_stabilizers: X稳定子比特列表
    - z_stabilizers: Z稳定子比特列表
    - cx_gates: CX门映射列表 [(1, (control, target)), ...]
    - params_file: 参数JSON文件路径

    Returns:
    - noisy_circuit: 包含噪声的电路
    """
    # 加载参数
    params = load_surface_code_params(params_file)

    # 创建CX门查找表
    cx_gate_lookup = {}
    for gate_id, (control, target) in cx_gates:
        cx_gate_lookup[(control, target)] = str(gate_id)

    # ----- Crosstalk setup (only via crosstalk_pairs; no global default) -----
    chi_pairs = params.get('crosstalk_pairs', {}) or {}
    pair_lookup = build_crosstalk_lookup(chi_pairs)
    crosstalk_enabled = bool(pair_lookup)
    if crosstalk_enabled:
        warns = validate_crosstalk_pairs_basic(chi_pairs, cx_gates)
        for w in warns:
            print(w)

    noisy_circuit = stim.Circuit()

    # 处理电路指令
    i = 0
    while i < len(base_circuit):
        instruction = base_circuit[i]

        if instruction.name == "QUBIT_COORDS":
            noisy_circuit.append(instruction.name, instruction.targets_copy(), instruction.gate_args_copy())

        elif instruction.name == "R":
            # R门 + 数据比特初始化噪声
            targets = [t.value for t in instruction.targets_copy()]
            noisy_circuit.append("R", targets)

            # 只对数据比特添加初始化噪声
            add_data_init_noise(noisy_circuit, data_qubits, params)

        elif instruction.name == "H":
            targets = [t.value for t in instruction.targets_copy()]

            for target in targets:
                noisy_circuit.append("H", target)
                add_single_gate_noise(noisy_circuit, target, params)
                # NEW: spectator crosstalk from H driven on `target`
                if crosstalk_enabled:
                    add_spectator_crosstalk(noisy_circuit, target, pair_lookup)

        elif instruction.name in ("CX", "CZ"):
            # XZZX surface code uses CZ for stabilizer entangling, CSS uses CX.
            # Both are 2-qubit gates with similar superconducting fidelity/length;
            # PAEMS uses one set of `cx_gates` params keyed by (control, target) pair
            # — the gate type (CX vs CZ) is irrelevant for noise injection here.
            # NOTE: real-chip templates can have classical-control CX (e.g.
            # `CX sweep[k] q` for per-shot Pauli-frame randomization) — those are
            # NOT real entangling gates and must pass through without noise.
            ts = instruction.targets_copy()
            gate_name = instruction.name

            for j in range(0, len(ts), 2):
                t_ctrl, t_tgt = ts[j], ts[j + 1]

                # Pass through with original target objects (preserves sweep_bit
                # / measurement_record semantics) — no noise on classical-control.
                if not (t_ctrl.is_qubit_target and t_tgt.is_qubit_target):
                    noisy_circuit.append(gate_name, [t_ctrl, t_tgt])
                    continue

                control, target = t_ctrl.value, t_tgt.value
                noisy_circuit.append(gate_name, [control, target])

                gate_key = (control, target)
                if gate_key in cx_gate_lookup:
                    cx_gate_id = cx_gate_lookup[gate_key]
                    add_two_gate_noise(noisy_circuit, cx_gate_id, params)

                # spectator crosstalk from BOTH control and target drives
                if crosstalk_enabled:
                    add_spectator_crosstalk(noisy_circuit, control, pair_lookup)
                    add_spectator_crosstalk(noisy_circuit, target, pair_lookup)

        elif instruction.name == "MR":
            # MR前：数据比特空闲噪声（仅中间轮） + 稳定子SPAM噪声

            targets = [t.value for t in instruction.targets_copy()]

            stabilizer_qubits = x_stabilizers + z_stabilizers
            measured_stabilizers = [q for q in targets if q in stabilizer_qubits]

            # 检查是否是最后一轮：查看后续是否还有MR指令
            is_last_round = True
            for j in range(i + 1, len(base_circuit)):
                if base_circuit[j].name == "MR":
                    is_last_round = False
                    break

            # 只在非最后轮添加数据比特空闲噪声
            if not is_last_round:
                add_readout_idle_noise(noisy_circuit, data_qubits, params)

            # 稳定子SPAM噪声
            add_measurement_spam_noise(noisy_circuit, measured_stabilizers, params)

            noisy_circuit.append("MR", targets)

        elif instruction.name == "M":
            # M前添加数据比特测量噪声（最终测量，无空闲噪声）
            targets = [t.value for t in instruction.targets_copy()]
            measured_data = [q for q in targets if q in data_qubits]

            add_data_measurement_noise(noisy_circuit, measured_data, params)
            noisy_circuit.append("M", targets)

        else:
            # 其他指令（DETECTOR, OBSERVABLE_INCLUDE等）直接复制
            if hasattr(instruction, 'targets_copy') and hasattr(instruction, 'gate_args_copy'):
                new_targets = []
                for t in instruction.targets_copy():
                    if hasattr(t, 'value'):
                        new_targets.append(t.value if t.value >= 0 else t)
                    else:
                        new_targets.append(t)
                noisy_circuit.append(instruction.name, new_targets, instruction.gate_args_copy())
            else:
                # 对于没有标准方法的指令，尝试直接添加
                try:
                    noisy_circuit += stim.Circuit([instruction])
                except:
                    pass

        i += 1

    return noisy_circuit


