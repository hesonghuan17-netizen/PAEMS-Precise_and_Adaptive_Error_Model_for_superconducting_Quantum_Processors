import numpy as np
from typing import Dict, List, Tuple


def preprocess_circuit_and_params(circuit, data_qubits: List[int], x_stab: List[int],
                                  z_stab: List[int], cx_gates: List[Tuple], params: Dict):
    """预处理电路结构和参数"""

    # 量子比特处理
    all_qubits = sorted(data_qubits + x_stab + z_stab)
    num_qubits = len(all_qubits)
    qubit_to_idx = {q: i for i, q in enumerate(all_qubits)}

    # 预处理量子比特参数为numpy数组
    lp_array = np.zeros(num_qubits, dtype=np.float32)
    sp_array = np.zeros(num_qubits, dtype=np.float32)

    for i, qubit in enumerate(all_qubits):
        qubit_str = str(qubit)
        if qubit_str in params.get("qubits", {}):
            lp_array[i] = params["qubits"][qubit_str].get("lp", 0.001)
            sp_array[i] = params["qubits"][qubit_str].get("sp", 0.1)
        else:
            lp_array[i] = 0.001
            sp_array[i] = 0.1

    # 预处理CX门参数
    cx_prop_dict = {}
    for gate_id, (control, target) in cx_gates:
        gate_str = str(gate_id)
        prob = 0.1  # 默认
        if gate_str in params.get("cx_gates", {}):
            cx_params = params["cx_gates"][gate_str]
            if (cx_params.get("control") == control and cx_params.get("target") == target):
                prob = cx_params.get("lp_propagation_prob", 0.1)
        cx_prop_dict[(control, target)] = prob

    # 预编译电路指令
    operations = []
    round_boundaries = []
    current_round = 0

    for instruction in circuit:
        name = instruction.name
        targets = [t.value for t in instruction.targets_copy()]

        if name == "H":
            target_indices = [qubit_to_idx[t] for t in targets if t in qubit_to_idx]
            operations.append({
                'type': 'H',
                'target_indices': np.array(target_indices, dtype=np.int32),
                'round': current_round
            })

        elif name == "CX":
            cx_pairs = [(targets[i], targets[i + 1]) for i in range(0, len(targets), 2)]

            # 转换为索引并预处理传染概率
            cx_indices = []
            cx_probs = []
            all_cx_targets = []

            for control, target in cx_pairs:
                if control in qubit_to_idx and target in qubit_to_idx:
                    c_idx = qubit_to_idx[control]
                    t_idx = qubit_to_idx[target]
                    cx_indices.append([c_idx, t_idx])
                    cx_probs.append(cx_prop_dict.get((control, target), 0.1))
                    all_cx_targets.extend([c_idx, t_idx])

            operations.append({
                'type': 'CX',
                'cx_indices': np.array(cx_indices, dtype=np.int32),  # (n_pairs, 2)
                'cx_probs': np.array(cx_probs, dtype=np.float32),
                'all_targets': np.array(sorted(set(all_cx_targets)), dtype=np.int32),
                'round': current_round
            })

        elif name in ["MR", "M"]:
            round_boundaries.append({
                'round': current_round,
                'operation_count': len(operations),
                'measurement_type': name
            })

            if name == "MR":
                current_round += 1

    return {
        'num_qubits': num_qubits,
        'qubit_to_idx': qubit_to_idx,
        'all_qubits': all_qubits,
        'lp_array': lp_array,
        'sp_array': sp_array,
        'operations': operations,
        'round_boundaries': round_boundaries,
        'total_rounds': len(round_boundaries)
    }


def generate_all_randoms(shots: int, preprocessed: Dict):
    """为所有shots预生成随机数"""

    num_operations = len(preprocessed['operations'])
    num_qubits = preprocessed['num_qubits']

    # 为每个operation的每个qubit预生成泄漏和恢复随机数
    leak_randoms = np.random.random((shots, num_operations, num_qubits)).astype(np.float32)
    recover_randoms = np.random.random((shots, num_operations, num_qubits)).astype(np.float32)

    # 为CX门传染预生成随机数
    max_cx_per_op = max(
        len(op.get('cx_indices', [])) for op in preprocessed['operations']
    )
    cx_randoms = np.random.random((shots, num_operations, max_cx_per_op)).astype(np.float32)

    return {
        'leak': leak_randoms,
        'recover': recover_randoms,
        'cx_prop': cx_randoms
    }


def simulate_all_shots_vectorized(preprocessed: Dict, all_randoms: Dict):
    """向量化模拟所有shots"""

    shots = all_randoms['leak'].shape[0]
    num_qubits = preprocessed['num_qubits']
    num_operations = len(preprocessed['operations'])

    # 初始化所有shots的泄漏状态
    all_leakage_states = np.zeros((shots, num_operations + 1, num_qubits), dtype=np.uint8)

    # 预提取参数数组
    lp = preprocessed['lp_array']
    sp = preprocessed['sp_array']

    # 逐操作处理（保持时间顺序）
    for op_idx, operation in enumerate(preprocessed['operations']):
        current_states = all_leakage_states[:, op_idx, :].copy()  # (shots, qubits)

        if operation['type'] == 'H':
            target_indices = operation['target_indices']

            # 向量化更新泄漏状态
            leak_rand = all_randoms['leak'][:, op_idx, target_indices]  # (shots, targets)
            recover_rand = all_randoms['recover'][:, op_idx, target_indices]  # (shots, targets)

            # 当前状态
            current_target_states = current_states[:, target_indices]  # (shots, targets)

            # 泄漏更新：未泄漏 -> 可能泄漏
            leak_mask = (current_target_states == 0) & (leak_rand < lp[target_indices])
            current_states[:, target_indices] = np.where(leak_mask, 1, current_target_states)

            # 恢复更新：已泄漏 -> 可能恢复
            recover_mask = (current_states[:, target_indices] == 1) & (recover_rand < sp[target_indices])
            current_states[:, target_indices] = np.where(recover_mask, 0, current_states[:, target_indices])

        elif operation['type'] == 'CX':
            cx_indices = operation['cx_indices']  # (n_pairs, 2)
            cx_probs = operation['cx_probs']  # (n_pairs,)
            all_targets = operation['all_targets']

            if len(cx_indices) > 0:
                # 1. 先更新所有CX相关量子比特的泄漏状态
                leak_rand = all_randoms['leak'][:, op_idx, all_targets]  # (shots, targets)
                recover_rand = all_randoms['recover'][:, op_idx, all_targets]  # (shots, targets)

                current_target_states = current_states[:, all_targets]

                # 泄漏更新
                leak_mask = (current_target_states == 0) & (leak_rand < lp[all_targets])
                current_states[:, all_targets] = np.where(leak_mask, 1, current_target_states)

                # 恢复更新
                recover_mask = (current_states[:, all_targets] == 1) & (recover_rand < sp[all_targets])
                current_states[:, all_targets] = np.where(recover_mask, 0, current_states[:, all_targets])

                # 2. 处理CX门传染
                for pair_idx, (c_idx, t_idx) in enumerate(cx_indices):
                    control_states = current_states[:, c_idx]  # (shots,)
                    target_states = current_states[:, t_idx]  # (shots,)

                    # 只有一个泄漏时才可能传染
                    one_leaked_mask = (control_states + target_states) == 1

                    if np.any(one_leaked_mask):
                        prop_rand = all_randoms['cx_prop'][:, op_idx, pair_idx]  # (shots,)
                        prop_prob = cx_probs[pair_idx]

                        # 传染条件：只有一个泄漏 且 随机数小于传染概率
                        propagate_mask = one_leaked_mask & (prop_rand < prop_prob)

                        # 传染：两个量子比特都变为泄漏
                        current_states[:, c_idx] = np.where(propagate_mask, 1, current_states[:, c_idx])
                        current_states[:, t_idx] = np.where(propagate_mask, 1, current_states[:, t_idx])

        # 记录更新后的状态
        all_leakage_states[:, op_idx + 1, :] = current_states

    return all_leakage_states


def calculate_affected_states_vectorized(all_leakage_states: np.ndarray, preprocessed: Dict):
    """向量化计算所有shots的affected_states"""

    shots, time_steps, num_qubits = all_leakage_states.shape
    round_boundaries = preprocessed['round_boundaries']
    operations = preprocessed['operations']

    affected_results = []

    for round_info in round_boundaries:
        round_num = round_info['round']
        op_count = round_info['operation_count']

        # 确定轮次的时间范围
        if round_num == 0:
            round_start = 0
        else:
            prev_round = round_boundaries[round_num - 1]
            round_start = prev_round['operation_count'] + 1

        round_end = op_count

        if round_start >= time_steps or round_end >= time_steps:
            # 处理边界情况
            affected_results.append(np.zeros((shots, num_qubits), dtype=np.uint8))
            continue

        # 规则1：时间范围内的泄漏 (向量化OR操作)
        # 一个 qubit 在本轮内任何时刻进入过 |2⟩ -> 本轮所有 measurement 都标记 affected
        affected = np.any(all_leakage_states[:, round_start:round_end + 1, :], axis=1).astype(
            np.uint8)  # (shots, qubits)

        # 规则2 已删除 (2026-04-29):
        #   原本会把 "CX 时一边泄漏 -> 另一边的 measurement 也打 affected" 标记进去。
        #   但 simulate_all_shots_vectorized 里的 lp_propagation_prob 已经在物理层
        #   决定了 partner 是否真正进入 |2⟩；如果真传过去，partner 自己的泄漏被规则 1
        #   自动覆盖；如果没传过去，强行把 partner 标 affected 是双重计数，会让
        #   hop>=2 cross-qubit 关联整体偏高 ~2-3x（验证 d=5 r=30 50K shots）。

        affected_results.append(affected.astype(np.uint8))

    # 修复Z基最后全0的问题
    if len(affected_results) >= 2:
        last_affected = affected_results[-1]
        prev_affected = affected_results[-2]

        # 检查每个shot的最后affected_state是否全0
        all_zero_mask = np.sum(last_affected, axis=1) == 0  # (shots,)
        affected_results[-1][all_zero_mask] = prev_affected[all_zero_mask]

    return np.array(affected_results)  # (rounds, shots, qubits)


def simulate_surface_code_leakage_vectorized(circuit, data_qubits, x_stab, z_stab,
                                             cx_gates, params, shots, batch_size=50):
    """
    混合版本：自动选择处理方式
    """

    if shots <= batch_size:
        # 小规模：完全向量化
        preprocessed = preprocess_circuit_and_params(circuit, data_qubits, x_stab, z_stab, cx_gates, params)
        all_randoms = generate_all_randoms(shots, preprocessed)
        all_leakage_states = simulate_all_shots_vectorized(preprocessed, all_randoms)
        affected_timelines = calculate_affected_states_vectorized(all_leakage_states, preprocessed)
        return affected_timelines.transpose(1, 0, 2)
    else:
        # 大规模：分批处理
        return simulate_surface_code_leakage_batched(
            circuit, data_qubits, x_stab, z_stab, cx_gates, params, shots, batch_size
        )


def extract_measurement_affected_vectorized(all_affected_timelines: np.ndarray,
                                            data_qubits: List[int], x_stab: List[int],
                                            z_stab: List[int], rounds: int):
    """
    向量化提取所有shots的测量影响状态

    Args:
        all_affected_timelines: (shots, rounds, qubits)

    Returns:
        all_measurement_affected: (shots, measurements)
    """
    shots, num_rounds, num_qubits = all_affected_timelines.shape

    all_qubits = sorted(data_qubits + x_stab + z_stab)
    qubit_to_idx = {q: i for i, q in enumerate(all_qubits)}

    stab_qubits = sorted(x_stab + z_stab)
    data_sorted = sorted(data_qubits)

    # 预计算索引
    stab_indices = np.array([qubit_to_idx[q] for q in stab_qubits], dtype=np.int32)
    data_indices = np.array([qubit_to_idx[q] for q in data_sorted], dtype=np.int32)

    measurements_per_shot = rounds * len(stab_qubits) + len(data_sorted)
    all_measurement_affected = np.zeros((shots, measurements_per_shot), dtype=np.uint8)

    # 稳定子轮次
    for r in range(min(rounds, num_rounds)):
        start_idx = r * len(stab_qubits)
        end_idx = start_idx + len(stab_qubits)
        all_measurement_affected[:, start_idx:end_idx] = all_affected_timelines[:, r, stab_indices]

    # 数据测量（使用最后一轮）
    if num_rounds > 0:
        data_start = rounds * len(stab_qubits)
        all_measurement_affected[:, data_start:] = all_affected_timelines[:, -1, data_indices]

    return all_measurement_affected


def simulate_surface_code_leakage_batched(circuit, data_qubits, x_stab, z_stab,
                                          cx_gates, params, shots, batch_size=50):
    """分批处理版本"""
    # 预处理一次（所有batch共享）
    preprocessed = preprocess_circuit_and_params(circuit, data_qubits, x_stab, z_stab, cx_gates, params)

    all_results = []

    for batch_start in range(0, shots, batch_size):
        batch_end = min(batch_start + batch_size, shots)
        batch_shots = batch_end - batch_start

        # 只为当前batch生成随机数
        batch_randoms = generate_all_randoms(batch_shots, preprocessed)

        # 处理当前batch
        batch_leakage_states = simulate_all_shots_vectorized(preprocessed, batch_randoms)
        batch_affected = calculate_affected_states_vectorized(batch_leakage_states, preprocessed)

        all_results.append(batch_affected.transpose(1, 0, 2))

        # 释放内存
        del batch_randoms, batch_leakage_states, batch_affected

    return np.concatenate(all_results, axis=0)