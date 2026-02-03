"""
调度工具模块 - 负责任务调度和 schedule 构建
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from src.client.scheduler import ScheduleConfig, build_schedule, TaskName, SampleIndex, Schedule
from src.runner.backend import BackendClient
from src.runner.config import ExperimentConfig, ROOT_DIR
from src.server.tasks.locomo.task import (
    Locomo0Task, Locomo1Task, Locomo2Task, Locomo3Task, Locomo4Task,
    Locomo5Task, Locomo6Task, Locomo7Task, Locomo8Task, Locomo9Task,
)


# 特殊标记：用于表示 session 内容注入
SESSION_INJECTION_MARKER = "__SESSION_INJECTION__"
# 特殊标记：用于表示 replay 模式的测试样本
REPLAY_TEST_MARKER = "__REPLAY_TEST__"

def load_task_instance(task_name: str, exp_cfg: ExperimentConfig):
    """根据 task_name 加载对应的 task 实例（用于 locomo 任务的特殊处理）"""
    # 查找任务配置
    task_cfg = None
    for t in exp_cfg.tasks:
        if t.get("name") == task_name:
            task_cfg = t
            break
    
    if not task_cfg:
        print(f"[load_task_instance] Task config not found for {task_name}")
        return None
    
    config_path = task_cfg.get("config_path")
    if not config_path:
        print(f"[load_task_instance] config_path not found for {task_name}")
        return None
    
    # 加载 YAML 配置
    config_path = ROOT_DIR / config_path
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            task_yaml = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[load_task_instance] Failed to load YAML from {config_path}: {e}")
        return None
    
    # 获取任务特定的配置（如果有）
    # 先获取 default 配置作为基础
    default_cfg = task_yaml.get("default", {})
    task_specific_cfg = task_yaml.get(task_name, {})
    
    # 合并配置：default 作为基础，task_specific 覆盖
    merged_cfg = default_cfg.copy() if default_cfg else {}
    if task_specific_cfg:
        # 合并 parameters（如果存在）
        if "parameters" in task_specific_cfg:
            merged_params = merged_cfg.get("parameters", {}).copy() if merged_cfg.get("parameters") else {}
            merged_params.update(task_specific_cfg.get("parameters", {}))
            merged_cfg["parameters"] = merged_params
        # 如果 task_specific 有 module，也覆盖
        if "module" in task_specific_cfg:
            merged_cfg["module"] = task_specific_cfg["module"]
    
    if not merged_cfg:
        print(f"[load_task_instance] No config found for {task_name} in {config_path}")
        return None
    
    module_path = merged_cfg.get("module", "")
    parameters = merged_cfg.get("parameters", {}) or {}
    
    print(f"[load_task_instance] module_path={module_path}, parameters={parameters}")
    
    # 动态导入并实例化
    try:
        # 支持所有 locomo 任务（locomo-0 到 locomo-9）
        task_classes = {
            "Locomo0Task": Locomo0Task, "Locomo1Task": Locomo1Task,
            "Locomo2Task": Locomo2Task, "Locomo3Task": Locomo3Task,
            "Locomo4Task": Locomo4Task, "Locomo5Task": Locomo5Task,
            "Locomo6Task": Locomo6Task, "Locomo7Task": Locomo7Task,
            "Locomo8Task": Locomo8Task, "Locomo9Task": Locomo9Task
        }
        
        for task_class_name, task_class in task_classes.items():
            if task_class_name in module_path:
                return task_class(**parameters)
        
        print(f"[load_task_instance] Unknown module_path: {module_path}")
        return None
    except Exception as e:
        print(f"[load_task_instance] Failed to instantiate task: {e}")
        import traceback
        traceback.print_exc()
        return None


def is_locomo_task(task_name: str) -> bool:
    """判断是否是 locomo 任务"""
    return task_name in tuple(f"locomo-{i}" for i in range(10))


def build_locomo_session_schedule(
    locomo_task_name: TaskName,
    locomo_task_instance: Any,
    shuffle_enabled: bool,
    seed: int | None,
) -> Schedule:
    """
    构建 locomo 任务的 session 顺序调度（用于 online 模式，cross_task=False）。
    
    按 session 顺序处理：先注入 session1 内容，然后处理 session1 的 QA（可选 shuffle），
    再注入 session2 内容，处理 session2 的 QA，以此类推。
    
    Args:
        locomo_task_name: locomo 任务名称
        locomo_task_instance: locomo 任务实例
        shuffle_enabled: 是否启用 shuffle（只影响每个 session 内的 QA 顺序）
        seed: shuffle 的随机种子
    
    Returns:
        调度序列，其中 session 内容注入使用特殊标记 (SESSION_INJECTION_MARKER, session_id)
    """
    import random as rnd
    
    rng = rnd.Random(seed)
    
    schedule: Schedule = []
    session_ids = locomo_task_instance.session_ids
    print(f"[Locomo Session Schedule] Processing {len(session_ids)} sessions: {session_ids}")
    
    for session_id in session_ids:
        # 1. 插入 session 内容注入标记
        schedule.append((SESSION_INJECTION_MARKER, session_id))
        
        # 2. 获取该 session 的所有 QA 索引
        qa_indices = locomo_task_instance.get_qa_indices_for_session(session_id)
        
        # 3. 如果 shuffle=True，打乱该 session 内的 QA 顺序
        if shuffle_enabled:
            qa_list = list(qa_indices)
            rng.shuffle(qa_list)
            schedule.extend([(locomo_task_name, qa_idx) for qa_idx in qa_list])
            print(f"  -> Session {session_id}: {len(qa_list)} QAs (shuffled)")
        else:
            schedule.extend([(locomo_task_name, qa_idx) for qa_idx in qa_indices])
            print(f"  -> Session {session_id}: {len(qa_indices)} QAs (original order)")
    
    print(f"[Locomo Session Schedule] Total schedule length: {len(schedule)}")
    return schedule


def build_transfer_schedule(
    task_to_indices: Dict[TaskName, List[SampleIndex]],
    transfer_task: TaskName,
    transfer_after_task: TaskName,
    shuffle_enabled: bool,
    seed: int | None,
) -> Schedule:
    """
    构建 transfer 模式的调度：先执行 transfer_task 的所有样本（训练），
    再执行 transfer_after_task 的所有样本（测试）。
    
    Args:
        task_to_indices: 任务到样本索引的映射
        transfer_task: 训练任务名称（update+enhance）
        transfer_after_task: 测试任务名称（仅 enhance）
        shuffle_enabled: 是否启用 shuffle
        seed: shuffle 的随机种子
    
    Returns:
        调度序列：先 transfer_task 的所有样本，再 transfer_after_task 的所有样本
    """
    import random as rnd
    
    schedule: Schedule = []
    
    # 1. 先处理 transfer_task（训练任务）
    if transfer_task not in task_to_indices:
        raise ValueError(f"transfer_task '{transfer_task}' not found in task_to_indices")
    transfer_indices = list(task_to_indices[transfer_task])
    
    if shuffle_enabled:
        rng = rnd.Random(seed)
        rng.shuffle(transfer_indices)
        print(f"[Transfer Schedule] Shuffled {len(transfer_indices)} samples for transfer_task={transfer_task}")
    else:
        print(f"[Transfer Schedule] {len(transfer_indices)} samples for transfer_task={transfer_task} (no shuffle)")
    
    schedule.extend([(transfer_task, idx) for idx in transfer_indices])
    
    # 2. 再处理 transfer_after_task（测试任务）
    if transfer_after_task not in task_to_indices:
        raise ValueError(f"transfer_after_task '{transfer_after_task}' not found in task_to_indices")
    transfer_after_indices = list(task_to_indices[transfer_after_task])
    
    if shuffle_enabled:
        # 使用同一个 seed，但重新创建 RNG 以确保两个任务各自 shuffle
        rng = rnd.Random(seed)
        rng.shuffle(transfer_after_indices)
        print(f"[Transfer Schedule] Shuffled {len(transfer_after_indices)} samples for transfer_after_task={transfer_after_task}")
    else:
        print(f"[Transfer Schedule] {len(transfer_after_indices)} samples for transfer_after_task={transfer_after_task} (no shuffle)")
    
    schedule.extend([(transfer_after_task, idx) for idx in transfer_after_indices])
    
    print(f"[Transfer Schedule] Total schedule length: {len(schedule)} (train={len(transfer_indices)}, test={len(transfer_after_indices)})")
    return schedule


def build_replay_schedule(
    task_to_indices: Dict[TaskName, List[SampleIndex]],
    replay_m: int,
    replay_n: int,
    replay_seed: int,
    shuffle_enabled: bool,
    seed: int | None,
) -> Tuple[Schedule, Dict[int, Dict[str, List[SampleIndex]]]]:
    """
    构建 replay 模式的调度：每 replay_m 个训练样本后，从已学过的所有样本中随机抽样 replay_n 个进行测试。
    
    Args:
        task_to_indices: 任务到样本索引的映射（应该只有一个任务）
        replay_m: 每学习 m 个样本后进行一次测试
        replay_n: 每次测试时从已学过的样本中随机抽样 n 个
        replay_seed: 测试样本随机抽样的种子
        shuffle_enabled: 是否对训练样本进行 shuffle
        seed: 训练样本 shuffle 的随机种子
    
    Returns:
        (schedule, replay_info): 
        - schedule: 调度序列：训练样本和测试样本交替出现
        - replay_info: 每个 replay 批次的信息 {replay_id: {"train": [...], "test": [...]}}
    """
    import random as rnd
    
    if len(task_to_indices) != 1:
        raise ValueError(f"replay mode requires exactly 1 task, but got {len(task_to_indices)} tasks")
    
    task_name = list(task_to_indices.keys())[0]
    all_indices = list(task_to_indices[task_name])
    
    # 1. 准备训练样本（如果 shuffle=True，则 shuffle）
    train_indices = list(all_indices)
    if shuffle_enabled:
        rng = rnd.Random(seed)
        rng.shuffle(train_indices)
        print(f"[Replay Schedule] Shuffled {len(train_indices)} training samples")
    else:
        print(f"[Replay Schedule] {len(train_indices)} training samples (no shuffle)")
    
    # 2. 构建调度：每 replay_m 个训练样本后，从已学过的样本中随机抽样 replay_n 个进行测试
    schedule: Schedule = []
    test_rng = rnd.Random(replay_seed)  # 测试样本随机抽样使用 replay_seed
    
    learned_samples: List[SampleIndex] = []  # 已学过的样本
    replay_info: Dict[int, Dict[str, List[SampleIndex]]] = {}  # 记录每个 replay 批次的信息
    
    replay_id = 1
    for i in range(0, len(train_indices), replay_m):
        # 添加当前 batch 的训练样本
        batch = train_indices[i:i + replay_m]
        schedule.extend([(task_name, idx) for idx in batch])
        learned_samples.extend(batch)
        
        # 从已学过的样本中随机抽样 replay_n 个进行测试
        if len(learned_samples) > 0:
            # 如果已学过的样本少于 replay_n，则全部使用
            n_samples = min(replay_n, len(learned_samples))
            test_samples = test_rng.sample(learned_samples, n_samples)
            # 使用特殊标记来标识测试样本
            schedule.extend([(REPLAY_TEST_MARKER, idx) for idx in test_samples])
            
            # 记录当前 replay 批次的信息
            replay_info[replay_id] = {
                "train": learned_samples.copy(),  # 已学过的所有样本（到当前 replay 为止）
                "test": test_samples.copy()  # 本次测试的样本
            }
            
            print(f"[Replay Schedule] Replay {replay_id}: {len(batch)} new train, {len(test_samples)} test (from {len(learned_samples)} learned)")
            replay_id += 1
    
    print(f"[Replay Schedule] Total schedule length: {len(schedule)} (train={len(train_indices)}, {len(replay_info)} replays)")
    return schedule, replay_info


def build_replay_schedule_for_locomo(
    task_name: TaskName,
    locomo_task_instance: Any,
    shuffle_enabled: bool,
    seed: int | None,
) -> Tuple[Schedule, Dict[int, Dict[str, List[SampleIndex]]]]:
    """
    构建 locomo 任务的 replay 模式调度：按 session 划分，每个 session 作为一个 replay 批次。
    
    对于 locomo 任务，replay 模式按 session 划分：
    - Replay 1: Session 1 的所有 QA（训练），然后 Session 1 的所有 QA（测试）
    - Replay 2: Session 1 + Session 2 的所有 QA（训练），然后从 Session 1 + Session 2 的所有 QA 中抽样（测试）
    - ...
    
    Args:
        task_name: locomo 任务名称
        locomo_task_instance: locomo 任务实例
        shuffle_enabled: 是否对每个 session 内的 QA 进行 shuffle
        seed: shuffle 的随机种子
    
    Returns:
        (schedule, replay_info): 
        - schedule: 调度序列：包含 session 注入标记和 QA 样本
        - replay_info: 每个 replay 批次的信息 {replay_id: {"train": [...], "test": [...]}}
    """
    import random as rnd
    
    rng = rnd.Random(seed) if shuffle_enabled else None
    
    schedule: Schedule = []
    replay_info: Dict[int, Dict[str, List[SampleIndex]]] = {}
    
    session_ids = locomo_task_instance.session_ids
    learned_samples: List[SampleIndex] = []  # 已学过的所有 QA 索引
    
    print(f"[Locomo Replay Schedule] Processing {len(session_ids)} sessions: {session_ids}")
    
    replay_id = 1
    for session_id in session_ids:
        # 1. 注入当前 session 的内容到 memory
        schedule.append((SESSION_INJECTION_MARKER, session_id))
        
        # 2. 获取当前 session 的所有 QA 索引
        session_qa_indices = locomo_task_instance.get_qa_indices_for_session(session_id)
        
        # 3. 如果 shuffle=True，打乱当前 session 内的 QA 顺序
        if shuffle_enabled and rng:
            qa_list = list(session_qa_indices)
            rng.shuffle(qa_list)
            session_qa_indices = qa_list
        
        # 4. 添加当前 session 的 QA 作为训练样本
        schedule.extend([(task_name, qa_idx) for qa_idx in session_qa_indices])
        learned_samples.extend(session_qa_indices)
        
        # 5. 从已学过的所有 QA 中抽样作为测试样本
        # 注意：对于 locomo，虽然忽略 m、n 参数，但我们仍然需要从已学过的 QA 中抽样
        # 这里我们使用当前 session 的所有 QA 作为测试样本（因为用户说"忽略 m、n 参数"）
        # 如果用户希望从所有已学过的 QA 中抽样，可以修改这里
        test_samples = session_qa_indices.copy()  # 使用当前 session 的所有 QA 作为测试样本
        schedule.extend([(REPLAY_TEST_MARKER, qa_idx) for qa_idx in test_samples])
        
        # 6. 记录当前 replay 批次的信息
        replay_info[replay_id] = {
            "train": learned_samples.copy(),  # 已学过的所有 QA（到当前 session 为止）
            "test": test_samples.copy()  # 当前 session 的所有 QA（作为测试样本）
        }
        
        print(f"[Locomo Replay Schedule] Replay {replay_id} (Session {session_id}): {len(session_qa_indices)} train, {len(test_samples)} test (total learned: {len(learned_samples)})")
        replay_id += 1
    
    print(f"[Locomo Replay Schedule] Total schedule length: {len(schedule)} ({len(replay_info)} replays)")
    return schedule, replay_info


def build_mixed_schedule(
    system_memory_tasks: Dict[TaskName, List[SampleIndex]],
    locomo_task_name: TaskName,
    locomo_task_instance: Any,
    shuffle_enabled: bool,
    seed: int | None,
) -> Schedule:
    """
    构建混合调度：先 shuffle system memory 任务，然后按 session 顺序插入 locomo 任务。
    
    Args:
        system_memory_tasks: system memory 任务的样本字典
        locomo_task_name: locomo 任务名称
        locomo_task_instance: locomo 任务实例
        shuffle_enabled: 是否启用 shuffle
        seed: shuffle 的随机种子
    
    Returns:
        混合后的调度序列，其中 session 内容注入使用特殊标记 (SESSION_INJECTION_MARKER, session_id)
    """
    import random as rnd
    
    rng = rnd.Random(seed)
    
    # 1. 先 shuffle system memory 任务的所有样本
    system_memory_schedule: Schedule = []
    for task_name, indices in system_memory_tasks.items():
        for idx in indices:
            system_memory_schedule.append((task_name, idx))
    
    if shuffle_enabled:
        rng.shuffle(system_memory_schedule)
    
    print(f"[Mixed Schedule] Shuffled {len(system_memory_schedule)} system memory samples")
    
    # 2. 对于 locomo 任务，按 session 顺序处理
    if locomo_task_instance is None:
        return system_memory_schedule
    
    session_ids = locomo_task_instance.session_ids
    print(f"[Mixed Schedule] Processing {len(session_ids)} locomo sessions: {session_ids}")
    
    # 3. 为每个 session 准备 QA 列表
    session_qa_map: Dict[int, List[SampleIndex]] = {}
    for session_id in session_ids:
        qa_indices = locomo_task_instance.get_qa_indices_for_session(session_id)
        if shuffle_enabled:
            qa_list = list(qa_indices)
            rng.shuffle(qa_list)
            session_qa_map[session_id] = qa_list
        else:
            session_qa_map[session_id] = list(qa_indices)
        print(f"  -> Session {session_id}: {len(session_qa_map[session_id])} QAs")
    
    # 4. 构建混合调度
    # 策略：按 session 顺序，每个 session 的注入和所有 QA 必须在下一个 session 之前完成
    # 每个 session 的内容分散在分配给它的 db 样本中
    mixed_schedule: Schedule = []
    system_idx = 0  # system memory 样本的当前位置
    
    for session_idx, session_id in enumerate(session_ids):
        qa_list = session_qa_map[session_id]
        if not qa_list:
            continue
        
        # 计算当前 session 可用的 db 样本范围
        remaining_system = len(system_memory_schedule) - system_idx
        if remaining_system <= 0:
            # 如果 db 样本已用完，直接插入该 session 的注入和所有 QA
            mixed_schedule.append((SESSION_INJECTION_MARKER, session_id))
            for qa_idx in qa_list:
                mixed_schedule.append((locomo_task_name, qa_idx))
            print(f"  -> Session {session_id}: injection + {len(qa_list)} QAs (no db samples remaining)")
            continue
        
        # 计算该 session 应该使用多少 db 样本
        # 为后续 session 保留样本（如果不是最后一个 session）
        is_last_session = (session_idx == len(session_ids) - 1)
        if is_last_session:
            # 最后一个 session，使用所有剩余样本
            session_db_count = remaining_system
        else:
            # 不是最后一个 session，按比例分配
            # 简单策略：平均分配剩余样本给所有后续 session（包括当前）
            remaining_sessions = len(session_ids) - session_idx
            session_db_count = remaining_system // remaining_sessions
            session_db_count = max(1, session_db_count)  # 至少使用 1 个样本
        
        # 记录该 session 的 db 样本范围
        session_start_idx = system_idx
        session_end_idx = min(system_idx + session_db_count, len(system_memory_schedule))
        
        # 4.1 在该 session 的 db 样本范围内，随机选择位置插入 session 注入
        if session_db_count > 1:
            # 随机选择注入位置（在前 50% 的范围内，确保后面有足够空间放 QA）
            injection_offset = rng.randint(0, max(1, session_db_count // 2))
            injection_pos = session_start_idx + injection_offset
        else:
            injection_pos = session_start_idx
        
        # 先添加 db 样本到注入位置
        while system_idx < injection_pos and system_idx < session_end_idx:
            mixed_schedule.append(system_memory_schedule[system_idx])
            system_idx += 1
        
        # 4.2 插入 session 内容注入标记
        mixed_schedule.append((SESSION_INJECTION_MARKER, session_id))
        print(f"  -> Inserted session {session_id} injection at position {len(mixed_schedule) - 1}")
        
        # 4.3 将该 session 的所有 QA 分散插入到剩余的 db 样本中（在该 session 的范围内）
        # 计算该 session 剩余的 db 样本数量（注入位置之后，到该 session 结束）
        session_remaining_db = session_end_idx - system_idx
        
        if session_remaining_db > 0 and len(qa_list) > 0:
            # 计算每个 QA 之间的间隔
            if len(qa_list) == 1:
                intervals = [session_remaining_db]
            else:
                # 将剩余的 db 样本分成 len(qa_list) 段
                base_interval = session_remaining_db // (len(qa_list) + 1)
                intervals = []
                remaining = session_remaining_db
                for i in range(len(qa_list)):
                    if i == len(qa_list) - 1:
                        # 最后一个 QA，使用所有剩余位置
                        intervals.append(remaining)
                    else:
                        # 随机选择间隔（在 base_interval 附近波动）
                        interval = base_interval + rng.randint(-base_interval // 2, base_interval // 2)
                        interval = max(1, min(interval, remaining - (len(qa_list) - i - 1)))
                        intervals.append(interval)
                        remaining -= interval
        else:
            # 如果没有剩余的 db 样本，直接插入所有 QA
            intervals = [0] * len(qa_list)
        
        # 4.4 按间隔插入 QA（在该 session 的 db 样本范围内）
        for qa_idx, interval in zip(qa_list, intervals):
            # 先插入 db 样本
            for _ in range(interval):
                if system_idx < session_end_idx and system_idx < len(system_memory_schedule):
                    mixed_schedule.append(system_memory_schedule[system_idx])
                    system_idx += 1
            # 再插入 QA
            mixed_schedule.append((locomo_task_name, qa_idx))
        
        print(f"  -> Inserted {len(qa_list)} QAs for session {session_id} (used {session_db_count} db samples, range: {session_start_idx}-{session_end_idx})")
    
    # 5. 添加剩余的 system memory 样本
    while system_idx < len(system_memory_schedule):
        mixed_schedule.append(system_memory_schedule[system_idx])
        system_idx += 1
    
    print(f"[Mixed Schedule] Final schedule length: {len(mixed_schedule)}")
    print(f"  -> System memory samples: {len(system_memory_schedule)}")
    print(f"  -> Locomo session injections: {len(session_ids)}")
    print(f"  -> Locomo QAs: {sum(len(session_qa_map[sid]) for sid in session_ids)}")
    
    return mixed_schedule


def build_offline_locomo_schedule(
    locomo_task_name: TaskName,
    locomo_task_instance: Any,
    shuffle_enabled: bool,
    seed: int | None,
) -> Schedule:
    """
    构建 offline 模式的 locomo 任务调度：一次性注入所有 session，然后处理所有 QA。

    Offline 模式的特点：
    - 在开头一次性注入所有 session 的内容（使用 SESSION_INJECTION_MARKER）
    - 然后按顺序（或 shuffle）处理所有 QA
    - 这样可以在 train/test 分割时，所有 session 的信息都已经注入到 memory 中

    Args:
        locomo_task_name: locomo 任务名称
        locomo_task_instance: locomo 任务实例
        shuffle_enabled: 是否对 QA 进行 shuffle
        seed: shuffle 的随机种子

    Returns:
        调度序列：先注入所有 session，再处理所有 QA
    """
    import random as rnd

    schedule: Schedule = []
    session_ids = locomo_task_instance.session_ids

    print(f"[Offline Locomo Schedule] Processing {len(session_ids)} sessions: {session_ids}")

    # 1. 在开头一次性注入所有 session
    for session_id in session_ids:
        schedule.append((SESSION_INJECTION_MARKER, session_id))
    print(f"[Offline Locomo Schedule] Added {len(session_ids)} session injection markers at the beginning")

    # 2. 收集所有 QA 索引
    all_qa_indices: List[SampleIndex] = []
    for session_id in session_ids:
        qa_indices = locomo_task_instance.get_qa_indices_for_session(session_id)
        all_qa_indices.extend(qa_indices)

    # 3. 如果 shuffle=True，打乱所有 QA 的顺序
    if shuffle_enabled:
        rng = rnd.Random(seed)
        rng.shuffle(all_qa_indices)
        print(f"[Offline Locomo Schedule] Shuffled {len(all_qa_indices)} QAs")
    else:
        print(f"[Offline Locomo Schedule] {len(all_qa_indices)} QAs (original order)")

    # 4. 添加所有 QA 到 schedule
    for qa_idx in all_qa_indices:
        schedule.append((locomo_task_name, qa_idx))

    print(f"[Offline Locomo Schedule] Total schedule length: {len(schedule)} ({len(session_ids)} injections + {len(all_qa_indices)} QAs)")
    return schedule


