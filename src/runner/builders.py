"""
构建器模块 - 负责构建 memory、execution engine 等组件
"""
from __future__ import annotations

from pathlib import Path

from execution.single_agent.single_agent import load_single_agent_engine_from_yaml
from src.runner.config import ExperimentConfig


ROOT_DIR = Path(__file__).resolve().parents[2]


def build_memory_from_config(cfg: ExperimentConfig):
    """
    根据 default.yaml 中的 memory_mechanism 构造记忆机制。

    支持的记忆机制（统一使用 snake_case 命名）：
    - zero_shot: 无记忆基线
    - stream_icl: 流式 In-Context Learning
    - mem0: Mem0 记忆系统
    - mems: Multi-Memory System (MEMs)
    - awm_pro: Agent Workflow Memory Pro

    注意：无论配置什么 memory_mechanism，都会返回对应的实例用于 update_memory。
    但在 offline 模式下，use_memory 会强制使用 zero_shot（见 main 函数）。
    """
    from memory.registry import get_memory_loader, list_available_memories

    mem_cfg = cfg.memory_mechanism or {}
    name = mem_cfg.get("name", "zero_shot")
    config_path = mem_cfg.get("config_path")

    try:
        loader_func, default_config_path = get_memory_loader(name)
    except ValueError as e:
        # 提供友好的错误信息
        available = list_available_memories()
        raise ValueError(
            f"Unknown memory mechanism '{name}'. "
            f"Available options: {', '.join(available)}"
        ) from e

    # 确定配置文件路径
    if not config_path:
        config_path = ROOT_DIR / default_config_path
    else:
        config_path = ROOT_DIR / config_path

    return loader_func(str(config_path))


def build_execution_engine_from_config(cfg: ExperimentConfig):
    """
    根据 assignment.yaml 中的 execution_method 构造执行引擎。
    支持 single_agent。
    """
    exec_cfg = cfg.execution_method or {}
    name = exec_cfg.get("name", "single_agent")
    config_path = exec_cfg.get("config_path")

    if name == "single_agent":
        if not config_path:
            config_path = ROOT_DIR / "execution" / "single_agent" / "single_agent.yaml"
        else:
            config_path = ROOT_DIR / config_path
        return load_single_agent_engine_from_yaml(str(config_path))
    else:
        raise NotImplementedError(f"Execution method '{name}' not implemented yet (supported: single_agent).")


def ensure_output_dir(base: Path) -> Path:
    """确保输出目录存在"""
    base.mkdir(parents=True, exist_ok=True)
    return base


def build_schedule_from_config(
    exp_cfg: ExperimentConfig,
    backend,
    locomo_task_instance=None,
    locomo_task_name=None
):
    """
    统一的调度构建入口，返回完整的调度信息。

    这个函数实现了清晰的调度构建流水线：
    1. 构建任务索引 (build_indices)
    2. 构建基础调度 (build_base_schedule)
    3. 如果是 offline，分割 train/test (split_train_test_if_needed)

    Args:
        exp_cfg: 实验配置
        backend: 后端客户端
        locomo_task_instance: locomo 任务实例（可选）
        locomo_task_name: locomo 任务名称（可选）

    Returns:
        {
            "train_schedule": [...],      # 训练集调度
            "test_schedule": [...],       # 测试集调度（offline 模式）或 None
            "task_to_indices": {...},     # 任务索引映射
            "replay_info": {...},         # replay 信息（replay 模式）或 None
        }
    """
    # 步骤 1: 构建任务索引
    task_to_indices = _build_task_indices(exp_cfg, backend)

    # 步骤 2: 构建基础调度
    base_schedule, replay_info = _build_base_schedule(
        exp_cfg=exp_cfg,
        task_to_indices=task_to_indices,
        locomo_task_instance=locomo_task_instance,
        locomo_task_name=locomo_task_name,
    )

    # 步骤 3: 如果是 offline，分割 train/test
    training_mode = exp_cfg.experiment.get("training_mode", "offline")
    if training_mode == "offline":
        train_schedule, test_schedule = _split_train_test(base_schedule, exp_cfg)
    else:
        train_schedule = base_schedule
        test_schedule = None

    return {
        "train_schedule": train_schedule,
        "test_schedule": test_schedule,
        "task_to_indices": task_to_indices,
        "replay_info": replay_info,
    }


def _build_task_indices(exp_cfg: ExperimentConfig, backend) -> dict:
    """
    步骤 1: 构建任务索引映射

    Args:
        exp_cfg: 实验配置
        backend: 后端客户端

    Returns:
        {task_name: [sample_indices]}
    """
    tasks_cfg = exp_cfg.tasks
    task_names = [t["name"] for t in tasks_cfg if "name" in t]

    task_to_indices = {}
    for task_name in task_names:
        try:
            indices = backend.get_indices(task_name)
            task_to_indices[task_name] = indices
        except Exception as e:
            print(f"Warning: Failed to get indices for task {task_name}: {e}")
            task_to_indices[task_name] = []

    return task_to_indices


def _build_base_schedule(
    exp_cfg: ExperimentConfig,
    task_to_indices: dict,
    locomo_task_instance,
    locomo_task_name: str | None,
):
    """
    步骤 2: 根据训练模式构建基础调度

    Args:
        exp_cfg: 实验配置
        task_to_indices: 任务索引映射
        locomo_task_instance: locomo 任务实例
        locomo_task_name: locomo 任务名称

    Returns:
        (schedule, replay_info) 元组
    """
    from src.runner.schedule_utils import (
        build_transfer_schedule,
        build_replay_schedule,
        build_replay_schedule_for_locomo,
        build_mixed_schedule,
        build_locomo_session_schedule,
        build_offline_locomo_schedule,
    )
    from src.client.scheduler import build_schedule, ScheduleConfig

    # 获取配置
    training_mode = exp_cfg.experiment.get("training_mode", "offline")
    cross_task = exp_cfg.experiment.get("cross_task", False)
    shuffle_cfg = exp_cfg.experiment.get("shuffle", {})
    shuffle_enabled = shuffle_cfg.get("enabled", False) if isinstance(shuffle_cfg, dict) else shuffle_cfg
    seed = shuffle_cfg.get("seed") if isinstance(shuffle_cfg, dict) else None

    replay_info = None

    # 根据 training_mode 构建不同的调度
    if training_mode == "transfer":
        # Transfer 模式：先训练 transfer_task，再测试 transfer_after_task
        transfer_task = exp_cfg.experiment.get("transfer_task")
        transfer_after_task = exp_cfg.experiment.get("transfer_after_task")
        schedule = build_transfer_schedule(
            task_to_indices=task_to_indices,
            transfer_task=transfer_task,
            transfer_after_task=transfer_after_task,
            shuffle_enabled=shuffle_enabled,
            seed=seed,
        )

    elif training_mode == "replay":
        # Replay 模式：周期性测试已学知识
        if locomo_task_name:
            # Locomo 任务的 replay：按 session 划分
            schedule, replay_info = build_replay_schedule_for_locomo(
                task_name=locomo_task_name,
                locomo_task_instance=locomo_task_instance,
                shuffle_enabled=shuffle_enabled,
                seed=seed,
            )
        else:
            # 普通任务的 replay：按 m/n 参数划分
            replay_m = exp_cfg.experiment.get("replay_m")
            replay_n = exp_cfg.experiment.get("replay_n")
            replay_seed = exp_cfg.experiment.get("replay_seed")
            schedule, replay_info = build_replay_schedule(
                task_to_indices=task_to_indices,
                replay_m=replay_m,
                replay_n=replay_n,
                replay_seed=replay_seed,
                shuffle_enabled=shuffle_enabled,
                seed=seed,
            )

    elif training_mode == "offline" and locomo_task_name and locomo_task_instance:
        # Offline 模式 + Locomo 任务：一次性注入所有 session，然后处理所有 QA
        schedule = build_offline_locomo_schedule(
            locomo_task_name=locomo_task_name,
            locomo_task_instance=locomo_task_instance,
            shuffle_enabled=shuffle_enabled,
            seed=seed,
        )

    elif training_mode == "online" and locomo_task_name and locomo_task_instance:
        # Online 模式 + Locomo 任务
        system_memory_tasks = {k: v for k, v in task_to_indices.items() if k != locomo_task_name}
        if system_memory_tasks:
            # 有其他任务，使用混合调度
            schedule = build_mixed_schedule(
                system_memory_tasks=system_memory_tasks,
                locomo_task_name=locomo_task_name,
                locomo_task_instance=locomo_task_instance,
                shuffle_enabled=shuffle_enabled,
                seed=seed,
            )
        else:
            # 只有 locomo 任务，使用 session 调度
            schedule = build_locomo_session_schedule(
                locomo_task_name=locomo_task_name,
                locomo_task_instance=locomo_task_instance,
                shuffle_enabled=shuffle_enabled,
                seed=seed,
            )

    else:
        # 默认：使用 scheduler 的 build_schedule（适用于普通 system memory 任务）
        schedule_cfg = ScheduleConfig(
            cross_task=cross_task,
            shuffle=shuffle_enabled,
            seed=seed,
        )
        schedule = build_schedule(task_to_indices, schedule_cfg)

    return schedule, replay_info


def _split_train_test(schedule, exp_cfg: ExperimentConfig):
    """
    步骤 3: 如果是 offline 模式，分割 train/test

    Args:
        schedule: 基础调度序列
        exp_cfg: 实验配置

    Returns:
        (train_schedule, test_schedule) 元组
    """
    train_size = exp_cfg.experiment.get("train_size", 0.8)
    split_point = int(len(schedule) * train_size)
    train_schedule = schedule[:split_point]
    test_schedule = schedule[split_point:]

    print(f"[Offline Mode] Split schedule: train={len(train_schedule)}, test={len(test_schedule)} (train_size={train_size})")

    return train_schedule, test_schedule
