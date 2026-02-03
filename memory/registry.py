"""
Memory Mechanism Registry - 统一管理所有记忆机制的注册和加载

统一命名规范：
- 配置中使用 snake_case 命名（如 stream_icl, awm_pro, mems）
- 通过 registry 映射到实际的类和加载函数
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Any

from memory.base import MemoryMechanism


# 记忆机制注册表：name -> (loader_function, default_config_path)
_MEMORY_REGISTRY: Dict[str, tuple[Callable[[str], MemoryMechanism], str]] = {}


def register_memory(
    name: str,
    loader_func: Callable[[str], MemoryMechanism],
    default_config_path: str,
) -> None:
    """
    注册一个记忆机制

    Args:
        name: 记忆机制名称（统一使用 snake_case，如 stream_icl, awm_pro）
        loader_func: 加载函数，接收 config_path 返回 MemoryMechanism 实例
        default_config_path: 默认配置文件路径（相对于项目根目录）
    """
    _MEMORY_REGISTRY[name] = (loader_func, default_config_path)


def get_memory_loader(name: str) -> tuple[Callable[[str], MemoryMechanism], str]:
    """
    获取记忆机制的加载函数和默认配置路径

    Args:
        name: 记忆机制名称

    Returns:
        (loader_func, default_config_path)

    Raises:
        ValueError: 如果记忆机制未注册
    """
    if name not in _MEMORY_REGISTRY:
        available = ", ".join(sorted(_MEMORY_REGISTRY.keys()))
        raise ValueError(
            f"Memory mechanism '{name}' not registered. "
            f"Available mechanisms: {available}"
        )
    return _MEMORY_REGISTRY[name]


def list_available_memories() -> list[str]:
    """返回所有已注册的记忆机制名称"""
    return sorted(_MEMORY_REGISTRY.keys())


# ===== 注册所有记忆机制 =====

def _register_all_memories():
    """注册所有内置的记忆机制"""

    # zero_shot
    from memory.zero_shot.zero_shot import load_zero_shot_from_yaml
    register_memory(
        name="zero_shot",
        loader_func=load_zero_shot_from_yaml,
        default_config_path="memory/zero_shot/zero_shot.yaml",
    )

    # stream_icl (统一使用 snake_case)
    from memory.streamICL.streamICL import load_stream_icl_from_yaml
    register_memory(
        name="stream_icl",
        loader_func=load_stream_icl_from_yaml,
        default_config_path="memory/streamICL/streamICL.yaml",
    )

    # mem0
    from memory.mem0.mem0 import load_mem0_from_yaml
    register_memory(
        name="mem0",
        loader_func=load_mem0_from_yaml,
        default_config_path="memory/mem0/mem0.yaml",
    )

    # mems (统一使用小写)
    from memory.MEMs.MEMs import load_mems_from_yaml
    register_memory(
        name="mems",
        loader_func=load_mems_from_yaml,
        default_config_path="memory/MEMs/MEMs.yaml",
    )

    # awm_pro (统一使用 snake_case)
    from memory.awmPro.awmPro import load_awmpro_from_yaml
    register_memory(
        name="awm_pro",
        loader_func=load_awmpro_from_yaml,
        default_config_path="memory/awmPro/awmPro.yaml",
    )


# 自动注册所有记忆机制
_register_all_memories()
