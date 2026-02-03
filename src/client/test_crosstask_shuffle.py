"""
Lifelong-learning backend demo:

- 读取 configs/assignment/default.yaml，解析任务列表与实验参数
- 调用后端 Controller (默认 http://localhost:5038/api) 的 get_indices
- 使用 src.client.scheduler 中的 ScheduleConfig/build_schedule 生成统一调度序列
- 对若干个样本调用 start_sample / cancel，验证与后端的端到端联通性

运行方式（在项目根目录）：
    python -m src.client.backend_demo
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import requests
import yaml

from .scheduler import ScheduleConfig, build_schedule, TaskName, SampleIndex, Schedule


ROOT_DIR = Path(__file__).resolve().parents[2]
ASSIGNMENT_PATH = ROOT_DIR / "configs" / "assignment" / "default.yaml"

# 默认后端地址，可通过环境变量覆盖
BACKEND_BASE_URL = os.getenv("LLBENCH_BACKEND_URL", "http://localhost:5038/api")


def is_locomo_task(task_name: str) -> bool:
    """判断是否是 locomo 任务"""
    return task_name in tuple(f"locomo-{i}" for i in range(10))


def load_task_instance(task_name: str, tasks_cfg: List[Dict[str, Any]]) -> Optional[Any]:
    """根据 task_name 加载对应的 task 实例（用于 locomo 任务的特殊处理）

    Note: This is a test-specific version that takes tasks_cfg as a list.
    The production version in schedule_utils.py takes ExperimentConfig.
    """
    # 查找任务配置
    task_cfg = None
    for t in tasks_cfg:
        if t.get("name") == task_name:
            task_cfg = t
            break

    if not task_cfg:
        return None

    config_path = task_cfg.get("config_path")
    if not config_path:
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
    default_cfg = task_yaml.get("default", {})
    task_specific_cfg = task_yaml.get(task_name, {})

    # 合并配置
    merged_cfg = default_cfg.copy() if default_cfg else {}
    if task_specific_cfg:
        if "parameters" in task_specific_cfg:
            merged_params = merged_cfg.get("parameters", {}).copy() if merged_cfg.get("parameters") else {}
            merged_params.update(task_specific_cfg.get("parameters", {}))
            merged_cfg["parameters"] = merged_params
        if "module" in task_specific_cfg:
            merged_cfg["module"] = task_specific_cfg["module"]

    if not merged_cfg:
        print(f"[load_task_instance] No config found for {task_name} in {config_path}")
        return None

    module_path = merged_cfg.get("module", "")
    parameters = merged_cfg.get("parameters", {}) or {}

    # 动态导入并实例化
    try:
        # 支持所有 locomo 任务（locomo-0 到 locomo-9）
        for i in range(10):
            task_class_name = f"Locomo{i}Task"
            if task_class_name in module_path:
                from src.server.tasks.locomo.task import (
                    Locomo0Task, Locomo1Task, Locomo2Task, Locomo3Task, Locomo4Task,
                    Locomo5Task, Locomo6Task, Locomo7Task, Locomo8Task, Locomo9Task
                )
                task_classes = {
                    "Locomo0Task": Locomo0Task, "Locomo1Task": Locomo1Task,
                    "Locomo2Task": Locomo2Task, "Locomo3Task": Locomo3Task,
                    "Locomo4Task": Locomo4Task, "Locomo5Task": Locomo5Task,
                    "Locomo6Task": Locomo6Task, "Locomo7Task": Locomo7Task,
                    "Locomo8Task": Locomo8Task, "Locomo9Task": Locomo9Task
                }
                task_class = task_classes.get(task_class_name)
                if task_class:
                    return task_class(**parameters)

        print(f"[load_task_instance] Unknown module_path: {module_path}")
        return None
    except Exception as e:
        print(f"[load_task_instance] Failed to instantiate task: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_default_config() -> Dict[str, Any]:
    with ASSIGNMENT_PATH.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def build_schedule_from_config(cfg: Dict[str, Any]) -> Tuple[Schedule, Dict[TaskName, List[SampleIndex]]]:
    tasks_cfg = cfg.get("tasks", []) or []
    experiment_cfg = cfg.get("experiment", {}) or {}

    task_names: List[str] = [t["name"] for t in tasks_cfg if "name" in t]

    # 检查是否有 locomo 任务，如果有则加载
    locomo_tasks = [name for name in task_names if is_locomo_task(name)]
    locomo_task_instance = None
    locomo_task_name = None
    if len(locomo_tasks) == 1:
        locomo_task_name = locomo_tasks[0]
        locomo_task_instance = load_task_instance(locomo_task_name, tasks_cfg)
        if locomo_task_instance is None:
            print(f"Warning: Failed to load locomo task {locomo_task_name}, will try to get indices from backend")
        else:
            print(f"[Locomo Task Loaded] {locomo_task_name}")

    # 1) 获取每个任务的 indices
    task_to_indices: Dict[TaskName, List[SampleIndex]] = {}
    for name in task_names:
        # 对于 locomo 任务，直接从任务实例获取 indices（不需要后端注册）
        if is_locomo_task(name):
            if locomo_task_instance is not None and locomo_task_name == name:
                indices = locomo_task_instance.get_indices()
                task_to_indices[name] = indices
                print(f"[Locomo] Got {len(indices)} indices from task instance for {name}")
            else:
                # 如果加载失败，尝试从后端获取（可能会失败）
                print(f"[Locomo] Task instance not loaded for {name}, trying backend...")
                try:
                    url = f"{BACKEND_BASE_URL}/get_indices"
                    resp = requests.get(url, params={"name": name}, timeout=30)
                    resp.raise_for_status()
                    indices = resp.json()
                    if not isinstance(indices, list):
                        raise RuntimeError(f"Unexpected indices response for task {name}: {indices}")
                    task_to_indices[name] = list(indices)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to get indices for locomo task {name} from backend: {e}. "
                        f"Please ensure the task instance can be loaded from config."
                    )
        else:
            # 对于其他任务，从后端获取 indices
            url = f"{BACKEND_BASE_URL}/get_indices"
            resp = requests.get(url, params={"name": name}, timeout=30)
            resp.raise_for_status()
            indices = resp.json()  # 后端约定返回一个列表
            if not isinstance(indices, list):
                raise RuntimeError(f"Unexpected indices response for task {name}: {indices}")
            task_to_indices[name] = list(indices)

    # 2) 根据 cross_task / shuffle / interval 构造 ScheduleConfig
    cross_task = bool(experiment_cfg.get("cross_task", False))
    shuffle_cfg = experiment_cfg.get("shuffle", {}) or {}
    shuffle_enabled = bool(shuffle_cfg.get("enabled", False))
    seed = shuffle_cfg.get("seed", None)
    sched_cfg = ScheduleConfig(
        cross_task=cross_task,
        shuffle=shuffle_enabled,
        seed=seed,
    )

    schedule = build_schedule(task_to_indices, sched_cfg)
    return schedule, task_to_indices


def demo_run_some_samples(schedule: Schedule, max_samples: int = 3, locomo_task_instance: Optional[Any] = None, locomo_task_name: Optional[str] = None) -> None:
    """
    用构造好的 schedule，任选前 max_samples 个样本：
    - 调用 /start_sample
    - 打印返回的 session_id 和任务状态
    - 立即调用 /cancel 结束会话（不跑真实 LLM 推理）
    主要用于验证端到端联通与调度顺序是否符合预期。
    
    对于 locomo 任务，由于不在后端注册，会跳过测试并给出提示。
    """
    to_run = schedule[:max_samples]
    if not to_run:
        print("Schedule is empty; nothing to run.")
        return

    print(f"Will run {len(to_run)} sample(s) from schedule:")
    for i, (task, idx) in enumerate(to_run, start=1):
        print(f"  [{i}] task={task}, index={idx}")

    for task, idx in to_run:
        # 对于 locomo 任务，跳过测试（因为不在后端注册）
        if is_locomo_task(task):
            print(f"\n[start_sample] Skipping locomo task {task} (index={idx})")
            print(f"  -> Note: Locomo tasks are not registered in the backend.")
            print(f"  -> They need to be handled directly by the task instance in the main runner.")
            continue
        
        # 1) start_sample
        start_url = f"{BACKEND_BASE_URL}/start_sample"
        payload = {"name": task, "index": idx}
        print(f"\n[start_sample] POST {start_url} payload={payload}")
        try:
            resp = requests.post(start_url, json=payload, timeout=60)
            resp.raise_for_status()

            # 当前后端版本直接返回 TaskOutput，而不是 {session_id, output}
            # 例如：{"messages": [...], "tools": [...]}
            raw_text = resp.text
            print("  raw response:", raw_text)

            try:
                data = resp.json()
            except Exception as e:
                print(f"  !! Failed to parse JSON response: {e}")
                continue

            messages = data.get("messages", [])
            tools = data.get("tools", [])
            print(f"  -> got TaskOutput: messages={len(messages)}, tools={len(tools)}")
        except requests.exceptions.HTTPError as e:
            print(f"  !! HTTP Error: {e}")
            if e.response is not None:
                print(f"  !! Response: {e.response.text}")
            continue

        # 在真实前端中，这里应该：
        #   - 用 messages 作为对话 history，tools 作为工具声明
        #   - 调用本地 LLM 得到 agent_response
        #   - 然后根据具体后端协议决定是否还需要 /interact（当前部署未暴露 session_id）
        # 这里 demo 只验证 start_sample 和调度是否工作，不继续交互。


def main() -> None:
    print(f"Using backend base URL: {BACKEND_BASE_URL}")

    # 简单健康检查：list_workers
    list_url = f"{BACKEND_BASE_URL}/list_workers"
    try:
        resp = requests.get(list_url, timeout=10)
        resp.raise_for_status()
        print("Controller /list_workers response:")
        print(resp.text)
    except Exception as e:
        print(f"Failed to call {list_url}: {e}")
        print("请确认后端 Controller 已在默认端口 5038 启动，或通过 LLBENCH_BACKEND_URL 覆盖地址。")
        return

    cfg = load_default_config()
    schedule, task_to_indices = build_schedule_from_config(cfg)

    # 加载 locomo 任务实例（如果存在）
    tasks_cfg = cfg.get("tasks", []) or []
    locomo_tasks = [t["name"] for t in tasks_cfg if "name" in t and is_locomo_task(t["name"])]
    locomo_task_instance = None
    locomo_task_name = None
    if len(locomo_tasks) == 1:
        locomo_task_name = locomo_tasks[0]
        locomo_task_instance = load_task_instance(locomo_task_name, tasks_cfg)

    print("\nTasks and available indices:")
    for task, indices in task_to_indices.items():
        # 显示该任务的所有 indices
        print(f"  {task}: {len(indices)} indices -> {indices}")

    max_show = 100
    print(f"\nTotal schedule length: {len(schedule)} (showing first {max_show} entries):")
    for pair in schedule[:max_show]:
        print(" ", pair)

    # 实际跑几个样本做 smoke test
    demo_run_some_samples(schedule, max_samples=3, locomo_task_instance=locomo_task_instance, locomo_task_name=locomo_task_name)


if __name__ == "__main__":
    main()


