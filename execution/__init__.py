"""
Execution engines for the lifelong-learning benchmark.

当前阶段实现：
- single_agent: 单一 LLM agent 跑完整个样本
"""

from .base import ExecutionEngine  # noqa: F401
# single_agent 是一个子包，实现位于 execution/single_agent/single_agent.py
from .single_agent.single_agent import SingleAgentExecutionEngine  # noqa: F401


