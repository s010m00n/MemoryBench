from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..base import MemoryMechanism
from src.utils.message_schema import (
    extract_message_info,
    enhance_messages_with_memory,
    extract_original_question,
)


def _serialize_history(history: List[Any], template_title: str, where: str) -> List[Dict[str, Any]]:
    """
    将 history 转换为可序列化的格式（JSON 兼容）。
    过滤掉 RewardHistoryItem 等非聊天消息，并将 Pydantic 模型转换为字典。
    关键：必须过滤掉本轮插入的记忆内容，只保留原始的交互。

    Args:
        history: 对话历史
        template_title: 模板标题（用于识别记忆内容）
        where: 插入位置（"tail" 或 "front"）
    """
    template_titles = [template_title]
    serialized = []

    for msg in history:
        role, content, msg_dict = extract_message_info(msg)

        # 跳过无法提取 role 的消息（如 RewardHistoryItem）
        if role is None:
            continue

        # 如果是第一个user消息且包含记忆内容，提取原始问题
        if role == "user" and content:
            # 检查是否包含记忆（使用公共工具的逻辑）
            from src.utils.message_schema import ORIGINAL_QUESTION_SEPARATOR
            has_memory = (
                ORIGINAL_QUESTION_SEPARATOR in str(content) or
                any(title in str(content) for title in template_titles)
            )

            if has_memory:
                # 提取原始问题
                question = extract_original_question([msg], where=where, template_titles=template_titles)
                if question:
                    content = question

        # 如果提取到了完整的消息字典，使用它
        if msg_dict is not None:
            # 确保是字典类型
            if isinstance(msg_dict, dict):
                # 创建新字典，更新content为过滤后的内容
                filtered_msg = dict(msg_dict)
                filtered_msg["content"] = str(content) if content else ""
                serialized.append(filtered_msg)
            else:
                # 如果是其他类型，创建基本结构
                serialized.append({
                    "role": role,
                    "content": str(content) if content else ""
                })
        else:
            # 如果没有提取到完整字典，创建基本结构
            serialized.append({
                "role": role,
                "content": str(content) if content else ""
            })

    return serialized


logger = logging.getLogger(__name__)

# 导入 Mem0 Platform 客户端
try:
    from mem0 import MemoryClient
    HAS_MEM0 = True
except ImportError:
    HAS_MEM0 = False


@dataclass
class Mem0Config:
    api_key: str = ""
    user_id: str = "default"  # 用户自定义的 user_id（不再固定用 task）
    infer: bool = True
    top_k: int = 5
    threshold: Optional[float] = 0.7
    rerank: bool = True
    success_only: bool = True
    reward_bigger_than_zero: bool = False  # True: 只存储 reward>0 的样本，False: 都存储
    prompt_template: str = "Based on your previous interactions, here are relevant memories:\n{memories}"
    where: str = "tail"  # "tail": 记忆放在 user question 后面 | "front": 记忆放在 user question 前面
    # 重试配置
    max_retries: int = -1  # -1 表示无限重试，0 表示不重试，>0 表示最大重试次数
    retry_delay: float = 1.0  # 重试延迟（秒），指数退避的初始值
    retry_backoff: float = 2.0  # 指数退避倍数
    # 等待配置
    wait_time: float = 0.0  # 每次成功添加记忆后等待的时间（秒），用于避免请求过快


class Mem0Memory(MemoryMechanism):
    """
    Mem0 记忆机制：基于 Mem0 Platform 的结构化记忆系统。
    
    参考 Mem0 文档：
    - Platform: https://docs.mem0.ai/platform/quickstart
    - Add Memory: https://docs.mem0.ai/core-concepts/memory-operations/add
    - Search Memory: https://docs.mem0.ai/core-concepts/memory-operations/search
    
    特性：
    - 自动提取结构化记忆（infer=True）或存储原始消息（infer=False）
    - 自动冲突解决和去重（infer=True 时）
    - 语义检索 + 过滤 + 重排序
    - 使用 Mem0 Platform 托管 API
    """
    
    def __init__(self, config: Mem0Config) -> None:
        self.config = config
        self._client: Any = None
        # 提取 template title（从 prompt_template 中提取，用于识别增强后的消息）
        # 例如: "Based on your previous interactions, here are relevant memories:\n{memories}" -> "Based on your previous interactions, here are relevant memories:"
        self.template_title = self.config.prompt_template.split('{memories}')[0].strip()
        self._init_client()
    
    def _init_client(self) -> None:
        """初始化 Mem0 Platform 客户端"""
        if not HAS_MEM0:
            raise ImportError(
                "Mem0 Platform client not available. "
                "Please install: pip install mem0"
            )
        if not self.config.api_key:
            raise ValueError("Mem0 Platform requires api_key in config")
        self._client = MemoryClient(api_key=self.config.api_key)
        logger.info(f"[Mem0Memory] Initialized Platform client with user_id={self.config.user_id}")
    
    def _extract_query(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """
        从 messages 中提取第一个 user message 作为检索 query。
        需要过滤掉插入的记忆内容，只返回原始问题。
        """
        template_titles = [self.template_title]
        return extract_original_question(messages, where=self.config.where, template_titles=template_titles)
    
    def _format_memories(self, memories: Any) -> str:
        """格式化检索到的记忆为文本"""
        if not memories:
            return ""
        
        # mem0 API 可能返回字典格式 {"results": [...]} 或直接返回列表
        if isinstance(memories, dict):
            # 如果返回的是字典，尝试提取 results 键
            if "results" in memories:
                memories = memories["results"]
            else:
                # 如果字典中没有 results，尝试直接使用字典本身
                memories = [memories]
        elif isinstance(memories, str):
            # 如果返回的是字符串，直接返回
            return memories
        elif not isinstance(memories, (list, tuple)):
            # 其他类型，转换为列表
            memories = [memories]
        
        formatted = []
        for mem in memories:
            # 确保 mem 是字典类型
            if isinstance(mem, dict):
                # mem0 返回的记忆格式：{"memory": "...", "metadata": {...}, ...}
                memory_text = mem.get("memory", "") or mem.get("content", "")
                if memory_text:
                    formatted.append(f"- {memory_text}")
            elif isinstance(mem, str):
                # 如果 mem 是字符串，直接使用
                formatted.append(f"- {mem}")
        
        return "\n".join(formatted)
    
    def _inject_memories(
        self,
        messages: List[Dict[str, Any]],
        memory_text: str
    ) -> List[Dict[str, Any]]:
        """将记忆注入到 messages 中，追加到第一个 user message 的末尾"""
        if not memory_text:
            return list(messages) if messages is not None else []

        # 格式化记忆内容
        memory_content = self.config.prompt_template.format(memories=memory_text)

        # 使用公共工具插入记忆
        return enhance_messages_with_memory(messages, memory_content, where=self.config.where)
    
    def use_memory(
        self,
        task: str,
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        基于当前任务名和原始 messages，检索相关记忆并注入。
        """
        enhanced = list(messages) if messages is not None else []
        
        # 提取 query
        query = self._extract_query(messages)
        if not query:
            return enhanced
        
        try:
            # 调用 mem0.search() 检索记忆（使用用户自定义的 user_id）
            # mem0 API 要求 filters 必须包含 user_id，不能为空字典
            # 参考文档: https://docs.mem0.ai/platform/quickstart
            search_kwargs = {
                "query": query,
                "user_id": self.config.user_id,
                "top_k": self.config.top_k,
                "filters": {"user_id": self.config.user_id},  # filters 必须包含 user_id
            }
            if self.config.threshold is not None:
                search_kwargs["threshold"] = self.config.threshold
            if self.config.rerank:
                search_kwargs["rerank"] = True
            
            memories = self._client.search(**search_kwargs)
            
            # 格式化记忆文本
            memory_text = self._format_memories(memories)
            
            # 注入到 messages
            return self._inject_memories(enhanced, memory_text)
        
        except Exception as e:
            logger.warning(f"[Mem0Memory] Search failed: {e}, returning original messages")
            return enhanced
    
    def update_memory(
        self,
        task: str,
        history: List[Dict[str, Any]],
        result: Dict[str, Any]
    ) -> None:
        """
        在单个样本执行结束后调用，将新的轨迹/结果写入 Mem0。
        如果 add 失败，会根据配置进行重试，直到成功或达到最大重试次数。
        """
        finish = result.get("finish", False)
        status = result.get("status", "")
        reward = result.get("reward", 0)
        # success_only 只负责检查是否成功完成（finish 或 status），不涉及 reward
        is_success = finish or status == "completed"
        
        # 过滤：如果 success_only=True，只存储成功完成的样本（不涉及 reward）
        if self.config.success_only and not is_success:
            print(f"[Mem0] Skipping memory storage: success_only=True but sample not completed (finish={finish}, status={status})")
            logger.debug(
                f"[Mem0Memory] Skipping memory storage: success_only=True but sample not completed "
                f"(finish={finish}, status={status})"
            )
            return
        
        # 过滤：如果 reward_bigger_than_zero=True，只存储 reward>0 的样本
        if self.config.reward_bigger_than_zero:
            if reward <= 0:
                print(f"[Mem0] Skipping memory storage: reward_bigger_than_zero=True but reward={reward}")
                logger.debug(
                    f"[Mem0Memory] Skipping memory storage: reward_bigger_than_zero=True but reward={reward}"
                )
                return

        metadata = {
            "task": task,
            "success": is_success,  # 使用计算好的 is_success（finish 或 status=="completed"）
        }
        
        # 将 history 转换为可序列化的格式（过滤 RewardHistoryItem，转换 Pydantic 模型）
        serialized_history = _serialize_history(history, self.template_title, self.config.where)
        
        # 过滤并规范化 messages：Mem0 API 要求每条消息必须有 role 和 content，且 content 不能为空
        # 同时确保消息格式严格符合 Mem0 的要求（只包含 role 和 content 字段）
        # Mem0 API 只接受 role 为 "user" 或 "assistant"，不接受 "system"、"tool" 等
        # 转换规则：
        # - system 和 tool → user
        # - assistant（包括有 tool_calls 的）→ assistant
        # - tool_calls 会转换为文本并合并到 content 中
        filtered_messages = []
        for msg in serialized_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls")  # 检查是否有 tool_calls 字段
            
            # 确保 role 存在
            if not role or not isinstance(role, str):
                continue
            
            # 对于有 tool_calls 的消息，将 tool_calls 信息转换为文本并合并到 content 中
            # 因为 Mem0 API 不支持 tool_calls 字段，需要将其转换为文本
            if tool_calls and isinstance(tool_calls, list) and len(tool_calls) > 0:
                # 将 tool_calls 转换为文本描述
                tool_calls_text = []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        func_name = tc.get("function", {}).get("name", "") if isinstance(tc.get("function"), dict) else ""
                        func_args = tc.get("function", {}).get("arguments", "") if isinstance(tc.get("function"), dict) else ""
                        if func_name:
                            tool_calls_text.append(f"Tool call: {func_name}({func_args})")
                if tool_calls_text:
                    tool_calls_str = "\n".join(tool_calls_text)
                    # 将 tool_calls 信息添加到 content 中
                    if content and str(content).strip():
                        content = f"{content}\n{tool_calls_str}"
                    else:
                        content = tool_calls_str
            
            # 确保 content 存在且有效（tool_calls 可能已经转换为 content）
            if not content or not isinstance(content, str) or not str(content).strip():
                continue
            
            # Mem0 API 只接受 "user" 和 "assistant" role
            # 转换规则：system 和 tool → user，assistant → assistant
            role_lower = str(role).strip().lower()
            if role_lower == "assistant":
                # assistant 保持为 assistant（即使有 tool_calls，也已经合并到 content 中）
                role_lower = "assistant"
            elif role_lower in ("system", "tool"):
                # system 和 tool 转换为 user
                if role_lower == "system":
                    logger.debug(f"[Mem0Memory] Converting system message to user role")
                else:
                    logger.debug(f"[Mem0Memory] Converting tool message to user role")
                role_lower = "user"
            elif role_lower == "user":
                # user 保持为 user
                role_lower = "user"
            else:
                # 其他未知 role，记录警告但转换为 "user"
                logger.warning(f"[Mem0Memory] Unknown role '{role}', converting to 'user'")
                role_lower = "user"
            
            # 规范化消息格式：只保留 role 和 content（移除其他字段，如 tool_calls, function_call 等）
            # 根据 Mem0 文档，messages 应该是 [{"role": "user", "content": "..."}, ...]
            filtered_messages.append({
                "role": role_lower,  # 使用转换后的 role
                "content": str(content).strip()
            })
        
        if not filtered_messages:
            print(f"[Mem0] Skipping memory storage: No valid messages in history after filtering for task={task}, user_id={self.config.user_id}")
            logger.warning(
                f"[Mem0Memory] No valid messages in history after filtering for task={task}, "
                f"user_id={self.config.user_id}, skipping add"
            )
            return
        
        # 添加详细的调试日志
        logger.info(
            f"[Mem0Memory] Attempting to add memory: task={task}, user_id={self.config.user_id}, "
            f"is_success={is_success}, reward={reward}, history_length={len(history)}, "
            f"serialized_length={len(serialized_history)}, filtered_length={len(filtered_messages)}"
        )
        logger.debug(
            f"[Mem0Memory] First 3 filtered messages: {filtered_messages[:3] if len(filtered_messages) >= 3 else filtered_messages}"
        )
        logger.debug(f"[Mem0Memory] Metadata: {metadata}, infer: {self.config.infer}")
        
        # 重试逻辑：一直重试直到成功（如果 max_retries=-1）或达到最大重试次数
        import time
        retry_count = 0
        current_delay = self.config.retry_delay
        
        while True:
            try:
                # 调用 mem0.add() 存储记忆（使用用户自定义的 user_id）
                # 根据 Mem0 文档 (https://docs.mem0.ai/core-concepts/memory-operations/add)：
                # - messages: 必需，格式为 [{"role": "user", "content": "..."}, ...]
                # - user_id: 必需
                # - metadata: 可选，用于过滤和检索
                # - infer: 可选，控制是否提取结构化记忆（默认 True）
                
                # 记录实际发送的数据（用于调试）
                logger.debug(
                    f"[Mem0Memory] Sending to Mem0 API: "
                    f"messages_count={len(filtered_messages)}, user_id={self.config.user_id}, "
                    f"metadata={metadata}, infer={self.config.infer}"
                )
                
                add_result = self._client.add(
                    messages=filtered_messages,
                    user_id=self.config.user_id,
                    metadata=metadata,
                    infer=self.config.infer,
                )

                
                # 检查返回值确认成功
                if add_result and "results" in add_result:
                    num_memories = len(add_result["results"])
                    if num_memories > 0:
                        # 成功：有结果返回
                        logger.debug(
                            f"[Mem0Memory] Successfully added {num_memories} memory(ies) "
                            f"for task={task}, user_id={self.config.user_id}"
                        )
                        # 根据配置等待一段时间，避免请求过快
                        if self.config.wait_time > 0:
                            print(f"[Mem0] Waiting {self.config.wait_time}s after successful add (task={task}, user_id={self.config.user_id})")
                            logger.info(
                                f"[Mem0Memory] Waiting {self.config.wait_time}s after successful add "
                                f"(task={task}, user_id={self.config.user_id})"
                            )
                            time.sleep(self.config.wait_time)
                            print(f"[Mem0] Wait completed, continuing...")
                        return
                    else:
                        # 返回值异常：results 为空
                        logger.warning(
                            f"[Mem0Memory] Add returned empty results for task={task}, "
                            f"user_id={self.config.user_id}, result={add_result}"
                        )
                        # 继续重试
                else:
                    # 返回值异常：没有 results 字段
                    logger.warning(
                        f"[Mem0Memory] Add returned unexpected result for task={task}, "
                        f"user_id={self.config.user_id}, result={add_result}"
                    )
                    # 继续重试
            
            except Exception as e:
                # 检查是否是不可恢复的错误（不应该重试）
                error_type = type(e).__name__
                error_str = str(e)
                
                # 识别网络连接错误（可重试）
                is_network_error = (
                    "disconnected" in error_str.lower() or
                    "connection" in error_str.lower() or
                    "timeout" in error_str.lower() or
                    "network" in error_str.lower() or
                    "Server disconnected" in error_str or
                    "ConnectionError" in error_type or
                    "TimeoutError" in error_type or
                    "RequestException" in error_type
                )
                
                # 对于 400 错误，记录详细的请求信息以便调试
                if "400" in error_str or "Validation" in error_type:
                    logger.error(
                        f"[Mem0Memory] Validation error (400) for task={task}, user_id={self.config.user_id}: {e}"
                    )
                    logger.error(
                        f"[Mem0Memory] Request details: "
                        f"messages_count={len(filtered_messages)}, "
                        f"first_message_role={filtered_messages[0].get('role') if filtered_messages else 'N/A'}, "
                        f"first_message_content_preview={filtered_messages[0].get('content', '')[:100] if filtered_messages else 'N/A'}, "
                        f"metadata={metadata}, infer={self.config.infer}"
                    )
                
                # 不可恢复的错误：认证错误、验证错误等
                if "Authentication" in error_type or "Validation" in error_type or "401" in error_str or "400" in error_str:
                    logger.error(
                        f"[Mem0Memory] Non-retryable error for task={task}, user_id={self.config.user_id}: {e}"
                    )
                    raise  # 直接抛出，不重试
                
                # 可恢复的错误：网络错误、限流错误等
                if is_network_error:
                    logger.warning(
                        f"[Mem0Memory] Network error detected (attempt {retry_count + 1}) "
                        f"for task={task}, user_id={self.config.user_id}: {error_type}: {error_str}"
                    )
                else:
                    logger.warning(
                        f"[Mem0Memory] Add memory failed (attempt {retry_count + 1}) "
                        f"for task={task}, user_id={self.config.user_id}: {error_type}: {error_str}"
                    )
                
                # 检查是否应该继续重试
                if self.config.max_retries == 0:
                    # 不重试
                    logger.error(
                        f"[Mem0Memory] Add memory failed and retry is disabled "
                        f"for task={task}, user_id={self.config.user_id}"
                    )
                    return
                
                # 增加重试计数（无论 max_retries 是多少，都需要记录重试次数）
                retry_count += 1
                
                if self.config.max_retries > 0:
                    # 有最大重试次数限制
                    if retry_count >= self.config.max_retries:
                        logger.error(
                            f"[Mem0Memory] Add memory failed after {retry_count} retries "
                            f"for task={task}, user_id={self.config.user_id}"
                        )
                        return
                
                # 等待后重试（指数退避）
                # 对于 max_retries=-1（无限重试），会一直重试直到成功
                logger.info(
                    f"[Mem0Memory] Retrying add memory in {current_delay:.2f}s "
                    f"(attempt {retry_count + 1}) for task={task}, user_id={self.config.user_id}"
                )
                time.sleep(current_delay)
                current_delay *= self.config.retry_backoff  # 指数退避


def load_mem0_from_yaml(config_path: str) -> Mem0Memory:
    """
    从 memory/mem0/mem0.yaml 读取配置，构造 Mem0Memory。
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    raw = cfg.get("mem0", {}) or {}

    api_key = str(raw.get("api_key", ""))
    user_id = str(raw.get("user_id", "default"))
    infer = bool(raw.get("infer", True))
    top_k = int(raw.get("top_k", 4))
    threshold = raw.get("threshold")
    if threshold is not None:
        threshold = float(threshold)
    rerank = bool(raw.get("rerank", True))
    success_only = bool(raw.get("success_only", True))
    prompt_template = raw.get(
        "prompt_template",
        "Based on your previous interactions, here are relevant memories:\n{memories}"
    )
    where = raw.get("where", "tail")
    # 重试配置
    max_retries = int(raw.get("max_retries", -1))  # -1 表示无限重试（一直重试直到成功）
    retry_delay = float(raw.get("retry_delay", 1.0))  # 重试延迟（秒）
    retry_backoff = float(raw.get("retry_backoff", 2.0))  # 指数退避倍数
    # reward_bigger_than_zero 配置
    reward_bigger_than_zero = bool(raw.get("reward_bigger_than_zero", False))
    # 等待配置
    wait_time = float(raw.get("wait_time", 0.0))  # 每次成功添加记忆后等待的时间（秒）

    config = Mem0Config(
        api_key=api_key,
        user_id=user_id,
        infer=infer,
        top_k=top_k,
        threshold=threshold,
        rerank=rerank,
        success_only=success_only,
        reward_bigger_than_zero=reward_bigger_than_zero,
        prompt_template=prompt_template,
        where=where,
        max_retries=max_retries,
        retry_delay=retry_delay,
        retry_backoff=retry_backoff,
        wait_time=wait_time,
    )

    return Mem0Memory(config)

