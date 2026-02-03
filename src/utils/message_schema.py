"""
消息格式兼容层 - 统一处理不同格式的消息对象

这个模块提供了统一的消息解析工具，用于处理多种消息格式：
- 字典类型
- Pydantic RootModel 包装的对象
- Pydantic 模型对象
- RewardHistoryItem 等非聊天消息

同时提供记忆机制通用的消息处理功能：
- 向消息中插入记忆内容
- 从增强后的消息中提取原始问题
- 判断内容是否包含记忆

所有 memory 模块都应该使用这个统一的工具，而不是各自实现。
"""
from typing import Any, Dict, List, Optional


def extract_message_info(msg: Any) -> tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    """
    从消息对象中提取 role、content 和完整消息字典。

    处理多种消息格式：
    - 字典类型：直接使用
    - RootModel 包装的对象：访问 .root 属性
    - RewardHistoryItem 等非聊天消息：返回 None
    - 其他 Pydantic 模型：尝试转换为字典

    Args:
        msg: 消息对象，可以是字典、Pydantic 模型或其他类型

    Returns:
        (role, content, msg_dict) 元组：
        - role: 消息角色（"user", "assistant", "system" 等），如果无法提取则为 None
        - content: 消息内容字符串，如果无法提取则为 ""
        - msg_dict: 完整的消息字典，如果无法转换则为 None

    Examples:
        >>> # 字典类型
        >>> extract_message_info({"role": "user", "content": "Hello"})
        ("user", "Hello", {"role": "user", "content": "Hello"})

        >>> # Pydantic 模型
        >>> from pydantic import BaseModel
        >>> class Message(BaseModel):
        ...     role: str
        ...     content: str
        >>> msg = Message(role="assistant", content="Hi")
        >>> extract_message_info(msg)
        ("assistant", "Hi", {"role": "assistant", "content": "Hi"})

        >>> # 非聊天消息（如 RewardHistoryItem）
        >>> extract_message_info({"reward": 1.0})
        (None, None, None)
    """
    # 如果是字典，直接使用
    if isinstance(msg, dict):
        return msg.get("role"), msg.get("content", ""), msg

    # 如果是 RootModel 包装的对象，访问 .root 属性
    if hasattr(msg, 'root'):
        root = msg.root
        # 检查是否是 RewardHistoryItem（有 reward 属性但没有 role 属性）
        if hasattr(root, 'reward') and not hasattr(root, 'role'):
            return None, None, None  # 跳过 RewardHistoryItem
        # 如果是字典，直接使用
        if isinstance(root, dict):
            return root.get("role"), root.get("content", ""), root
        # 如果是 Pydantic 模型，尝试转换为字典
        if hasattr(root, 'model_dump'):
            root_dict = root.model_dump(exclude_none=True)
            return root_dict.get("role"), root_dict.get("content", ""), root_dict

    # 如果是 RewardHistoryItem 等非聊天消息对象（有 reward 属性但没有 role 属性），跳过
    if hasattr(msg, 'reward') and not hasattr(msg, 'role'):
        return None, None, None

    # 如果是 Pydantic 模型，尝试转换为字典
    if hasattr(msg, 'model_dump'):
        msg_dict = msg.model_dump(exclude_none=True)
        return msg_dict.get("role"), msg_dict.get("content", ""), msg_dict

    # 其他情况，尝试访问属性
    if hasattr(msg, 'role') and hasattr(msg, 'content'):
        return getattr(msg, 'role', None), getattr(msg, 'content', ""), None

    return None, None, None


def is_chat_message(msg: Any) -> bool:
    """
    判断消息是否是聊天消息（有 role 和 content 属性）。

    Args:
        msg: 消息对象

    Returns:
        如果是聊天消息返回 True，否则返回 False

    Examples:
        >>> is_chat_message({"role": "user", "content": "Hello"})
        True
        >>> is_chat_message({"reward": 1.0})
        False
    """
    role, _, _ = extract_message_info(msg)
    return role is not None


def filter_chat_messages(messages: list[Any]) -> list[Dict[str, Any]]:
    """
    从消息列表中过滤出所有聊天消息，并转换为字典格式。

    Args:
        messages: 消息对象列表

    Returns:
        聊天消息字典列表

    Examples:
        >>> messages = [
        ...     {"role": "user", "content": "Hello"},
        ...     {"reward": 1.0},
        ...     {"role": "assistant", "content": "Hi"}
        ... ]
        >>> filter_chat_messages(messages)
        [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
    """
    chat_messages = []
    for msg in messages:
        role, content, msg_dict = extract_message_info(msg)
        if role is not None:
            # 如果有完整的字典，使用字典；否则构造一个
            if msg_dict is not None:
                chat_messages.append(msg_dict)
            else:
                chat_messages.append({"role": role, "content": content})
    return chat_messages


# ============================================================================
# 记忆机制通用函数
# ============================================================================

# 统一的分隔符，用于标记原始问题在 where=front 模式下的位置
ORIGINAL_QUESTION_SEPARATOR = "--- Original Question Below ---"


def enhance_messages_with_memory(
    messages: List[Dict[str, Any]],
    memory_content: str,
    where: str = "tail",
) -> List[Dict[str, Any]]:
    """
    向第一个 user message 插入记忆内容。

    这是所有记忆机制的通用逻辑：找到第一个 user message，根据 where 参数
    决定将记忆内容插入到问题前面还是后面。

    Args:
        messages: 原始消息列表
        memory_content: 要插入的记忆内容（已格式化好的完整文本）
        where: 插入位置
            - "tail": 记忆放在原始问题后面（默认）
            - "front": 记忆放在原始问题前面，使用分隔符标记原始问题位置

    Returns:
        增强后的消息列表（原消息列表的副本）

    Examples:
        >>> messages = [{"role": "user", "content": "What is 1+1?"}]
        >>> memory = "Example: 2+2=4"
        >>> enhanced = enhance_messages_with_memory(messages, memory, where="tail")
        >>> enhanced[0]["content"]
        'What is 1+1?\\n\\nExample: 2+2=4'

        >>> enhanced = enhance_messages_with_memory(messages, memory, where="front")
        >>> enhanced[0]["content"]
        'Example: 2+2=4\\n\\n--- Original Question Below ---\\n\\nWhat is 1+1?'
    """
    enhanced = list(messages) if messages is not None else []

    if not memory_content:
        return enhanced

    # 找到第一个 user message
    for i, msg in enumerate(enhanced):
        role, content, msg_dict = extract_message_info(msg)
        if role == "user":
            content = content if content else ""

            # 根据 where 参数决定记忆位置
            if where == "front":
                # where=front: 记忆在前，使用分隔符标记原始问题
                new_content = f"{memory_content}\n\n{ORIGINAL_QUESTION_SEPARATOR}\n\n{content}"
            else:  # tail
                # where=tail: 原始问题在前，记忆在后
                new_content = f"{content}\n\n{memory_content}"

            # 使用 msg_dict 确保正确处理 Pydantic 模型
            enhanced[i] = {
                **msg_dict,
                "content": new_content
            }
            break

    return enhanced


def extract_original_question(
    messages: List[Dict[str, Any]],
    where: str = "tail",
    template_titles: Optional[List[str]] = None,
) -> Optional[str]:
    """
    从可能包含记忆的消息中提取原始问题。

    处理三种情况：
    1. 消息未增强（没有记忆） -> 直接返回第一个 user message
    2. where=front 模式 -> 使用分隔符提取原始问题
    3. where=tail 模式 -> 使用模板标题分割，取前面部分

    Args:
        messages: 消息列表（可能包含插入的记忆）
        where: 插入位置（与 enhance_messages_with_memory 对应）
            - "tail": 记忆在后面
            - "front": 记忆在前面
        template_titles: 要识别的模板标题列表，用于 where=tail 模式
            例如: ["Here are some examples", "Based on your previous interactions"]

    Returns:
        原始问题文本，如果无法提取则返回 None

    Examples:
        >>> # 未增强的消息
        >>> messages = [{"role": "user", "content": "What is 1+1?"}]
        >>> extract_original_question(messages)
        'What is 1+1?'

        >>> # where=front 增强的消息
        >>> messages = [{"role": "user", "content": "Memory\\n\\n--- Original Question Below ---\\n\\nWhat is 1+1?"}]
        >>> extract_original_question(messages, where="front")
        'What is 1+1?'

        >>> # where=tail 增强的消息
        >>> messages = [{"role": "user", "content": "What is 1+1?\\n\\nHere are some examples"}]
        >>> extract_original_question(messages, where="tail", template_titles=["Here are some examples"])
        'What is 1+1?'
    """
    template_titles = template_titles or []

    for msg in messages:
        role, content, _ = extract_message_info(msg)
        if role == "user":
            content = content if content else ""

            # 情况 1: 检查是否使用了标准分隔符（where=front 模式）
            if ORIGINAL_QUESTION_SEPARATOR in content:
                parts = content.split(ORIGINAL_QUESTION_SEPARATOR, 1)
                if len(parts) > 1:
                    question = parts[1].strip()
                    return question if question else None
                # 如果分隔符后面没有内容，返回 None
                return None

            # 情况 2: where=tail 模式，检查是否包含任一模板标题
            if where == "tail" and template_titles:
                for template_title in template_titles:
                    if template_title in content:
                        # 原始问题在模板标题之前
                        question = content.split(template_title)[0].strip()
                        return question if question else None

            # 情况 3: 未增强的消息，直接返回内容
            return content

    return None


def is_memory_content(content: str, template_titles: List[str]) -> bool:
    """
    判断内容是否包含插入的记忆。

    检查内容是否包含以下特征之一：
    1. 包含标准分隔符（where=front 模式）
    2. 包含任一模板标题

    Args:
        content: 要检查的内容
        template_titles: 模板标题列表

    Returns:
        如果内容包含记忆返回 True，否则返回 False

    Examples:
        >>> is_memory_content("Normal question", ["Examples:"])
        False

        >>> is_memory_content("Examples:\\nSome examples\\n\\nNormal question", ["Examples:"])
        True

        >>> is_memory_content("Memory\\n\\n--- Original Question Below ---\\n\\nQuestion", ["Examples:"])
        True
    """
    if not content:
        return False

    # 检查标准分隔符
    if ORIGINAL_QUESTION_SEPARATOR in content:
        return True

    # 检查模板标题
    for template_title in template_titles:
        if template_title in content:
            return True

    return False


def extract_question_from_history(
    history: List[Dict[str, Any]],
    where: str = "tail",
    template_titles: Optional[List[str]] = None,
) -> Optional[str]:
    """
    从 history（完整的对话历史）中提取原始问题。

    这是 extract_original_question 的别名，语义更清晰。
    用于 update_memory 时从 history 中提取问题。

    Args:
        history: 对话历史
        where: 插入位置
        template_titles: 模板标题列表

    Returns:
        原始问题文本
    """
    return extract_original_question(history, where, template_titles)
