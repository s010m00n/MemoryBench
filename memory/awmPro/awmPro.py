from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from requests.exceptions import ReadTimeout, Timeout
import yaml

from ..base import MemoryMechanism, parse_llm_json_response
from ..streamICL.streamICL import RAG
from src.utils.message_schema import (
    extract_message_info,
    enhance_messages_with_memory,
    extract_original_question,
)

logger = logging.getLogger(__name__)


def _serialize_history(history: List[Any], template_title: Optional[str] = None, where: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    将 history 转换为可序列化的格式（JSON 兼容）。
    过滤掉 RewardHistoryItem 等非聊天消息，并将 Pydantic 模型转换为字典。
    如果提供了 template_title 和 where，会过滤掉插入的记忆内容。

    Args:
        history: 对话历史
        template_title: 模板标题（用于识别记忆内容），可选
        where: 插入位置（"tail" 或 "front"），可选
    """
    serialized = []
    template_titles = [template_title] if template_title else []

    for msg in history:
        role, content, msg_dict = extract_message_info(msg)

        # 跳过无法提取 role 的消息（如 RewardHistoryItem）
        if role is None:
            continue

        # 如果提供了 template_title 和 where，过滤掉记忆内容
        if role == "user" and content and template_title and where:
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


@dataclass
class AWMProConfig:
    # 模型配置（统一使用一个模型）
    model_name: str

    # Prompt 配置
    workflow_induction_prompt: str
    workflow_management_prompt: str

    # 工作流 RAG 配置（必需字段）
    workflow_rag_embedding_model: str
    workflow_rag_top_k: int
    workflow_rag_order: str
    workflow_rag_seed: int
    workflow_rag_prompt_template: str
    workflow_rag_where: str  # "tail": 记忆放在 user question 后面 | "front": 记忆放在 user question 前面
    workflow_rag_success_only: bool
    workflow_rag_reward_bigger_than_zero: bool

    # 工作流管理配置（必需字段）
    workflow_management_similarity_top_k: int  # 向量搜索时每个新工作流找 top_k 个相似工作流

    # 工作流存储路径（必需字段）
    workflow_storage_path: Path

    # 可选字段（有默认值的字段必须放在最后）
    workflow_induction_max_retries: int = 5  # 工作流归纳最大重试次数，默认5次
    workflow_management_max_retries: int = 5  # 工作流管理最大重试次数，默认5次


class AWMPro(MemoryMechanism):
    """
    Agent Workflow Memory Pro (AWMPro):
    - 专注于工作流提取和管理
    - 从任务轨迹中提取可重用工作流
    - 使用 RAG 进行工作流检索和管理
    """

    def __init__(self, config: AWMProConfig) -> None:
        self.config = config
        self._workflow_storage_path = self.config.workflow_storage_path
        self._workflow_storage_path.parent.mkdir(parents=True, exist_ok=True)

        # 提取 template title（从 workflow_rag_prompt_template 中提取，用于识别增强后的消息）
        # 例如: "Here are some useful workflows:\n{workflows}" -> "Here are some useful workflows:"
        self.template_title = self.config.workflow_rag_prompt_template.split('{workflows}')[0].strip()
        self.where = self.config.workflow_rag_where

        # 初始化工作流 RAG
        self._workflow_rag: Optional[RAG] = None
        try:
            self._workflow_rag = RAG(
                embedding_model=self.config.workflow_rag_embedding_model,
                top_k=self.config.workflow_rag_top_k,
                order=self.config.workflow_rag_order,
                seed=self.config.workflow_rag_seed,
            )
        except ImportError as e:
            logger.warning(f"[AWMPro] Failed to init workflow RAG: {e}. Vector retrieval disabled.")
            self._workflow_rag = None

    def _load_agent_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        读取 LLM HTTP 配置：
        - 从 configs/llmapi/api.yaml + agent.yaml 中为指定模型组装 url / headers / body。
        """
        model_name = (model_name or "").strip()
        if not model_name:
            return None

        # 配置目录相对项目根目录
        root_dir = Path(__file__).resolve().parents[2]
        llmapi_dir = root_dir / "configs" / "llmapi"
        agent_cfg_path = llmapi_dir / "agent.yaml"
        api_cfg_path = llmapi_dir / "api.yaml"

        if not agent_cfg_path.exists() or not api_cfg_path.exists():
            logger.warning(
                f"[AWMPro] LLM config files not found: {agent_cfg_path}, {api_cfg_path}"
            )
            return None

        try:
            with agent_cfg_path.open("r", encoding="utf-8") as f:
                agents_cfg = yaml.safe_load(f) or {}
            if model_name not in agents_cfg:
                logger.warning(
                    f"[AWMPro] Model '{model_name}' not found in {agent_cfg_path}"
                )
                return None
            agent_cfg = agents_cfg[model_name] or {}

            with api_cfg_path.open("r", encoding="utf-8") as f:
                api_cfg = yaml.safe_load(f) or {}

            base_params = api_cfg.get("parameters", {}) or {}
            agent_params = agent_cfg.get("parameters", {}) or {}

            body = dict(base_params.get("body", {}) or {})
            body.update(agent_params.get("body", {}) or {})

            url = base_params.get("url") or api_cfg.get("parameters", {}).get("url")
            if not url:
                logger.warning("[AWMPro] URL not found in api.yaml / agent.yaml")
                return None

            headers = dict(base_params.get("headers", {}) or {})
            headers.update(agent_params.get("headers", {}) or {})

            return {"url": url, "headers": headers, "body": body}
        except Exception as e:
            logger.warning(f"[AWMPro] failed to load agent config: {e}")
            return None

    def _call_llm(self, model_name: str, messages: List[Dict[str, Any]], max_retries: int = 3, purpose: str = "LLM call") -> Optional[str]:
        """调用 LLM API，支持无限重试（max_retries=-1）"""
        print(f"[AWMPro] Calling {purpose} with model={model_name}, messages_count={len(messages)}")
        cfg = self._load_agent_config(model_name)
        if not cfg:
            print(f"[AWMPro] ERROR: Failed to load agent config for model={model_name}")
            return None

        url = cfg["url"]
        headers = cfg["headers"]
        base_body = cfg["body"]
        
        # 记录请求信息（不包含完整 messages，避免日志过长）
        print(f"[AWMPro] {purpose} request: url={url}, model={model_name}, body_keys={list(base_body.keys())}")

        body: Dict[str, Any] = {**(base_body or {}), "messages": messages}
        
        # 获取可用模型列表（用于错误提示）
        try:
            root_dir = Path(__file__).resolve().parents[2]
            agent_cfg_path = root_dir / "configs" / "llmapi" / "agent.yaml"
            if agent_cfg_path.exists():
                with agent_cfg_path.open("r", encoding="utf-8") as f:
                    agents_cfg = yaml.safe_load(f) or {}
                    available_models = list(agents_cfg.keys())
            else:
                available_models = []
        except:
            available_models = []

        # 参考 main.py 的重试逻辑：429/500/超时/网络错误一直重试（线性递增，最大 60 秒）；400 等非可重试错误直接返回
        attempt = 0
        infinite_retry = (max_retries == -1)
        
        while infinite_retry or attempt < max_retries:
            try:
                # 单次请求超时设置为 250 秒，避免单个样本阻塞过久
                resp = requests.post(url, headers=headers, json=body, timeout=250)
                
                # 400 Bad Request 通常是请求格式问题，不应该重试
                # 但如果是 token 超限错误，需要特殊处理
                if resp.status_code == 400:
                    try:
                        error_detail = resp.json()
                        error_message = str(error_detail.get("message", "")) if isinstance(error_detail, dict) else str(error_detail)
                        # 检查是否是 token 超限错误
                        if "max_total_tokens" in error_message or "max_seq_len" in error_message or "exceeds" in error_message.lower():
                            print(f"[AWMPro] ERROR: LLM 400 Bad Request - Token limit exceeded (model={model_name}, purpose={purpose})")
                            print(f"[AWMPro] Error detail: {error_detail}")
                            # 返回特殊的错误标记字符串，让调用者知道是 token 超限
                            return "__TOKEN_LIMIT_EXCEEDED__"
                    except:
                        pass
                    # 其他 400 错误
                    try:
                        error_detail = resp.json()
                    except:
                        error_detail = resp.text[:500]
                    print(f"[AWMPro] ERROR: LLM 400 Bad Request (model={model_name}, purpose={purpose})")
                    print(f"[AWMPro] Error detail: {error_detail}")
                    print(f"[AWMPro] Request body model field: {body.get('model', 'NOT SET')}")
                    if available_models:
                        print(f"[AWMPro] Available models in agent.yaml: {available_models}")
                    return None
                
                # Too Many Requests / 500: 重试（线性递增，最大 60 秒）
                if resp.status_code in (429, 500):
                    # 线性递增：5 * (attempt + 1) 秒（5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 60, ...），最大 60 秒
                    wait_sec = min(5 * (attempt + 1), 60)
                    retry_info = "infinite retries" if infinite_retry else f"{attempt + 1}/{max_retries}"
                    print(
                        f"[AWMPro] LLM HTTP {resp.status_code} (attempt {retry_info}), "
                        f"retrying after {wait_sec}s (linear backoff, max 60s)..."
                    )
                    time.sleep(wait_sec)
                    attempt += 1
                    continue
                
                # 对于其他 HTTP 错误（如 401, 403 等），直接抛出
                resp.raise_for_status()
                data = resp.json()
                choices = data.get("choices") or []
                if not choices:
                    print(f"[AWMPro] WARNING: {purpose} returned no choices in response")
                    return None
                message = choices[0].get("message") or {}
                content = (message.get("content") or "").strip()
                if content:
                    print(f"[AWMPro] {purpose} succeeded, response_length={len(content)}")
                else:
                    print(f"[AWMPro] WARNING: {purpose} returned empty content")
                return content or None
                
            except (ReadTimeout, Timeout) as e:
                # 超时错误：重试（线性递增，最大 60 秒）
                # 线性递增：5 * (attempt + 1) 秒（5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 60, ...），最大 60 秒
                wait_sec = min(5 * (attempt + 1), 60)
                retry_info = "infinite retries" if infinite_retry else f"{attempt + 1}/{max_retries}"
                print(
                    f"[AWMPro] LLM timeout (attempt {retry_info}), "
                    f"retrying after {wait_sec}s (linear backoff, max 60s)..."
                )
                time.sleep(wait_sec)
                attempt += 1
                continue
                
            except requests.exceptions.RequestException as e:
                # 其他网络错误（如连接错误）：重试（线性递增，最大 60 秒）
                # 线性递增：5 * (attempt + 1) 秒（5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 60, ...），最大 60 秒
                wait_sec = min(5 * (attempt + 1), 60)
                retry_info = "infinite retries" if infinite_retry else f"{attempt + 1}/{max_retries}"
                print(
                    f"[AWMPro] LLM network error (attempt {retry_info}): {str(e)}, "
                    f"retrying after {wait_sec}s (linear backoff, max 60s)..."
                )
                time.sleep(wait_sec)
                attempt += 1
                continue
                
            except Exception as e:
                # 其他不可预期的错误，不重试
                print(f"[AWMPro] LLM fatal error: {e}")
                return None

        # 所有重试都失败（仅在非无限重试模式下）
        print(f"[AWMPro] ERROR: {purpose} failed after {max_retries} attempts")
        return None

    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        使用通用的 JSON 解析器（3 层容错）。

        参数：
            response: LLM 返回的原始文本

        返回：
            解析后的 JSON 字典，失败返回 None

        容错机制：
            1. 输出清洗：移除 markdown、注释等
            2. 智能修复：使用 json_repair 抢救 90% 残次品
            3. Schema 校验：跳过（AWMPro 不需要严格的 schema）
        """
        result = parse_llm_json_response(
            response_text=response,
            schema=None,  # AWMPro 不需要严格的 schema 校验
            logger_prefix="AWMPro"
        )

        if result is None:
            print(f"[AWMPro] JSON parsing failed after all attempts")
            return None

        return result

    def _call_workflow_induction(self, trajectory_text: str) -> Optional[str]:
        """调用 Workflow Induction Model 提取工作流（有限重试）"""
        print(f"[AWMPro] Calling workflow induction, trajectory_text_length={len(trajectory_text)}")
        prompt = self.config.workflow_induction_prompt

        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    "Here is one completed trajectory. Please extract concise reusable workflow(s) from it.\n\n"
                    f"{trajectory_text}"
                ),
            },
        ]

        attempt = 0
        max_retries = self.config.workflow_induction_max_retries
        while attempt < max_retries:
            attempt += 1
            print(f"[AWMPro] Workflow induction attempt {attempt}/{max_retries}")

            response = self._call_llm(self.config.model_name, messages, max_retries=-1, purpose="workflow induction")
            if response and len(response.strip()) > 0:
                # 检查是否包含 workflow 格式（至少有一个 ## workflow_name）
                if "## " in response:
                    print(f"[AWMPro] Workflow induction succeeded (attempt {attempt})")
                    return response
                else:
                    print(f"[AWMPro] WARNING: Workflow induction response doesn't contain workflow format (attempt {attempt})")
                    if attempt < max_retries:
                        wait_sec = min(5 * attempt, 60)
                        print(f"[AWMPro] Retrying workflow induction after {wait_sec}s...")
                        time.sleep(wait_sec)
                    continue
            else:
                if attempt < max_retries:
                    wait_sec = min(5 * attempt, 60)
                    print(f"[AWMPro] Workflow induction returned empty response, retrying after {wait_sec}s...")
                    time.sleep(wait_sec)
                continue

        print(f"[AWMPro] ERROR: Workflow induction failed after {max_retries} attempts")
        return None

    def _parse_new_workflows(self, new_workflows_text: str) -> List[str]:
        """从新工作流文本中解析出各个工作流"""
        workflows = []
        # 按 ## workflow_name 分割
        parts = new_workflows_text.split("## ")
        for part in parts:
            part = part.strip()
            if part:
                workflows.append(part)
        return workflows
    
    def _find_similar_workflows(
        self, new_workflow_text: str, existing_workflows: List[Dict[str, Any]], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """使用向量搜索找到与新工作流相似的现有工作流（参考 mem0 的做法）
        
        参考 mem0 的 add 方法：
        1. 对每个新工作流，使用向量搜索在现有工作流中找 top-k 相似项
        2. 只把相似的工作流传给 LLM，而不是全部工作流
        """
        if not self._workflow_rag or not existing_workflows:
            return []
        
        try:
            # 使用 RAG 检索相似工作流（返回的是工作流文本）
            retrieved_texts = self._workflow_rag.retrieve(query=new_workflow_text, top_k=top_k)
            if not retrieved_texts:
                return []
            
            # 根据检索到的文本，找到对应的现有工作流对象
            # RAG 存储时 key=value=工作流文本，所以检索到的文本应该就是工作流文本
            similar_workflows = []
            
            # 构建工作流文本到工作流对象的映射（用于快速查找）
            text_to_workflow = {}
            for wf in existing_workflows:
                wf_text = wf.get("text", "").strip()
                if wf_text:
                    # 使用规范化后的文本作为 key（去除多余空白）
                    normalized_text = " ".join(wf_text.split())
                    text_to_workflow[normalized_text] = wf
            
            # 匹配检索到的文本与现有工作流
            for retrieved_text in retrieved_texts:
                retrieved_text = retrieved_text.strip()
                if not retrieved_text:
                    continue
                
                # 尝试精确匹配
                normalized_retrieved = " ".join(retrieved_text.split())
                if normalized_retrieved in text_to_workflow:
                    similar_workflows.append(text_to_workflow[normalized_retrieved])
                    continue
                
                # 如果精确匹配失败，尝试模糊匹配（检查文本相似度）
                best_match = None
                best_similarity = 0.0
                
                for wf_text_normalized, wf in text_to_workflow.items():
                    # 计算简单的文本相似度（基于共同词汇）
                    retrieved_words = set(normalized_retrieved.lower().split())
                    wf_words = set(wf_text_normalized.lower().split())
                    
                    if not retrieved_words or not wf_words:
                        continue
                    
                    # Jaccard 相似度
                    intersection = len(retrieved_words & wf_words)
                    union = len(retrieved_words | wf_words)
                    similarity = intersection / union if union > 0 else 0.0
                    
                    # 也检查是否包含（更宽松的匹配）
                    if normalized_retrieved in wf_text_normalized or wf_text_normalized in normalized_retrieved:
                        similarity = max(similarity, 0.5)  # 至少 50% 相似度
                    
                    if similarity > best_similarity and similarity > 0.3:  # 30% 相似度阈值
                        best_similarity = similarity
                        best_match = wf
                
                if best_match and best_match not in similar_workflows:
                    similar_workflows.append(best_match)
            
            # 去重（保持顺序）
            seen_ids = set()
            unique_workflows = []
            for wf in similar_workflows:
                wf_id = wf.get("id", "")
                if wf_id and wf_id not in seen_ids:
                    seen_ids.add(wf_id)
                    unique_workflows.append(wf)
            
            return unique_workflows[:top_k]  # 限制返回数量
        except Exception as e:
            logger.warning(f"[AWMPro] Failed to find similar workflows using RAG: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return []


    def _call_workflow_management(
        self, existing_workflows: List[Dict[str, Any]], new_workflows: str
    ) -> Optional[Dict[str, Any]]:
        """调用 Workflow Management Model 进行增删改查（无限重试直到成功）
        
        参考 mem0 的做法：
        1. 先解析新工作流
        2. 对每个新工作流，用向量搜索找到相似的现有工作流
        3. 只把相似的工作流传给 LLM，而不是全部工作流
        4. LLM 判断操作类型（ADD/UPDATE/DELETE/NONE）
        """
        # 1. 解析新工作流
        new_workflow_list = self._parse_new_workflows(new_workflows)
        if not new_workflow_list:
            logger.warning("[AWMPro] No new workflows parsed from induction result")
            return None
        
        print(f"[AWMPro] Parsed {len(new_workflow_list)} new workflow(s) from induction result")
        
        # 2. 对每个新工作流，找到相似的现有工作流（使用向量搜索）
        all_similar_workflows = {}  # {new_workflow_text: [similar_existing_workflows]}
        similarity_top_k = self.config.workflow_management_similarity_top_k
        for new_wf_text in new_workflow_list:
            similar_wfs = self._find_similar_workflows(new_wf_text, existing_workflows, top_k=similarity_top_k)
            all_similar_workflows[new_wf_text] = similar_wfs
            print(f"[AWMPro] Found {len(similar_wfs)} similar workflow(s) for new workflow (top_k={similarity_top_k}, first 100 chars: {new_wf_text[:100]}...)")
        
        # 3. 合并所有相似工作流（去重）
        all_similar_ids = set()
        similar_workflows_list = []
        for similar_wfs in all_similar_workflows.values():
            for wf in similar_wfs:
                wf_id = wf.get("id", "")
                if wf_id not in all_similar_ids:
                    all_similar_ids.add(wf_id)
                    similar_workflows_list.append(wf)
        
        # 4. 格式化相似工作流（只传相似的工作流，而不是全部）
        existing_text = ""
        if similar_workflows_list:
            workflow_list = []
            for wf in similar_workflows_list:
                wf_id = wf.get("id", "")
                wf_text = wf.get("text", "")
                workflow_list.append(f"ID: {wf_id}\n{wf_text}")
            existing_text = "\n\n".join(workflow_list)
            print(f"[AWMPro] Using {len(similar_workflows_list)} similar workflow(s) for LLM comparison (out of {len(existing_workflows)} total)")
        else:
            print(f"[AWMPro] No similar workflows found, all new workflows will be ADDed")
            existing_text = "Current workflow memory is empty."

        # 构建 prompt
        prompt = self.config.workflow_management_prompt.format(
            existing_workflows=existing_text,
            new_workflows=new_workflows
        )

        messages = [
            {"role": "user", "content": prompt}
        ]

        attempt = 0
        max_workflows_to_keep = len(existing_workflows)  # 记录原始数量，用于截断
        max_retries = self.config.workflow_management_max_retries  # 从配置读取最大重试次数
        while attempt < max_retries:  # 限制最大重试次数
            attempt += 1
            print(f"[AWMPro] Workflow management attempt {attempt}/{max_retries}")
            
            response = self._call_llm(self.config.model_name, messages, max_retries=-1, purpose="workflow management")
            
            # 检查是否是 token 超限错误
            if response == "__TOKEN_LIMIT_EXCEEDED__":
                print(f"[AWMPro] Token limit exceeded, trying to truncate input...")
                # 如果现有工作流太多，尝试只保留最近的一部分
                if max_workflows_to_keep > 10:
                    # 保留最近的一半工作流
                    max_workflows_to_keep = max(10, max_workflows_to_keep // 2)
                    print(f"[AWMPro] Truncating to {max_workflows_to_keep} most recent workflows")
                    
                    # 重新格式化（只保留最近的工作流）
                    truncated_workflows = existing_workflows[-max_workflows_to_keep:]
                    workflow_list = []
                    for wf in truncated_workflows:
                        wf_id = wf.get("id", "")
                        wf_text = wf.get("text", "")
                        workflow_list.append(f"ID: {wf_id}\n{wf_text}")
                    existing_text = "\n\n".join(workflow_list)
                    
                    # 重新构建 prompt
                    prompt = self.config.workflow_management_prompt.format(
                        existing_workflows=existing_text,
                        new_workflows=new_workflows
                    )
                    messages = [
                        {"role": "user", "content": prompt}
                    ]
                    wait_sec = min(5 * attempt, 60)
                    print(f"[AWMPro] Retrying workflow management with truncated input after {wait_sec}s...")
                    time.sleep(wait_sec)
                    continue
                else:
                    # 如果已经截断到最小，但仍然超限，跳过这次更新
                    print(f"[AWMPro] WARNING: Token limit exceeded even with minimal workflows, skipping workflow update")
                    return None
            
            if not response:
                wait_sec = min(5 * attempt, 60)
                print(f"[AWMPro] Workflow management returned no response, retrying after {wait_sec}s...")
                time.sleep(wait_sec)
                continue

            # 解析 JSON 响应
            result = self._parse_json_response(response)
            if result and "memory" in result:
                memory_ops = result.get("memory", [])
                print(f"[AWMPro] Workflow management succeeded (attempt {attempt})")
                print(f"[AWMPro] Workflow management operations: {len(memory_ops)} operation(s)")
                # 打印每个操作的详细信息
                for idx, op in enumerate(memory_ops, 1):
                    op_id = op.get("id", "N/A")
                    op_event = op.get("event", "UNKNOWN").upper()
                    op_text = op.get("text", "")
                    op_old_memory = op.get("old_memory", "")
                    text_preview = op_text[:100] + "..." if len(op_text) > 100 else op_text
                    print(f"  [{idx}] {op_event}: id={op_id}, text_preview=\"{text_preview}\"")
                    if op_old_memory and op_event == "UPDATE":
                        old_preview = op_old_memory[:100] + "..." if len(op_old_memory) > 100 else op_old_memory
                        print(f"       old_memory_preview=\"{old_preview}\"")
                return result
            else:
                if result:
                    print(f"[AWMPro] WARNING: Workflow management response missing 'memory' key: {result}")
                else:
                    print(f"[AWMPro] WARNING: Failed to parse workflow management response (attempt {attempt})")
                    # 打印完整响应以便调试（限制长度避免日志过长）
                    response_preview = response[:2000] if len(response) > 2000 else response
                    print(f"[AWMPro] Workflow management raw response (first {len(response_preview)} chars): {response_preview}")
                    if len(response) > 2000:
                        print(f"[AWMPro] ... (truncated, total length: {len(response)} chars)")
                    # 尝试手动测试 JSON 解析
                    try:
                        import json
                        test_result = json.loads(response.strip())
                        print(f"[AWMPro] DEBUG: Manual JSON parse succeeded! Result keys: {list(test_result.keys()) if isinstance(test_result, dict) else 'not a dict'}")
                    except Exception as e:
                        print(f"[AWMPro] DEBUG: Manual JSON parse also failed: {e}")
                        # 尝试移除可能的 BOM 或不可见字符
                        try:
                            cleaned_test = response.strip().encode('utf-8').decode('utf-8-sig')
                            test_result2 = json.loads(cleaned_test)
                            print(f"[AWMPro] DEBUG: JSON parse succeeded after BOM removal!")
                        except Exception as e2:
                            print(f"[AWMPro] DEBUG: JSON parse still failed after BOM removal: {e2}")
                wait_sec = min(5 * attempt, 60)
                print(f"[AWMPro] Retrying workflow management after {wait_sec}s...")
                time.sleep(wait_sec)
                continue

        # 如果达到最大重试次数仍未成功，跳过本次更新
        print(f"[AWMPro] WARNING: Workflow management failed after {max_retries} attempts, skipping workflow update")
        return None

    def _load_workflows(self) -> List[Dict[str, Any]]:
        """加载工作流记忆"""
        if not self._workflow_storage_path.exists():
            return []
        try:
            with open(self._workflow_storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception:
            pass
        return []

    def _save_workflows(self, workflows: List[Dict[str, Any]]) -> None:
        """保存工作流记忆"""
        with open(self._workflow_storage_path, "w", encoding="utf-8") as f:
            json.dump(workflows, f, ensure_ascii=False, indent=2)

    def _extract_question_from_messages(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """
        从 messages 中提取第一个 user message 作为检索 query。
        需要过滤掉插入的工作流记忆，只返回原始问题。
        """
        template_titles = [self.template_title]
        return extract_original_question(messages, where=self.where, template_titles=template_titles)

    def use_memory(
        self, task: str, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        增强 messages：
        1. 检索工作流记忆（RAG）
        2. 追加到第一个 user message 的末尾
        """
        # 检索工作流记忆（RAG）
        if self._workflow_rag:
            question = self._extract_question_from_messages(messages)
            if question:
                retrieved_texts = self._workflow_rag.retrieve(
                    query=question, top_k=self._workflow_rag.top_k
                )
                if retrieved_texts:
                    print(f"[AWMPro] Retrieved {len(retrieved_texts)} workflows from RAG")

                    # 组合检索到的 workflows
                    formatted_workflows = "\n\n".join(retrieved_texts)
                    workflow_memory_text = self.config.workflow_rag_prompt_template.format(
                        workflows=formatted_workflows
                    )

                    # 使用公共工具插入记忆
                    return enhance_messages_with_memory(messages, workflow_memory_text, where=self.where)

        return list(messages) if messages is not None else []

    def _build_trajectory_text(
        self, task: str, history: List[Dict[str, Any]], result: Dict[str, Any]
    ) -> Optional[str]:
        """
        构建轨迹文本。
        关键：必须过滤掉本轮use_memory()插入的工作流记忆，只保留原始的交互。
        """
        # 检查是否成功
        if self.config.workflow_rag_success_only:
            status = result.get("status", "")
            reward = result.get("reward", 0)
            is_success = status == "completed" or reward > 0
            if not is_success:
                return None

        if self.config.workflow_rag_reward_bigger_than_zero:
            reward = result.get("reward", 0)
            if reward <= 0:
                return None

        if not history:
            return None

        template_titles = [self.template_title]
        parts: List[str] = [f"Task: {task}"]

        for msg in history:
            role, content, msg_dict = extract_message_info(msg)
            if role is None:
                continue
            content = str(content).strip() if content else ""
            if not content:
                continue

            if role == "user":
                # 如果是第一个user消息且包含记忆内容，提取原始问题
                from src.utils.message_schema import ORIGINAL_QUESTION_SEPARATOR
                has_memory = (
                    ORIGINAL_QUESTION_SEPARATOR in content or
                    any(title in content for title in template_titles)
                )

                if has_memory:
                    # 使用公共工具提取原始问题
                    question = extract_original_question([msg], where=self.where, template_titles=template_titles)
                    if question:
                        content = question
                    else:
                        content = ""

                if content:  # 确保过滤后还有内容
                    parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            elif role == "tool":
                tool_name = "tool"
                if msg_dict:
                    tool_name = msg_dict.get("name") or msg_dict.get("tool_call_id") or "tool"
                parts.append(f"Tool[{tool_name}]: {content}")

        return "\n".join(parts)

    def update_memory(
        self, task: str, history: List[Dict[str, Any]], result: Dict[str, Any]
    ) -> None:
        """
        更新工作流记忆：
        1. 检查过滤条件（success_only, reward_bigger_than_zero）
        2. 调用 Workflow Induction 提取工作流
        3. 调用 Workflow Management 进行增删改查
        """
        # 检查过滤条件
        status = result.get("status", "")
        finish = result.get("finish", False)
        reward = result.get("reward", 0)
        is_success = finish or (status == "completed")
        
        print(f"[AWMPro] update_memory called: task={task}, finish={finish}, status={status}, reward={reward}, "
              f"is_success={is_success}, success_only={self.config.workflow_rag_success_only}, "
              f"reward_bigger_than_zero={self.config.workflow_rag_reward_bigger_than_zero}")
        
        if self.config.workflow_rag_success_only:
            if not is_success:
                print(f"[AWMPro] Skipping workflow update: success_only=True but sample not completed "
                      f"(finish={finish}, status={status})")
                return
        
        if self.config.workflow_rag_reward_bigger_than_zero:
            if reward <= 0:
                print(f"[AWMPro] Skipping workflow update: reward_bigger_than_zero=True but reward={reward}")
                return
        
        # 直接更新工作流记忆
        print("[AWMPro] Updating workflow memory...")
        self._update_system_memory(task, history, result)

    def _update_system_memory(
        self, task: str, history: List[Dict[str, Any]], result: Dict[str, Any]
    ) -> None:
        """更新工作流记忆"""
        # 构建轨迹文本
        trajectory_text = self._build_trajectory_text(task, history, result)
        if not trajectory_text:
            return

        # 调用 Workflow Induction
        new_workflows_text = self._call_workflow_induction(trajectory_text)
        if not new_workflows_text:
            logger.warning("[AWMPro] Workflow induction failed")
            return

        # 加载现有工作流
        existing_workflows = self._load_workflows()

        # 调用 Workflow Management
        management_result = self._call_workflow_management(existing_workflows, new_workflows_text)
        if not management_result:
            logger.warning("[AWMPro] Workflow management failed")
            return

        # 应用增删改查操作
        memory_ops = management_result.get("memory", [])
        if not memory_ops:
            print("[AWMPro] No memory operations to apply")
            return

        print(f"[AWMPro] Applying {len(memory_ops)} workflow management operation(s)...")

        # 构建工作流字典（id -> workflow）
        workflow_dict = {wf.get("id", ""): wf for wf in existing_workflows}
        
        # 计算最大 ID（用于生成新 ID）
        max_id = -1
        for wf_id in workflow_dict.keys():
            try:
                id_num = int(wf_id)
                max_id = max(max_id, id_num)
            except ValueError:
                pass

        # 处理每个操作
        rag_needs_rebuild = False
        add_count = 0
        update_count = 0
        delete_count = 0
        none_count = 0
        
        for op in memory_ops:
            op_id = op.get("id", "")
            op_event = op.get("event", "").upper()
            op_text = op.get("text", "")

            if op_event == "ADD":
                # 添加新工作流：生成新 ID（最大 ID + 1）
                new_id = str(max_id + 1)
                max_id += 1
                workflow_dict[new_id] = {"id": new_id, "text": op_text}
                # 添加到 RAG（使用 workflow 文本作为 key 和 value）
                if self._workflow_rag:
                    self._workflow_rag.insert(key=op_text, value=op_text)
                add_count += 1
                text_preview = op_text[:80] + "..." if len(op_text) > 80 else op_text
                print(f"  [ADD] Created new workflow id={new_id}, text_preview=\"{text_preview}\"")
            elif op_event == "UPDATE":
                # 更新现有工作流
                if op_id in workflow_dict:
                    old_text = workflow_dict[op_id].get("text", "")
                    workflow_dict[op_id]["text"] = op_text
                    # RAG 不支持直接更新，标记需要重建索引
                    rag_needs_rebuild = True
                    update_count += 1
                    old_preview = old_text[:80] + "..." if len(old_text) > 80 else old_text
                    new_preview = op_text[:80] + "..." if len(op_text) > 80 else op_text
                    print(f"  [UPDATE] Updated workflow id={op_id}")
                    print(f"    old: \"{old_preview}\"")
                    print(f"    new: \"{new_preview}\"")
                else:
                    print(f"  [UPDATE] WARNING: Workflow id={op_id} not found, skipping update")
            elif op_event == "DELETE":
                # 删除工作流
                if op_id in workflow_dict:
                    deleted_text = workflow_dict[op_id].get("text", "")
                    del workflow_dict[op_id]
                    # RAG 不支持直接删除，标记需要重建索引
                    rag_needs_rebuild = True
                    delete_count += 1
                    text_preview = deleted_text[:80] + "..." if len(deleted_text) > 80 else deleted_text
                    print(f"  [DELETE] Deleted workflow id={op_id}, text_preview=\"{text_preview}\"")
                else:
                    print(f"  [DELETE] WARNING: Workflow id={op_id} not found, skipping delete")
            elif op_event == "NONE":
                # 不做任何操作
                none_count += 1
                text_preview = op_text[:80] + "..." if len(op_text) > 80 else op_text
                print(f"  [NONE] No change for workflow id={op_id}, text_preview=\"{text_preview}\"")
            else:
                print(f"  [UNKNOWN] Unknown event type: {op_event}, id={op_id}")
        
        # 打印操作统计
        print(f"[AWMPro] Workflow management operations summary: ADD={add_count}, UPDATE={update_count}, DELETE={delete_count}, NONE={none_count}")

        # 保存更新后的工作流
        updated_workflows = list(workflow_dict.values())
        self._save_workflows(updated_workflows)
        
        # 如果 RAG 需要重建（UPDATE 或 DELETE 操作），从 workflows.json 重新加载并重建索引
        if rag_needs_rebuild and self._workflow_rag:
            self._rebuild_rag_index()

    def _rebuild_rag_index(self) -> None:
        """重建 RAG 索引：从 workflows.json 重新加载所有工作流并重建索引
        注意：不重新创建RAG实例，只清空并重新插入数据"""
        if not self._workflow_rag:
            return

        # 清空现有索引（通过重新创建RAG实例）
        # 注意：这里仍然需要重新创建，因为RAG可能没有clear方法
        # 但这是为了清空数据，不是为了重置配置
        try:
            self._workflow_rag = RAG(
                embedding_model=self.config.workflow_rag_embedding_model,
                top_k=self.config.workflow_rag_top_k,
                order=self.config.workflow_rag_order,
                seed=self.config.workflow_rag_seed,
            )
        except Exception as e:
            logger.warning(f"[AWMPro] Failed to rebuild RAG index: {e}")
            return

        # 从 workflows.json 重新加载并插入所有工作流
        workflows = self._load_workflows()
        print(f"[AWMPro] Rebuilding RAG index with {len(workflows)} workflow(s) from workflows.json...")
        for wf in workflows:
            wf_text = wf.get("text", "")
            if wf_text:
                self._workflow_rag.insert(key=wf_text, value=wf_text)

        print(f"[AWMPro] RAG index rebuilt successfully with {len(workflows)} workflow(s)")


def load_awmpro_from_yaml(config_path: str) -> AWMPro:
    """从 YAML 配置文件加载 AWMPro"""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    raw = cfg.get("awmpro", {}) or {}

    # 模型配置（统一使用一个模型）
    model_name = str(raw.get("model_name", ""))
    if not model_name:
        raise ValueError("AWMPro config must specify 'model_name'")

    # Prompt 配置
    workflow_induction_prompt = raw.get("workflow_induction_prompt", "") or ""
    workflow_management_prompt = raw.get("workflow_management_prompt", "") or ""

    # 工作流 RAG 配置
    workflow_rag_raw = raw.get("workflow_rag", {}) or {}
    workflow_rag_embedding_model = workflow_rag_raw.get("embedding_model", "")
    workflow_rag_top_k = int(workflow_rag_raw.get("top_k", 100))
    workflow_rag_order = str(workflow_rag_raw.get("order", "similar_at_top"))
    workflow_rag_seed = int(workflow_rag_raw.get("seed", 42))
    workflow_rag_prompt_template = workflow_rag_raw.get("prompt_template", "") or ""
    workflow_rag_where = workflow_rag_raw.get("where", "tail")
    workflow_rag_success_only = bool(workflow_rag_raw.get("success_only", True))
    workflow_rag_reward_bigger_than_zero = bool(workflow_rag_raw.get("reward_bigger_than_zero", True))

    # 工作流管理配置
    workflow_management_similarity_top_k = int(raw.get("workflow_management_similarity_top_k", 5))  # 默认 5，参考 mem0

    # 最大重试次数配置
    workflow_induction_max_retries = int(raw.get("workflow_induction_max_retries", 5))  # 默认 5
    workflow_management_max_retries = int(raw.get("workflow_management_max_retries", 5))  # 默认 5

    # 工作流存储路径
    workflow_storage_path = Path(raw.get("workflow_storage_path", "memory/awmPro/workflows.json"))

    config = AWMProConfig(
        model_name=model_name,
        workflow_induction_prompt=workflow_induction_prompt,
        workflow_management_prompt=workflow_management_prompt,
        workflow_rag_embedding_model=workflow_rag_embedding_model,
        workflow_rag_top_k=workflow_rag_top_k,
        workflow_rag_order=workflow_rag_order,
        workflow_rag_seed=workflow_rag_seed,
        workflow_rag_prompt_template=workflow_rag_prompt_template,
        workflow_rag_where=workflow_rag_where,
        workflow_rag_success_only=workflow_rag_success_only,
        workflow_rag_reward_bigger_than_zero=workflow_rag_reward_bigger_than_zero,
        workflow_management_similarity_top_k=workflow_management_similarity_top_k,
        workflow_storage_path=workflow_storage_path,
        workflow_induction_max_retries=workflow_induction_max_retries,
        workflow_management_max_retries=workflow_management_max_retries,
    )
    return AWMPro(config)

