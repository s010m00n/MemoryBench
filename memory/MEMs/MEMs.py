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
from ..mem0.mem0 import Mem0Memory, Mem0Config
from src.utils.message_schema import extract_message_info

logger = logging.getLogger(__name__)


def _serialize_history(history: List[Any]) -> List[Dict[str, Any]]:
    """
    将 history 转换为可序列化的格式（JSON 兼容）。
    过滤掉 RewardHistoryItem 等非聊天消息，并将 Pydantic 模型转换为字典。
    """
    serialized = []
    for msg in history:
        role, content, msg_dict = extract_message_info(msg)
        
        # 跳过无法提取 role 的消息（如 RewardHistoryItem）
        if role is None:
            continue
        
        # 如果提取到了完整的消息字典，使用它
        if msg_dict is not None:
            # 确保是字典类型
            if isinstance(msg_dict, dict):
                serialized.append(msg_dict)
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
class MEMsConfig:
    # 模型配置（统一使用一个模型）
    model_name: str

    # Prompt 配置
    workflow_induction_prompt: str
    trigger_prompt: str
    workflow_management_prompt: str

    # Trigger Model 过滤配置
    trigger_model_success_only: bool
    trigger_model_reward_bigger_than_zero: bool

    # 工作流 RAG 配置
    workflow_rag_embedding_model: str
    workflow_rag_top_k: int
    workflow_rag_order: str
    workflow_rag_seed: int
    workflow_rag_prompt_template: str  # 改为 prompt_template
    workflow_rag_where: str  # "tail": 记忆放在 user question 后面 | "front": 记忆放在 user question 前面
    workflow_rag_success_only: bool
    workflow_rag_reward_bigger_than_zero: bool

    # 工作流管理配置
    workflow_management_similarity_top_k: int  # 向量搜索时每个新工作流找 top_k 个相似工作流

    # 工作流存储路径
    workflow_storage_path: Path

    # Mem0 配置路径
    mem0_config_path: Path

    # 最大重试次数配置（有默认值的字段必须放在最后）
    trigger_model_max_retries: int = 5
    workflow_induction_max_retries: int = 5
    workflow_management_max_retries: int = 5


class MEMs(MemoryMechanism):
    """
    Multi-Enhanced Memory System (MEMs):
    - Coordinates multiple memory sources (system memory and personal memory)
    - Uses Trigger Model to decide which memory source to update
    - System memory: stores task execution workflows (RAG-based)
    - Personal memory: stores user preferences (Mem0-based)
    """

    def __init__(self, config: MEMsConfig) -> None:
        self.config = config
        self._workflow_storage_path = self.config.workflow_storage_path
        self._workflow_storage_path.parent.mkdir(parents=True, exist_ok=True)

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
            logger.warning(f"[MEMs] Failed to init workflow RAG: {e}. Vector retrieval disabled.")
            self._workflow_rag = None

        # 初始化 Mem0 记忆源
        self._mem0_memory: Optional[Mem0Memory] = None
        try:
            mem0_config = self._load_mem0_config()
            if mem0_config:
                self._mem0_memory = Mem0Memory(mem0_config)
        except Exception as e:
            logger.warning(f"[MEMs] Failed to init Mem0 memory: {e}")
            self._mem0_memory = None

    def _load_mem0_config(self) -> Optional[Mem0Config]:
        """加载 Mem0 配置"""
        if not self.config.mem0_config_path.exists():
            logger.warning(f"[MEMs] Mem0 config not found: {self.config.mem0_config_path}")
            return None
        
        try:
            with self.config.mem0_config_path.open("r", encoding="utf-8") as f:
                mem0_yaml = yaml.safe_load(f) or {}
            
            mem0_cfg = mem0_yaml.get("mem0", {})
            if not mem0_cfg:
                logger.warning(f"[MEMs] Mem0 config section not found in {self.config.mem0_config_path}")
                return None
            
            return Mem0Config(
                api_key=mem0_cfg.get("api_key", ""),
                user_id=mem0_cfg.get("user_id", "default_user"),
                infer=mem0_cfg.get("infer", True),
                top_k=mem0_cfg.get("top_k", 100),
                threshold=mem0_cfg.get("threshold"),
                rerank=mem0_cfg.get("rerank", True),
                success_only=mem0_cfg.get("success_only", True),
                reward_bigger_than_zero=mem0_cfg.get("reward_bigger_than_zero", True),
                prompt_template=mem0_cfg.get("prompt_template", ""),
                max_retries=mem0_cfg.get("max_retries", -1),
                retry_delay=mem0_cfg.get("retry_delay", 5.0),
                retry_backoff=mem0_cfg.get("retry_backoff", 2.0),
                wait_time=mem0_cfg.get("wait_time", 0.0),
            )
        except Exception as e:
            logger.warning(f"[MEMs] Failed to load Mem0 config: {e}")
            return None

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
                f"[MEMs] LLM config files not found: {agent_cfg_path}, {api_cfg_path}"
            )
            return None

        try:
            with agent_cfg_path.open("r", encoding="utf-8") as f:
                agents_cfg = yaml.safe_load(f) or {}
            if model_name not in agents_cfg:
                logger.warning(
                    f"[MEMs] Model '{model_name}' not found in {agent_cfg_path}"
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
                logger.warning("[MEMs] URL not found in api.yaml / agent.yaml")
                return None

            headers = dict(base_params.get("headers", {}) or {})
            headers.update(agent_params.get("headers", {}) or {})

            return {"url": url, "headers": headers, "body": body}
        except Exception as e:
            logger.warning(f"[MEMs] failed to load agent config: {e}")
            return None

    def _call_llm(self, model_name: str, messages: List[Dict[str, Any]], max_retries: int = 3, purpose: str = "LLM call") -> Optional[str]:
        """调用 LLM API，支持无限重试（max_retries=-1）"""
        print(f"[MEMs] Calling {purpose} with model={model_name}, messages_count={len(messages)}")
        cfg = self._load_agent_config(model_name)
        if not cfg:
            print(f"[MEMs] ERROR: Failed to load agent config for model={model_name}")
            return None

        url = cfg["url"]
        headers = cfg["headers"]
        base_body = cfg["body"]
        
        # 记录请求信息（不包含完整 messages，避免日志过长）
        print(f"[MEMs] {purpose} request: url={url}, model={model_name}, body_keys={list(base_body.keys())}")

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
                            print(f"[MEMs] ERROR: LLM 400 Bad Request - Token limit exceeded (model={model_name}, purpose={purpose})")
                            print(f"[MEMs] Error detail: {error_detail}")
                            # 返回特殊的错误标记字符串，让调用者知道是 token 超限
                            return "__TOKEN_LIMIT_EXCEEDED__"
                    except:
                        pass
                    # 其他 400 错误
                    try:
                        error_detail = resp.json()
                    except:
                        error_detail = resp.text[:500]
                    print(f"[MEMs] ERROR: LLM 400 Bad Request (model={model_name}, purpose={purpose})")
                    print(f"[MEMs] Error detail: {error_detail}")
                    print(f"[MEMs] Request body model field: {body.get('model', 'NOT SET')}")
                    if available_models:
                        print(f"[MEMs] Available models in agent.yaml: {available_models}")
                    return None
                
                # Too Many Requests / 500: 重试（线性递增，最大 60 秒）
                if resp.status_code in (429, 500):
                    # 线性递增：5 * (attempt + 1) 秒（5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 60, ...），最大 60 秒
                    wait_sec = min(5 * (attempt + 1), 60)
                    retry_info = "infinite retries" if infinite_retry else f"{attempt + 1}/{max_retries}"
                    print(
                        f"[MEMs] LLM HTTP {resp.status_code} (attempt {retry_info}), "
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
                    print(f"[MEMs] WARNING: {purpose} returned no choices in response")
                    return None
                message = choices[0].get("message") or {}
                content = (message.get("content") or "").strip()
                if content:
                    print(f"[MEMs] {purpose} succeeded, response_length={len(content)}")
                else:
                    print(f"[MEMs] WARNING: {purpose} returned empty content")
                return content or None
                
            except (ReadTimeout, Timeout) as e:
                # 超时错误：重试（线性递增，最大 60 秒）
                # 线性递增：5 * (attempt + 1) 秒（5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 60, ...），最大 60 秒
                wait_sec = min(5 * (attempt + 1), 60)
                retry_info = "infinite retries" if infinite_retry else f"{attempt + 1}/{max_retries}"
                print(
                    f"[MEMs] LLM timeout (attempt {retry_info}), "
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
                    f"[MEMs] LLM network error (attempt {retry_info}): {str(e)}, "
                    f"retrying after {wait_sec}s (linear backoff, max 60s)..."
                )
                time.sleep(wait_sec)
                attempt += 1
                continue
                
            except Exception as e:
                # 其他不可预期的错误，不重试
                print(f"[MEMs] LLM fatal error: {e}")
                return None

        # 所有重试都失败（仅在非无限重试模式下）
        print(f"[MEMs] ERROR: {purpose} failed after {max_retries} attempts")
        return None

    def _call_trigger_model(self, history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """调用 Trigger Model 决定更新哪个记忆源（无限重试直到成功）"""
        print(f"[MEMs] Calling trigger model, history_length={len(history)}")
        # 序列化 history
        serialized_history = _serialize_history(history)
        history_text = json.dumps(serialized_history, ensure_ascii=False, indent=2)
        print(f"[MEMs] Trigger model input: serialized_history_length={len(serialized_history)}, history_text_length={len(history_text)}")

        # 构建 prompt
        prompt = self.config.trigger_prompt.format(history=history_text)

        messages = [
            {"role": "user", "content": prompt}
        ]

        attempt = 0
        max_retries = self.config.trigger_model_max_retries
        while attempt < max_retries:
            attempt += 1
            print(f"[MEMs] Trigger model attempt {attempt}/{max_retries}")
            
            response = self._call_llm(self.config.model_name, messages, max_retries=-1, purpose="trigger model")
            if not response:
                wait_sec = min(5 * attempt, 60)
                print(f"[MEMs] Trigger model returned no response, retrying after {wait_sec}s...")
                time.sleep(wait_sec)
                continue

            # 解析 JSON 响应
            try:
                import re
                # 先尝试移除 markdown 代码块标记（```json ... ```）
                cleaned_response = response.strip()
                if cleaned_response.startswith("```"):
                    # 移除开头的 ```json 或 ```
                    lines = cleaned_response.split("\n")
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    # 移除结尾的 ```
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    cleaned_response = "\n".join(lines).strip()
                
                # 尝试直接解析
                result = json.loads(cleaned_response)
                if "sources" in result:
                    sources = result.get("sources", [])
                    reasoning = result.get("reasoning", "")
                    print(f"[MEMs] Trigger model decision: sources={sources}, reasoning={reasoning[:100]}...")
                    return result
                else:
                    print(f"[MEMs] WARNING: Trigger model response missing 'sources' key: {result}")
                    wait_sec = min(5 * attempt, 60)
                    print(f"[MEMs] Retrying trigger model after {wait_sec}s...")
                    time.sleep(wait_sec)
                    continue
            except Exception as e:
                print(f"[MEMs] WARNING: Failed to parse trigger model response (attempt {attempt}): {e}")
                print(f"[MEMs] Trigger model raw response (first 500 chars): {response[:500]}")
                wait_sec = min(5 * attempt, 60)
                print(f"[MEMs] Retrying trigger model after {wait_sec}s...")
                time.sleep(wait_sec)
                continue

        # 如果达到最大重试次数仍未成功，返回默认值（只更新system_memory）
        print(f"[MEMs] WARNING: Trigger model failed after {max_retries} attempts, defaulting to system_memory only")
        return {"sources": ["system_memory"], "reasoning": "Trigger model failed, using default"}

    def _call_workflow_induction(self, trajectory_text: str) -> Optional[str]:
        """调用 Workflow Induction Model 提取工作流（无限重试直到成功）"""
        print(f"[MEMs] Calling workflow induction, trajectory_text_length={len(trajectory_text)}")
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
            print(f"[MEMs] Workflow induction attempt {attempt}/{max_retries}")
            
            response = self._call_llm(self.config.model_name, messages, max_retries=-1, purpose="workflow induction")
            if response and len(response.strip()) > 0:
                # 检查是否包含 workflow 格式（至少有一个 ## workflow_name）
                if "## " in response:
                    print(f"[MEMs] Workflow induction succeeded (attempt {attempt})")
                    return response
                else:
                    print(f"[MEMs] WARNING: Workflow induction response doesn't contain workflow format (attempt {attempt})")
                    wait_sec = min(5 * attempt, 60)
                    print(f"[MEMs] Retrying workflow induction after {wait_sec}s...")
                    time.sleep(wait_sec)
                    continue
            else:
                wait_sec = min(5 * attempt, 60)
                print(f"[MEMs] Workflow induction returned empty response, retrying after {wait_sec}s...")
                time.sleep(wait_sec)
                continue

        # 如果达到最大重试次数仍未成功，返回None
        print(f"[MEMs] WARNING: Workflow induction failed after {max_retries} attempts, skipping workflow extraction")
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
            logger.warning(f"[MEMs] Failed to find similar workflows using RAG: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return []

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
            3. Schema 校验：跳过（MEMs 不需要严格的 schema）
        """
        result = parse_llm_json_response(
            response_text=response,
            schema=None,  # MEMs 不需要严格的 schema 校验
            logger_prefix="MEMs"
        )

        if result is None:
            logger.warning(f"[MEMs] JSON parsing failed after all attempts")
            return None

        return result

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
            logger.warning("[MEMs] No new workflows parsed from induction result")
            return None
        
        print(f"[MEMs] Parsed {len(new_workflow_list)} new workflow(s) from induction result")
        
        # 2. 对每个新工作流，找到相似的现有工作流（使用向量搜索）
        all_similar_workflows = {}  # {new_workflow_text: [similar_existing_workflows]}
        similarity_top_k = self.config.workflow_management_similarity_top_k
        for new_wf_text in new_workflow_list:
            similar_wfs = self._find_similar_workflows(new_wf_text, existing_workflows, top_k=similarity_top_k)
            all_similar_workflows[new_wf_text] = similar_wfs
            print(f"[MEMs] Found {len(similar_wfs)} similar workflow(s) for new workflow (top_k={similarity_top_k}, first 100 chars: {new_wf_text[:100]}...)")
        
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
            print(f"[MEMs] Using {len(similar_workflows_list)} similar workflow(s) for LLM comparison (out of {len(existing_workflows)} total)")
        else:
            print(f"[MEMs] No similar workflows found, all new workflows will be ADDed")
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
        max_retries = self.config.workflow_management_max_retries
        while attempt < max_retries:
            attempt += 1
            print(f"[MEMs] Workflow management attempt {attempt}/{max_retries}")
            
            response = self._call_llm(self.config.model_name, messages, max_retries=-1, purpose="workflow management")
            
            # 检查是否是 token 超限错误
            if response == "__TOKEN_LIMIT_EXCEEDED__":
                print(f"[MEMs] Token limit exceeded, trying to truncate input...")
                # 如果现有工作流太多，尝试只保留最近的一部分
                if max_workflows_to_keep > 10:
                    # 保留最近的一半工作流
                    max_workflows_to_keep = max(10, max_workflows_to_keep // 2)
                    print(f"[MEMs] Truncating to {max_workflows_to_keep} most recent workflows")
                    
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
                    print(f"[MEMs] Retrying workflow management with truncated input after {wait_sec}s...")
                    time.sleep(wait_sec)
                    continue
                else:
                    # 如果已经截断到最小，但仍然超限，跳过这次更新
                    print(f"[MEMs] WARNING: Token limit exceeded even with minimal workflows, skipping workflow update")
                    return None
            
            if not response:
                wait_sec = min(5 * attempt, 60)
                print(f"[MEMs] Workflow management returned no response, retrying after {wait_sec}s...")
                time.sleep(wait_sec)
                continue

            # 解析 JSON 响应
            result = self._parse_json_response(response)
            if result and "memory" in result:
                memory_ops = result.get("memory", [])
                print(f"[MEMs] Workflow management succeeded (attempt {attempt})")
                print(f"[MEMs] Workflow management operations: {len(memory_ops)} operation(s)")
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
                    print(f"[MEMs] WARNING: Workflow management response missing 'memory' key: {result}")
                else:
                    print(f"[MEMs] WARNING: Failed to parse workflow management response (attempt {attempt})")
                    # 打印完整响应以便调试（限制长度避免日志过长）
                    response_preview = response[:2000] if len(response) > 2000 else response
                    print(f"[MEMs] Workflow management raw response (first {len(response_preview)} chars): {response_preview}")
                    if len(response) > 2000:
                        print(f"[MEMs] ... (truncated, total length: {len(response)} chars)")
                    # 尝试手动测试 JSON 解析
                    try:
                        import json
                        test_result = json.loads(response.strip())
                        print(f"[MEMs] DEBUG: Manual JSON parse succeeded! Result keys: {list(test_result.keys()) if isinstance(test_result, dict) else 'not a dict'}")
                    except Exception as e:
                        print(f"[MEMs] DEBUG: Manual JSON parse also failed: {e}")
                        # 尝试移除可能的 BOM 或不可见字符
                        try:
                            cleaned_test = response.strip().encode('utf-8').decode('utf-8-sig')
                            test_result2 = json.loads(cleaned_test)
                            print(f"[MEMs] DEBUG: JSON parse succeeded after BOM removal!")
                        except Exception as e2:
                            print(f"[MEMs] DEBUG: JSON parse still failed after BOM removal: {e2}")
                wait_sec = min(5 * attempt, 60)
                print(f"[MEMs] Retrying workflow management after {wait_sec}s...")
                time.sleep(wait_sec)
                continue

        # 如果达到最大重试次数仍未成功，跳过本次更新
        print(f"[MEMs] WARNING: Workflow management failed after {max_retries} attempts, skipping workflow update")
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
        需要过滤掉插入的记忆内容，只返回原始问题。
        """
        if not messages:
            return None
        # MEMs使用两个模板标题
        workflow_prefix = "Here are some useful workflows learned from past similar episodes"
        mem0_prefix = "Based on your previous interactions, here are relevant memories:"

        for msg in messages:
            role, content, _ = extract_message_info(msg)
            if role == "user":
                content = str(content).strip() if content else ""
                # 如果包含任一模板标题，根据 where 参数判断原始问题的位置
                if workflow_prefix in content:
                    if self.where == "front":
                        # where=front: 记忆在前，原始问题在后（模板前缀之后）
                        parts = content.split(workflow_prefix, 1)
                        if len(parts) > 1:
                            question = parts[1].strip()
                            return question if question else None
                        return None
                    else:  # tail
                        # where=tail: 原始问题在前，记忆在后（模板前缀之前）
                        question = content.split(workflow_prefix)[0].strip()
                        return question if question else None
                if mem0_prefix in content:
                    if self.where == "front":
                        # where=front: 记忆在前，原始问题在后（模板前缀之后）
                        parts = content.split(mem0_prefix, 1)
                        if len(parts) > 1:
                            question = parts[1].strip()
                            return question if question else None
                        return None
                    else:  # tail
                        # where=tail: 原始问题在前，记忆在后（模板前缀之前）
                        question = content.split(mem0_prefix)[0].strip()
                        return question if question else None
                return content
        return None

    def use_memory(
        self, task: str, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        增强 messages：
        1. 检索 system memory（工作流）
        2. 检索 personal memory（Mem0）
        3. 将两者合并，追加到第一个 user message 的末尾
        """
        enhanced = list(messages) if messages is not None else []

        # 1. 检索 system memory（工作流）
        system_memory_text = ""
        if self._workflow_rag:
            question = self._extract_question_from_messages(messages)
            if question:
                retrieved_texts = self._workflow_rag.retrieve(
                    query=question, top_k=self._workflow_rag.top_k
                )
                if retrieved_texts:
                    formatted_workflows = "\n\n".join(retrieved_texts)
                    system_memory_text = self.config.workflow_rag_prompt_template.format(
                        workflows=formatted_workflows
                    )

        # 2. 检索 personal memory（Mem0）
        personal_memory_text = ""
        if self._mem0_memory:
            try:
                # 使用 Mem0 的检索逻辑
                query = self._extract_question_from_messages(messages)
                if query:
                    mem0_config = self._mem0_memory.config
                    search_kwargs = {
                        "query": query,
                        "user_id": mem0_config.user_id,
                        "top_k": mem0_config.top_k,
                        "filters": {"user_id": mem0_config.user_id},
                    }
                    if mem0_config.threshold is not None:
                        search_kwargs["threshold"] = mem0_config.threshold
                    if mem0_config.rerank:
                        search_kwargs["rerank"] = True

                    memories = self._mem0_memory._client.search(**search_kwargs)
                    formatted_memories = self._mem0_memory._format_memories(memories)
                    if formatted_memories:
                        personal_memory_text = mem0_config.prompt_template.format(
                            memories=formatted_memories
                        )
            except Exception as e:
                logger.warning(f"[MEMs] Personal memory retrieval failed: {e}")

        # 3. 合并记忆并追加到第一个 user message 的末尾
        combined_memory = ""
        if system_memory_text and personal_memory_text:
            combined_memory = system_memory_text + "\n\n" + personal_memory_text
        elif system_memory_text:
            combined_memory = system_memory_text
        elif personal_memory_text:
            combined_memory = personal_memory_text

        if combined_memory:
            for i, msg in enumerate(enhanced):
                role, content, msg_dict = extract_message_info(msg)
                if role == "user":
                    content = content if content else ""
                    # 根据 where 参数决定记忆放在前面还是后面
                    # 注意：MEMs 使用 workflow_rag_where，因为它主要基于 workflow 配置
                    # 如果需要考虑 mem0 的 where，需要在组合记忆时分别处理
                    if self.config.workflow_rag_where == "front":
                        new_content = combined_memory + "\n\n" + content
                    else:  # tail
                        new_content = content + "\n\n" + combined_memory
                    enhanced[i] = {
                        **msg_dict,
                        "content": new_content
                    }
                    break

        return enhanced

    def _build_trajectory_text(
        self, task: str, history: List[Dict[str, Any]], result: Dict[str, Any]
    ) -> Optional[str]:
        """
        构建轨迹文本。
        关键：必须过滤掉本轮use_memory()插入的记忆内容，只保留原始的交互。
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

        # MEMs使用两个模板标题
        workflow_prefix = "Here are some useful workflows learned from past similar episodes"
        mem0_prefix = "Based on your previous interactions, here are relevant memories:"
        parts: List[str] = [f"Task: {task}"]

        for msg in history:
            role, content, msg_dict = extract_message_info(msg)
            if role is None:
                continue
            content = str(content).strip() if content else ""
            if not content:
                continue

            if role == "user":
                # 如果包含任一模板标题，根据 where 参数判断原始问题的位置
                if workflow_prefix in content:
                    if self.where == "front":
                        # where=front: 记忆在前，原始问题在后（模板前缀之后）
                        parts_split = content.split(workflow_prefix, 1)
                        if len(parts_split) > 1:
                            content = parts_split[1].strip()
                        else:
                            content = ""
                    else:  # tail
                        # where=tail: 原始问题在前，记忆在后（模板前缀之前）
                        content = content.split(workflow_prefix)[0].strip()
                elif mem0_prefix in content:
                    if self.where == "front":
                        # where=front: 记忆在前，原始问题在后（模板前缀之后）
                        parts_split = content.split(mem0_prefix, 1)
                        if len(parts_split) > 1:
                            content = parts_split[1].strip()
                        else:
                            content = ""
                    else:  # tail
                        # where=tail: 原始问题在前，记忆在后（模板前缀之前）
                        content = content.split(mem0_prefix)[0].strip()
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
        更新记忆：
        1. 调用 Trigger Model 决定更新哪个记忆源
        2. 如果需要更新 system_memory：
           - 调用 Workflow Induction 提取工作流
           - 调用 Workflow Management 进行增删改查
        3. 如果需要更新 personal_memory：
           - 调用 Mem0 的 update_memory
        """
        # 0. 检查过滤条件：如果配置了过滤，只有满足条件才调用 trigger model
        status = result.get("status", "")
        finish = result.get("finish", False)
        reward = result.get("reward", 0)
        is_success = finish or (status == "completed")
        
        print(f"[MEMs] update_memory called: task={task}, finish={finish}, status={status}, reward={reward}, "
              f"is_success={is_success}, success_only={self.config.trigger_model_success_only}, "
              f"reward_bigger_than_zero={self.config.trigger_model_reward_bigger_than_zero}")
        
        if self.config.trigger_model_success_only:
            if not is_success:
                print(f"[MEMs] Skipping trigger model: success_only=True but sample not completed "
                      f"(finish={finish}, status={status})")
                return
        
        if self.config.trigger_model_reward_bigger_than_zero:
            if reward <= 0:
                print(f"[MEMs] Skipping trigger model: reward_bigger_than_zero=True but reward={reward}")
                return
        
        # 1. 调用 Trigger Model
        print("[MEMs] Calling trigger model to decide memory sources...")
        trigger_result = self._call_trigger_model(history)
        if not trigger_result:
            print("[MEMs] WARNING: Trigger model failed, skipping memory update")
            return

        sources = trigger_result.get("sources", [])
        if not sources:
            print("[MEMs] Trigger model returned no sources to update")
            return

        print(f"[MEMs] Trigger model selected sources: {sources}")

        # 2. 更新 system_memory
        if "system_memory" in sources:
            print("[MEMs] Updating system_memory (workflow RAG)...")
            self._update_system_memory(task, history, result)
        else:
            print("[MEMs] Skipping system_memory update (not in sources)")

        # 3. 更新 personal_memory
        if "personal_memory" in sources:
            if self._mem0_memory:
                print("[MEMs] Updating personal_memory (Mem0)...")
                self._mem0_memory.update_memory(task, history, result)
            else:
                print("[MEMs] WARNING: Personal memory update requested but Mem0 not initialized")
        else:
            print("[MEMs] Skipping personal_memory update (not in sources)")

    def _update_system_memory(
        self, task: str, history: List[Dict[str, Any]], result: Dict[str, Any]
    ) -> None:
        """更新 system memory（工作流）"""
        # 构建轨迹文本
        trajectory_text = self._build_trajectory_text(task, history, result)
        if not trajectory_text:
            return

        # 调用 Workflow Induction
        new_workflows_text = self._call_workflow_induction(trajectory_text)
        if not new_workflows_text:
            logger.warning("[MEMs] Workflow induction failed")
            return

        # 加载现有工作流
        existing_workflows = self._load_workflows()

        # 调用 Workflow Management
        management_result = self._call_workflow_management(existing_workflows, new_workflows_text)
        if not management_result:
            logger.warning("[MEMs] Workflow management failed")
            return

        # 应用增删改查操作
        memory_ops = management_result.get("memory", [])
        if not memory_ops:
            print("[MEMs] No memory operations to apply")
            return

        print(f"[MEMs] Applying {len(memory_ops)} workflow management operation(s)...")

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
        print(f"[MEMs] Workflow management operations summary: ADD={add_count}, UPDATE={update_count}, DELETE={delete_count}, NONE={none_count}")

        # 保存更新后的工作流
        updated_workflows = list(workflow_dict.values())
        self._save_workflows(updated_workflows)
        
        # 如果 RAG 需要重建（UPDATE 或 DELETE 操作），从 workflows.json 重新加载并重建索引
        if rag_needs_rebuild and self._workflow_rag:
            self._rebuild_rag_index()

    def _rebuild_rag_index(self) -> None:
        """重建 RAG 索引：从 workflows.json 重新加载所有工作流并重建索引"""
        if not self._workflow_rag:
            return
        
        # 重新创建 RAG 实例（清空旧索引）
        try:
            self._workflow_rag = RAG(
                embedding_model=self.config.workflow_rag_embedding_model,
                top_k=self.config.workflow_rag_top_k,
                order=self.config.workflow_rag_order,
                seed=self.config.workflow_rag_seed,
            )
        except Exception as e:
            logger.warning(f"[MEMs] Failed to rebuild RAG index: {e}")
            return
        
        # 从 workflows.json 重新加载并插入所有工作流
        workflows = self._load_workflows()
        for wf in workflows:
            wf_text = wf.get("text", "")
            if wf_text:
                self._workflow_rag.insert(key=wf_text, value=wf_text)
        
        logger.info(f"[MEMs] Rebuilt RAG index with {len(workflows)} workflows")


def load_mems_from_yaml(config_path: str) -> MEMs:
    """从 YAML 配置文件加载 MEMs"""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    raw = cfg.get("mems", {}) or {}

    # 模型配置（统一使用一个模型）
    # 优先使用新的 model_name 配置，如果没有则尝试从旧配置中读取（向后兼容）
    model_name = str(raw.get("model_name", ""))
    if not model_name:
        # 向后兼容：尝试从旧配置中读取
        model_name = str(raw.get("trigger_model_name", "") or 
                        raw.get("workflow_induction_model", "") or 
                        raw.get("workflow_management_model", ""))
    if not model_name:
        raise ValueError("MEMs config must specify 'model_name' (or one of the legacy model configs)")

    # Prompt 配置
    workflow_induction_prompt = raw.get("workflow_induction_prompt", "") or ""
    trigger_prompt = raw.get("trigger_model_prompt", "") or ""  # 使用 trigger_model_prompt
    workflow_management_prompt = raw.get("workflow_management_prompt", "") or ""
    
    # Trigger Model 过滤配置
    trigger_model_config_raw = raw.get("trigger_model_config", {}) or {}
    trigger_model_success_only = bool(trigger_model_config_raw.get("success_only", True))
    trigger_model_reward_bigger_than_zero = bool(trigger_model_config_raw.get("reward_bigger_than_zero", True))

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

    # 工作流存储路径
    workflow_storage_path = Path("memory/MEMs/workflows.json")

    # Mem0 配置路径
    mem0_raw = raw.get("mem0", {}) or {}
    mem0_config_path = Path(mem0_raw.get("config_path", "memory/mem0/mem0.yaml"))

    config = MEMsConfig(
        model_name=model_name,
        workflow_induction_prompt=workflow_induction_prompt,
        trigger_prompt=trigger_prompt,
        workflow_management_prompt=workflow_management_prompt,
        trigger_model_success_only=trigger_model_success_only,
        trigger_model_reward_bigger_than_zero=trigger_model_reward_bigger_than_zero,
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
        mem0_config_path=mem0_config_path,
    )
    return MEMs(config)

