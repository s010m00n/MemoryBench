from __future__ import annotations

from typing import List, Dict, Any, Optional
import yaml
import json
import re
import random
import numpy as np
import logging

from src.utils.message_schema import (
    extract_message_info,
    enhance_messages_with_memory,
    extract_original_question,
    extract_question_from_history,
)

try:
    import torch
    import faiss
    from transformers import AutoTokenizer, AutoModel
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False
    print("Warning: faiss, torch, or transformers not installed. StreamICL will not work.")

from ..base import MemoryMechanism
from src.utils.message_schema import extract_message_info


class RAG:
    """
    RAG (Retrieval-Augmented Generation) 向量检索系统。
    参考 stream-bench 的实现。
    """
    
    def __init__(
        self,
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        top_k: int = 5,
        order: str = "similar_at_top",  # "similar_at_top" | "similar_at_bottom" | "random"
        seed: int = 42,
    ):
        if not HAS_DEPENDENCIES:
            raise ImportError("faiss, torch, or transformers not installed. Please install them to use StreamICL.")
        
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.embed_model = AutoModel.from_pretrained(embedding_model).eval()

        self.index = None
        self.id2evidence = dict()
        # 直接从模型配置读取 embedding 维度，避免不必要的推理
        self.embed_dim = self.embed_model.config.hidden_size
        self.insert_acc = 0
        
        self.seed = seed
        self.top_k = top_k
        self.order = order
        random.seed(self.seed)
        
        self.create_faiss_index()
    
    def create_faiss_index(self):
        """创建 FAISS 向量索引"""
        self.index = faiss.IndexFlatL2(self.embed_dim)
    
    def encode_data(self, sentence: str) -> np.ndarray:
        """对文本进行 embedding 编码"""
        encoded_input = self.tokenizer([sentence], padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            model_output = self.embed_model(**encoded_input)
            # CLS pooling
            sentence_embeddings = model_output[0][:, 0]
        feature = sentence_embeddings.numpy()[0]
        norm = np.linalg.norm(feature)
        return feature / norm
    
    def insert(self, key: str, value: str) -> None:
        """
        插入经验到 RAG 中。
        
        Args:
            key: question（用于检索的 key）
            value: formatted example chunk（存储的值）
        """
        embedding = self.encode_data(key).astype('float32')
        self.index.add(np.expand_dims(embedding, axis=0))
        self.id2evidence[str(self.insert_acc)] = value
        self.insert_acc += 1
    
    def retrieve(self, query: str, top_k: int) -> List[str]:
        """
        检索 top_k 个最相似的经验。
        
        Args:
            query: 当前 question
            top_k: 检索数量
            
        Returns:
            检索到的 formatted chunks 列表
        """
        if self.insert_acc == 0:
            return []
        
        embedding = self.encode_data(query).astype('float32')
        top_k = min(top_k, self.insert_acc)
        distances, indices = self.index.search(np.expand_dims(embedding, axis=0), top_k)
        distances = distances[0].tolist()
        indices = indices[0].tolist()
        
        results = [{'link': str(idx), '_score': {'faiss': dist}} for dist, idx in zip(distances, indices)]
        
        # 根据排序策略重新排序
        if self.order == "similar_at_bottom":
            results = list(reversed(results))
        elif self.order == "random":
            random.shuffle(results)
        
        text_list = [self.id2evidence[result["link"]] for result in results]
        return text_list


class StreamICLMemory(MemoryMechanism):
    """
    StreamICL 记忆机制：基于向量检索的 RAG 系统。
    
    参考 stream-bench 的实现：
    - 使用 FAISS 向量数据库存储经验
    - 根据当前 question 的 embedding 检索 top_k 个最相似的经验
    - 只存储正确答案（is_correct=True）
    - 支持两种插入位置：system 或 user message 中的占位符
    """
    
    def __init__(
        self,
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        top_k: int = 5,
        order: str = "similar_at_top",  # "similar_at_top" | "similar_at_bottom" | "random"
        success_only: bool = True,  # True: 只存储成功完成的样本（finish 或 status=="completed"），False: 都存储
        reward_bigger_than_zero: bool = False,  # True: 只存储 reward>0 的样本，False: 都存储
        prompt_template: str = "Here are some examples of the task you have completed:\n\n{examples}",
        where: str = "tail",  # "tail": 记忆放在 user question 后面 | "front": 记忆放在 user question 前面
        seed: int = 42,
    ):
        """
        初始化 StreamICL 记忆机制。

        Args:
            embedding_model: 用于编码的 embedding 模型
            top_k: 检索 top_k 个最相似的经验
            order: 检索结果的排序方式
            success_only: True 表示只存储成功完成的样本（finish 或 status=="completed"），False 表示都存储
            reward_bigger_than_zero: True 表示只存储 reward>0 的样本，False 表示都存储
            prompt_template: 记忆内容的模板
            where: "tail" 表示记忆放在 user question 后面，"front" 表示记忆放在 user question 前面
            seed: 随机种子
        """
        if not HAS_DEPENDENCIES:
            raise ImportError("faiss, torch, or transformers not installed. Please install them to use StreamICL.")

        # 保存 RAG 配置
        self.rag_config = dict(
            embedding_model=embedding_model,
            top_k=top_k,
            order=order,
            seed=seed,
        )
        self.success_only = success_only
        self.reward_bigger_than_zero = reward_bigger_than_zero
        self.prompt_template = prompt_template
        self.where = where

        # 提取 template title（从 prompt_template 中提取，用于识别增强后的消息）
        # 例如: "Here are some examples:\n\n{examples}" -> "Here are some examples:"
        self.template_title = self.prompt_template.split('{examples}')[0].strip()

        # 全局单一向量库（不再按任务分组）
        self.rag: Optional[RAG] = None
        try:
            self.rag = RAG(**self.rag_config)
        except Exception as e:
            raise ImportError(f"Failed to initialize RAG: {e}")
    
    def use_memory(self, task: str, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        基于当前任务名和原始 messages，返回改写后的 messages。
        记忆内容会追加到第一个 user message 的末尾。
        """
        # 提取当前 question（使用公共工具）
        template_titles = [self.template_title]
        question = extract_original_question(messages, where=self.where, template_titles=template_titles)
        if not question:
            return list(messages) if messages is not None else []

        # 检索相似经验（全局向量库，不按任务隔离）
        if not self.rag:
            return list(messages) if messages is not None else []
        shots = self.rag.retrieve(query=question, top_k=self.rag.top_k)

        if not shots:
            return list(messages) if messages is not None else []

        # 格式化经验文本
        fewshot_text = "\n\n\n".join(shots).replace("\\", "\\\\")
        memory_content = self.prompt_template.format(examples=fewshot_text)

        # 使用公共工具插入记忆
        return enhance_messages_with_memory(messages, memory_content, where=self.where)
    
    def update_memory(self, task: str, history: List[Dict[str, Any]], result: Dict[str, Any]) -> None:
        """
        在单个样本执行结束后调用，用于把新的轨迹/结果写入记忆。
        """
        finish = result.get("finish", False)
        status = result.get("status", "")
        reward = result.get("reward", 0)
        # success_only 只负责检查是否成功完成（finish 或 status），不涉及 reward
        is_success = finish or status == "completed"
        
        # 过滤：如果 success_only=True，只存储成功完成的样本（不涉及 reward）
        if self.success_only and not is_success:
            print(f"[StreamICL] Skipping sample storage: success_only=True but sample not completed (finish={finish}, status={status}, task={task})")
            logging.info(f"[StreamICL] Skipping sample storage: success_only=True but sample not completed (finish={finish}, status={status}, task={task})")
            return
        
        # 过滤：如果 reward_bigger_than_zero=True，只存储 reward>0 的样本
        if self.reward_bigger_than_zero:
            if reward <= 0:
                print(f"[StreamICL] Skipping sample storage: reward_bigger_than_zero=True but reward={reward} (task={task})")
                logging.info(f"[StreamICL] Skipping sample storage: reward_bigger_than_zero=True but reward={reward} (task={task})")
                return
        
        # 提取 question（用于检索的 key，使用公共工具）
        template_titles = [self.template_title]
        question = extract_question_from_history(history, where=self.where, template_titles=template_titles)
        if not question:
            print(f"[StreamICL] Skipping sample storage: No question extracted from history (task={task})")
            logging.info(f"[StreamICL] Skipping sample storage: No question extracted from history (task={task})")
            return
        
        # 格式化经验 chunk（参考 stream-bench 的 shot_template 格式）
        chunk = self._format_experience(history, result)
        
        # 插入到 RAG（全局向量库，不按任务隔离）
        if not self.rag:
            return
        self.rag.insert(key=question, value=chunk)
    
    def _format_experience(self, history: List[Dict[str, Any]], result: Dict[str, Any]) -> str:
        """
        格式化经验为 chunk 文本。

        格式：
        Question: {question}
        {answer}

        关键：必须过滤掉本轮use_memory()插入的few-shot examples，只保留原始的question和answer。
        """
        template_prefix = self.template_title

        # 1. 提取原始question（使用公共工具）
        template_titles = [template_prefix]
        question = extract_question_from_history(history, where=self.where, template_titles=template_titles)

        if not question:
            return ""

        # 2. 提取answer（assistant和tool的交互，跳过第一个user消息）
        answer_lines = []
        skip_first_user = True

        for msg in history:
            role, content, msg_dict = extract_message_info(msg) 
            if role is None or role == "system":
                continue

            content = content if content else ""

            # 跳过第一个user消息（已经作为question了）
            if role == "user":
                if skip_first_user:
                    skip_first_user = False
                    continue
                # 后续user消息：只有不包含模板标题的才保留
                if template_prefix not in content:
                    answer_lines.append(f"User: {content}")
                continue

            # 格式化assistant消息
            if role == "assistant":
                tool_calls = msg_dict.get("tool_calls", []) if msg_dict else []
                if tool_calls:
                    tool_calls_info = []
                    for tc in tool_calls:
                        func = tc.get("function", {})
                        func_name = func.get("name", "unknown")
                        func_args = func.get("arguments", "{}")
                        try:
                            args_dict = json.loads(func_args)
                            args_str = json.dumps(args_dict, ensure_ascii=False)
                        except:
                            args_str = func_args
                        tool_calls_info.append(f"{func_name}({args_str})")
                    tool_calls_str = " ".join(tool_calls_info)
                    answer_lines.append(f"Assistant: {tool_calls_str}" + (f" {content}" if content else ""))
                else:
                    if content:
                        answer_lines.append(f"Assistant: {content}")

            # 格式化tool消息
            elif role == "tool":
                tool_content = content[:500] + "..." if len(content) > 500 else content
                answer_lines.append(f"Tool: {tool_content}")

        answer = "\n".join(answer_lines) if answer_lines else "Completed successfully."
        return f"Question: {question}\n{answer}"


def load_stream_icl_from_yaml(config_path: str) -> StreamICLMemory:
    """
    从 memory/streamICL/streamICL.yaml 读取配置，构造 StreamICLMemory。
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    stream_icl_cfg = cfg.get("stream_icl", {})
    rag_cfg = stream_icl_cfg.get("rag", {})

    embedding_model = rag_cfg.get("embedding_model", "BAAI/bge-base-en-v1.5")
    top_k = rag_cfg.get("top_k", 5)
    order = rag_cfg.get("order", "similar_at_top")
    seed = rag_cfg.get("seed", 42)

    success_only = bool(stream_icl_cfg.get("success_only", True))
    reward_bigger_than_zero = bool(stream_icl_cfg.get("reward_bigger_than_zero", False))
    prompt_template = stream_icl_cfg.get("prompt_template", "Here are some examples of the task you have completed:\n\n{examples}")
    where = stream_icl_cfg.get("where", "tail")

    return StreamICLMemory(
        embedding_model=embedding_model,
        top_k=top_k,
        order=order,
        success_only=success_only,
        reward_bigger_than_zero=reward_bigger_than_zero,
        prompt_template=prompt_template,
        where=where,
        seed=seed,
    )

