import json
import logging
import re
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

# Python 3.8 compatibility: functools.cache is available from Python 3.9+
try:
    from functools import cache
except ImportError:
    from functools import lru_cache
    cache = lru_cache(maxsize=None)

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# BLEU score calculation using n-gram overlap
from collections import Counter

from .task_base import Task, Session
from .typings import (
    SampleIndex,
    SampleStatus,
    TaskSampleExecutionResult,
    RewardHistoryItem
)
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam
)

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on conversation history.
Given a question, provide a clear and accurate answer based on the information from the conversations."""

# Default RoBERTa tokenizer path (can be overridden in task config)
DEFAULT_TOKENIZER_PATH = "C:/Users/123/Desktop/python/agent/agent-memory-bench/xlm-roberta-base"


def convert_session_to_history(session_dialogues: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    将 session 对话列表转换为 memory 格式的 history。
    
    Args:
        session_dialogues: 格式为 [{"speaker": "...", "dia_id": "...", "text": "..."}, ...]
    
    Returns:
        history: 格式为 [{"role": "user", "content": "'Speaker': 'text'"}, {"role": "assistant", "content": "..."}, ...]
    """
    history = []
    for i, dialogue in enumerate(session_dialogues):
        role = "user" if i % 2 == 0 else "assistant"
        speaker = dialogue.get("speaker", "")
        text = dialogue.get("text", "")
        content = f"'{speaker}': '{text}'"
        history.append({
            "role": role,
            "content": content
        })
    return history


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """简单的递归字典合并：override 覆盖 base。"""
    result = dict(base)
    for k, v in override.items():
        if (
            k in result
            and isinstance(result[k], dict)
            and isinstance(v, dict)
        ):
            result[k] = _deep_merge_dict(result[k], v)
        else:
            result[k] = v
    return result


def load_evaluate_agent_config(agent_name: str) -> Optional[Dict[str, Any]]:
    """
    加载并合并 evaluate_api.yaml + evaluate_agent.yaml 中的配置，返回：
    {
        "url": ...,
        "headers": {...},
        "body": {...},
    }
    """
    # 从 src/server/tasks/locomo/task.py 向上4级到项目根目录
    # parents[0] = src/server/tasks/locomo/
    # parents[1] = src/server/tasks/
    # parents[2] = src/server/
    # parents[3] = src/
    # parents[4] = 项目根目录
    ROOT_DIR = Path(__file__).resolve().parents[4]
    LLMAPI_DIR = ROOT_DIR / "configs" / "llmapi"
    
    agent_cfg_path = LLMAPI_DIR / "evaluate_agent.yaml"
    
    if not agent_cfg_path.exists():
        return None
    
    try:
        with agent_cfg_path.open("r", encoding="utf-8") as f:
            agents_cfg = yaml.safe_load(f) or {}
        
        logger = logging.getLogger(__name__)
        logger.info(f"Loading LLM judge config for agent: '{agent_name}'")
        logger.info(f"Available agents in evaluate_agent.yaml: {list(agents_cfg.keys())}")
        
        if agent_name not in agents_cfg:
            logger.warning(f"Agent '{agent_name}' not found in evaluate_agent.yaml. Available agents: {list(agents_cfg.keys())}")
            return None
        
        agent_cfg = agents_cfg[agent_name] or {}
        logger.info(f"Found agent config: {agent_cfg}")
        
        # 处理 import 字段：如果存在 import，读取对应的文件；否则使用 evaluate_api.yaml
        import_path = agent_cfg.get("import", "./evaluate_api.yaml")
        logger.info(f"Import path from agent config: {import_path}")
        
        if import_path.startswith("./"):
            api_cfg_path = LLMAPI_DIR / import_path[2:]  # 移除 "./"
        else:
            api_cfg_path = LLMAPI_DIR / import_path
        
        logger.info(f"Trying to load API config from: {api_cfg_path}")
        
        if not api_cfg_path.exists():
            # 如果 import 的文件不存在，尝试使用 evaluate_api.yaml
            logger.warning(f"Import file {api_cfg_path} does not exist, trying evaluate_api.yaml")
            api_cfg_path = LLMAPI_DIR / "evaluate_api.yaml"
            if not api_cfg_path.exists():
                logger.error(f"evaluate_api.yaml also does not exist at {api_cfg_path}")
                return None
        
        # 读取基础配置（从 import 指定的文件或 evaluate_api.yaml）
        with api_cfg_path.open("r", encoding="utf-8") as f:
            api_cfg = yaml.safe_load(f) or {}
        
        base_params = api_cfg.get("parameters", {}) or {}
        agent_params = agent_cfg.get("parameters", {}) or {}
        
        # 深度合并 parameters（agent 覆盖 api）
        merged_params = _deep_merge_dict(base_params, agent_params)
        
        url = merged_params.get("url") or api_cfg.get("parameters", {}).get("url")
        if not url:
            logger.error(f"No URL found in merged config. merged_params keys: {list(merged_params.keys())}, api_cfg parameters keys: {list(api_cfg.get('parameters', {}).keys())}")
            return None
        
        headers = merged_params.get("headers", {}) or api_cfg.get("parameters", {}).get("headers", {})
        body = merged_params.get("body", {}) or api_cfg.get("parameters", {}).get("body", {})
        
        logger.info(f"Successfully loaded LLM judge config: url={url}, model={body.get('model', 'N/A')}")
        
        return {
            "url": url,
            "headers": headers,
            "body": body,
        }
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to load evaluate agent config: {e}", exc_info=True)
        return None


class LocomoBaseTask(Task):
    """Locomo 任务的基类"""
    
    def __init__(
        self, 
        data_file: str,
        llm_judge_agent: str = "gpt-4o-mini",  # 从 evaluate_agent.yaml 中选择 agent
        tokenizer_path: Optional[str] = None,  # RoBERTa tokenizer 路径
        **kwargs
    ):
        super().__init__(**kwargs)
        self.logger = logging.getLogger(__name__)
        self.data_file = data_file
        self.llm_judge_agent = llm_judge_agent
        
        # 初始化 RoBERTa tokenizer
        self.tokenizer = None
        if HAS_TRANSFORMERS:
            tokenizer_path = tokenizer_path or DEFAULT_TOKENIZER_PATH
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                self.logger.info(f"Loaded RoBERTa tokenizer from: {tokenizer_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load tokenizer from {tokenizer_path}: {e}. BLEU score will be 0.")
                self.logger.warning("Please ensure transformers is installed: pip install transformers")
        else:
            self.logger.warning("transformers not installed. BLEU score will be 0. Install with: pip install transformers")
        
        # 从 configs/llmapi/evaluate_api.yaml 和 evaluate_agent.yaml 加载 LLM judge 配置
        self.llm_judge_config = None
        if HAS_REQUESTS:
            try:
                self.logger.info(f"Attempting to load LLM judge config for agent: '{llm_judge_agent}'")
                self.llm_judge_config = load_evaluate_agent_config(llm_judge_agent)
                if self.llm_judge_config:
                    self.logger.info(f"✓ Successfully loaded LLM judge config for agent: {llm_judge_agent}")
                    self.logger.info(f"  -> URL: {self.llm_judge_config.get('url', 'N/A')}")
                    self.logger.info(f"  -> Model: {self.llm_judge_config.get('body', {}).get('model', 'N/A')}")
                else:
                    self.logger.error(f"✗ LLM judge agent '{llm_judge_agent}' not found or config invalid. LLM judge will return 0.")
                    self.logger.error(f"  This means _llm_judge() will always return 0. Please check:")
                    self.logger.error(f"  1. Agent name '{llm_judge_agent}' exists in configs/llmapi/evaluate_agent.yaml")
                    self.logger.error(f"  2. The import file (e.g., api.yaml) exists and has correct structure")
                    self.logger.error(f"  3. The config has 'url' field in parameters")
            except Exception as e:
                self.logger.error(f"✗ Failed to load LLM judge config: {e}. LLM judge will return 0.", exc_info=True)
                self.llm_judge_config = None
        else:
            self.logger.warning("requests not installed, LLM judge will return 0")
        
        # 加载数据
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not data or len(data) == 0:
            raise ValueError(f"Empty data file: {data_file}")
        
        item = data[0]
        self.conversation = item.get("conversation", {})
        self.qa_list = item.get("qa", [])
        
        # 按 where 字段组织 qa（where -> [qa_index, ...]）
        self.qa_by_session: Dict[int, List[int]] = defaultdict(list)
        for idx, qa in enumerate(self.qa_list):
            where = qa.get("where")
            if where is not None:
                self.qa_by_session[where].append(idx)
        
        # 获取所有 session ID（按数字排序）
        self.session_ids = sorted([int(k.replace("session_", "")) 
                                   for k in self.conversation.keys() 
                                   if k.startswith("session_") and not k.endswith("_date_time")])
        
        self.logger.info(f"Loaded {len(self.qa_list)} QA pairs, {len(self.session_ids)} sessions")
        self.logger.info(f"Session IDs: {self.session_ids}")
        self.logger.info(f"QA distribution by session: {dict(self.qa_by_session)}")
    
    def get_session_history(self, session_id: int) -> List[Dict[str, Any]]:
        """获取指定 session 的对话历史（转换为 memory 格式）"""
        session_key = f"session_{session_id}"
        session_dialogues = self.conversation.get(session_key, [])
        if not session_dialogues:
            self.logger.warning(f"Session {session_id} not found")
            return []
        return convert_session_to_history(session_dialogues)
    
    def get_qa_indices_for_session(self, session_id: int) -> List[int]:
        """获取指定 session 的所有 QA 索引，过滤掉 category=5 的任务"""
        indices = self.qa_by_session.get(session_id, [])
        # 过滤掉 category=5 的任务（mem0 的做法）
        filtered_indices = []
        for idx in indices:
            if idx < len(self.qa_list):
                qa_item = self.qa_list[idx]
                category = qa_item.get("category", None)
                if category is not None and int(category) == 5:
                    continue  # 跳过 category=5 的任务
                filtered_indices.append(idx)
        return filtered_indices
    
    @cache
    def get_indices(self) -> List[SampleIndex]:
        """返回所有 QA 的索引（用于 offline 模式的数据分割），过滤掉 category=5 的任务"""
        # 过滤掉 category=5 的任务（mem0 的做法）
        indices = []
        for idx, qa_item in enumerate(self.qa_list):
            category = qa_item.get("category", None)
            if category is not None and int(category) == 5:
                continue  # 跳过 category=5 的任务
            indices.append(idx)
        return indices
    
    def _simple_tokenize(self, text: str) -> List[str]:
        """简单的 tokenization（用于 F1 计算）"""
        text = str(text).lower()
        # 移除标点，分割
        text = re.sub(r'[.,!?;:]', ' ', text)
        return text.split()
    
    def _calculate_f1_score(self, predicted: str, gold: str) -> float:
        """计算 token-based F1 分数"""
        pred_tokens = set(self._simple_tokenize(predicted))
        gold_tokens = set(self._simple_tokenize(gold))
        
        if not pred_tokens or not gold_tokens:
            return 0.0
        
        common_tokens = pred_tokens & gold_tokens
        
        if not common_tokens:
            return 0.0
        
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(gold_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return float(f1)
    
    def _calculate_bleu_score(self, predicted: str, gold: str) -> float:
        """
        计算 BLEU 分数（使用 BLEU-1，基于 RoBERTa tokenizer）
        
        使用 RoBERTa tokenizer 进行 tokenization，然后计算 n-gram 重叠度。
        BLEU-1 只考虑 unigram（单个 token）的重叠。
        """
        if not self.tokenizer:
            return 0.0
        
        try:
            # 使用 RoBERTa tokenizer 进行 tokenization
            # 注意：tokenizer 返回的是 token 字符串列表（如 ['hello', 'Ġworld']）
            pred_tokens = self.tokenizer.tokenize(predicted.lower())
            ref_tokens = self.tokenizer.tokenize(gold.lower())
            
            # 计算 BLEU-1（unigram precision）
            # BLEU-1 = (matched unigrams) / (total unigrams in prediction)
            if len(pred_tokens) == 0:
                return 0.0
            
            # 计算匹配的 unigrams
            ref_counter = Counter(ref_tokens)
            pred_counter = Counter(pred_tokens)
            
            # 计算匹配的 token 数量（取 min(ref_count, pred_count)）
            matched = sum(min(ref_counter[token], pred_counter[token]) for token in pred_counter)
            
            # BLEU-1 = matched / total_pred_tokens
            bleu1_score = matched / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
            
            return float(bleu1_score)
        except Exception as e:
            self.logger.warning(f"BLEU score calculation failed: {e}")
            return 0.0
    
    def _llm_judge(self, question: str, gold_answer: str, predicted_answer: str) -> int:
        """使用 LLM judge 评估答案（返回 0 或 1）"""
        if not self.llm_judge_config or not HAS_REQUESTS:
            if not self.llm_judge_config:
                self.logger.debug("LLM judge config is None, returning 0")
            if not HAS_REQUESTS:
                self.logger.debug("requests not available, returning 0")
            return 0
        
        ACCURACY_PROMPT = """
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user), 
    (2) a 'gold' (ground truth) answer, 
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. 

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {predicted_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. 
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""
        
        try:
            url = self.llm_judge_config["url"]
            headers = self.llm_judge_config["headers"]
            base_body = self.llm_judge_config["body"].copy()
            
            prompt_content = ACCURACY_PROMPT.format(
                question=question,
                gold_answer=gold_answer,
                predicted_answer=predicted_answer
            )
            
            body = {
                **base_body,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt_content,
                    }
                ],
            }
            
            # 如果 body 中有 model 字段，使用它；否则尝试从 agent 名称推断
            if "model" not in body:
                body["model"] = self.llm_judge_agent
            
            self.logger.info(f"Calling LLM judge: url={url}, model={body.get('model', 'N/A')}")
            
            response = requests.post(url, headers=headers, json=body, timeout=300)
            response.raise_for_status()
            
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            self.logger.info(f"LLM judge response (first 500 chars): {content[:500]}...")  # 记录前500个字符
            
            if not content:
                self.logger.warning("LLM judge returned empty content")
                return 0
            
            # 尝试多种方法提取 JSON
            label = None
            score = 0
            
            # 方法1: 尝试找到完整的 JSON 对象（支持嵌套）
            json_patterns = [
                r'\{[^{}]*"label"\s*:\s*"[^"]*"[^{}]*\}',  # 简单 JSON
                r'\{[^}]*"label"[^}]*\}',  # 原始模式
                r'\{"label"\s*:\s*"[^"]*"\}',  # 最简模式
            ]
            
            for pattern in json_patterns:
                json_match = re.search(pattern, content, re.IGNORECASE)
                if json_match:
                    try:
                        json_str = json_match.group()
                        self.logger.debug(f"Found JSON pattern: {json_str}")
                        label_data = json.loads(json_str)
                        label = label_data.get("label", "").upper()
                        if label in ("CORRECT", "WRONG"):
                            score = 1 if label == "CORRECT" else 0
                            self.logger.info(f"LLM judge parsed JSON: label={label}, score={score}")
                            return score
                    except json.JSONDecodeError as e:
                        self.logger.debug(f"JSON parse failed for pattern '{pattern}': {e}, matched text: {json_match.group()}")
                        continue
            
            # 方法2: 尝试提取整个响应中的 JSON（可能跨多行）
            try:
                # 查找所有可能的 JSON 对象
                json_objects = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content)
                for json_str in json_objects:
                    try:
                        label_data = json.loads(json_str)
                        if "label" in label_data:
                            label = str(label_data.get("label", "")).upper()
                            if label in ("CORRECT", "WRONG"):
                                score = 1 if label == "CORRECT" else 0
                                self.logger.info(f"LLM judge parsed JSON (method 2): label={label}, score={score}")
                                return score
                    except json.JSONDecodeError:
                        continue
            except Exception as e:
                self.logger.debug(f"Method 2 failed: {e}")
            
            # 方法3: 如果没有找到 JSON，尝试直接查找 CORRECT/WRONG 关键词
            content_upper = content.upper()
            if "CORRECT" in content_upper:
                # 确保不是 "INCORRECT" 的一部分
                correct_idx = content_upper.find("CORRECT")
                if correct_idx > 0:
                    prev_word = content_upper[max(0, correct_idx-10):correct_idx].strip()
                    if "IN" not in prev_word[-2:]:  # 不是 "INCORRECT"
                        self.logger.info("LLM judge found CORRECT keyword (no JSON)")
                        return 1
            elif "WRONG" in content_upper:
                self.logger.info("LLM judge found WRONG keyword (no JSON)")
                return 0
            
            # 方法4: 如果都没有找到，默认返回 0
            self.logger.warning(f"LLM judge could not parse response. Content preview: {content[:200]}...")
            self.logger.warning("No valid JSON or keyword found, returning 0")
            return 0
        except Exception as e:
            self.logger.warning(f"LLM judge evaluation failed: {e}", exc_info=True)
            return 0
    
    def _evaluate_answer(self, question: str, predicted: str, gold: Any) -> Tuple[Dict[str, float]]:
        """
        评估答案（同时计算所有指标：BLEU、F1、LLM judge）
        
        Returns:
            metrics: 所有评估指标的字典
            {
                "f1_score": float,
                "bleu_score": float,
                "llm_score": float (0 or 1),
            }
        """
        gold_str = str(gold) if gold is not None else ""
        predicted_str = str(predicted).strip()
        
        # 如果 gold_answer 为空，LLM judge 可能无法正确评估，记录警告
        if not gold_str:
            self.logger.warning(f"Gold answer is empty for question: {question[:50]}...")
        
        # 同时计算所有指标
        f1_score = self._calculate_f1_score(predicted_str, gold_str)
        bleu_score = self._calculate_bleu_score(predicted_str, gold_str)
        self.logger.info(f"Calculating LLM judge score (gold='{gold_str[:50]}...', predicted='{predicted_str[:50]}...')")
        llm_score = self._llm_judge(question, gold_str, predicted_str)
        self.logger.info(f"LLM judge returned: {llm_score}")
        
        metrics = {
            "f1_score": f1_score,
            "bleu_score": bleu_score,
            "llm_score": float(llm_score),
        }
        
        return metrics
    
    def _extract_answer_from_messages(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """从消息历史中提取 LLM 的答案"""
        # 查找最后一个 assistant 消息的 content
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if content:
                    # 清理答案：移除可能的格式标记
                    content = content.strip()
                    # 移除 "Answer:" 等前缀
                    content = re.sub(r'^(Answer|The answer|Final answer)[:\s]+', '', content, flags=re.IGNORECASE)
                    return content.strip()
        return None
    
    def sync_start_sample(self, index: SampleIndex, session: Session) -> TaskSampleExecutionResult:
        """启动一个 QA 样本"""
        self.logger.info(f'Starting sample {index} with session id {session.id}')
        
        if index >= len(self.qa_list):
            self.logger.error(f"Invalid index {index}, total QA count: {len(self.qa_list)}")
            return TaskSampleExecutionResult(status=SampleStatus.AGENT_VALIDATION_FAILED)
        
        qa_item = self.qa_list[index]
        question = qa_item.get("question", "")
        gold_answer = qa_item.get("answer", "")
        category = qa_item.get("category", None)  # 提取 category
        
        self.logger.info(f'[session {session.id}] Processing question: {question[:50]}...')
        self.logger.info(f'[session {session.id}] Gold answer: {gold_answer}')
        if category is not None:
            self.logger.info(f'[session {session.id}] Category: {category}')
        
        # 注入 system prompt
        session.inject(ChatCompletionSystemMessageParam(
            role='system',
            content=SYSTEM_PROMPT
        ))
        
        # 注入 question 作为 user message
        session.inject(ChatCompletionUserMessageParam(
            role='user',
            content=question
        ))
        
        # 等待 LLM 回答（通过 sync_action）
        try:
            response = session.sync_action()
            predicted_answer = self._extract_answer_from_messages(response.messages)
            
            if predicted_answer is None:
                self.logger.warning(f'[session {session.id}] No answer extracted from LLM response')
                # 对于 locomo 任务，无法提取答案时 reward 设为 0（等于 llm_score）
                empty_metrics = {
                    "f1_score": 0.0,
                    "bleu_score": 0.0,
                    "llm_score": 0.0,
                }
                session.inject(RewardHistoryItem(
                    reward=0.0,  # 无法提取答案时 reward 为 0（等于 llm_score）
                    metrics=empty_metrics
                ))
                return TaskSampleExecutionResult(
                    status=SampleStatus.COMPLETED,
                    result={
                        "question": question,
                        "gold_answer": gold_answer,
                        "predicted_answer": None,
                        "category": category,  # 添加 category
                        "metrics": empty_metrics,
                    }
                )
            
            self.logger.info(f'[session {session.id}] Predicted answer: {predicted_answer}')
            
            # 评估答案（同时计算所有指标：BLEU、F1、LLM judge）
            self.logger.info(f'[session {session.id}] Starting answer evaluation...')
            if not self.llm_judge_config:
                self.logger.warning(f'[session {session.id}] LLM judge config is None! This should have been logged during initialization.')
                self.logger.warning(f'[session {session.id}] LLM judge will return 0. Check initialization logs above.')
            metrics = self._evaluate_answer(question, predicted_answer, gold_answer)
            
            self.logger.info(
                f'[session {session.id}] Evaluation metrics: '
                f'f1={metrics["f1_score"]:.4f}, bleu={metrics["bleu_score"]:.4f}, llm={metrics["llm_score"]:.0f}'
            )
            
            # 对于 locomo 任务，reward 等于 llm_score（0 或 1）
            # llm_score 是 LLM judge 评估的结果，表示答案是否正确
            session.inject(RewardHistoryItem(
                reward=metrics["llm_score"],  # 使用 llm_score 作为 reward（0 或 1）
                metrics=metrics  # 保存所有评估指标：f1_score, bleu_score, llm_score
            ))
            
            # 返回结果，包含所有评估指标（用于后续分析脚本）
            return TaskSampleExecutionResult(
                status=SampleStatus.COMPLETED,
                result={
                    "question": question,
                    "gold_answer": gold_answer,
                    "predicted_answer": predicted_answer,
                    "category": category,  # 添加 category
                    "metrics": metrics,  # 保存所有指标：f1_score, bleu_score, llm_score
                }
            )
            
        except Exception as e:
            self.logger.error(f'[session {session.id}] Error during answer evaluation: {e}', exc_info=True)
            # 对于 locomo 任务，异常情况下 reward 设为 0（因为 llm_score 为 0）
            empty_metrics = {
                "f1_score": 0.0,
                "bleu_score": 0.0,
                "llm_score": 0.0,
            }
            session.inject(RewardHistoryItem(
                reward=0.0,  # 异常情况下 reward 为 0（等于 llm_score）
                metrics=empty_metrics
            ))
            return TaskSampleExecutionResult(
                status=SampleStatus.AGENT_VALIDATION_FAILED,
                result={
                    "category": category,  # 添加 category（即使出错也记录）
                    "question": question if 'question' in locals() else "",
                    "gold_answer": gold_answer if 'gold_answer' in locals() else "",
                    "predicted_answer": None,
                    "metrics": empty_metrics,
                }
            )
    
    def get_gold_answer(self, index: SampleIndex) -> Any:
        """获取标准答案（用于验证）"""
        if index >= len(self.qa_list):
            return None
        return self.qa_list[index].get("answer")


class Locomo0Task(LocomoBaseTask):
    """Locomo0 任务"""
    pass


class Locomo1Task(LocomoBaseTask):
    """Locomo1 任务"""
    pass


class Locomo2Task(LocomoBaseTask):
    """Locomo2 任务"""
    pass


class Locomo3Task(LocomoBaseTask):
    """Locomo3 任务"""
    pass


class Locomo4Task(LocomoBaseTask):
    """Locomo4 任务"""
    pass


class Locomo5Task(LocomoBaseTask):
    """Locomo5 任务"""
    pass


class Locomo6Task(LocomoBaseTask):
    """Locomo6 任务"""
    pass


class Locomo7Task(LocomoBaseTask):
    """Locomo7 任务"""
    pass


class Locomo8Task(LocomoBaseTask):
    """Locomo8 任务"""
    pass


class Locomo9Task(LocomoBaseTask):
    """Locomo9 任务"""
    pass

