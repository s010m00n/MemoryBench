"""
LLM Agent 模块
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
import yaml
from requests.exceptions import ReadTimeout, Timeout


ROOT_DIR = Path(__file__).resolve().parents[2]
LLMAPI_DIR = ROOT_DIR / "configs" / "llmapi"


class SimpleHTTPChatAgent:
    """
    一个最简版的 LLM agent：
    - 从 configs/llmapi/api.yaml + agent.yaml 读取 HTTP 配置
    - 调用 OpenAI 风格的 chat completions 接口，支持 tools / tool_choice=auto
    - 简单的 429 / 500 重试
    """

    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name
        self.url, self.headers, self.base_body = self._load_agent_config(agent_name)

    @staticmethod
    def _load_agent_config(agent_name: str) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        """
        复用 test_client.py 中的合并逻辑：
        - api.yaml 提供基础参数（url/headers/body/prompter/return_format）
        - agent.yaml 针对具体 agent 覆盖 body 中的 model/max_tokens 等
        """
        agent_cfg_path = LLMAPI_DIR / "agent.yaml"
        api_cfg_path = LLMAPI_DIR / "api.yaml"

        with agent_cfg_path.open("r", encoding="utf-8") as f:
            agents_cfg = yaml.safe_load(f) or {}
        if agent_name not in agents_cfg:
            raise ValueError(f"Agent '{agent_name}' not found in {agent_cfg_path}")

        agent_cfg = agents_cfg[agent_name] or {}

        with api_cfg_path.open("r", encoding="utf-8") as f:
            api_cfg = yaml.safe_load(f) or {}

        base_params = api_cfg.get("parameters", {}) or {}
        agent_params = agent_cfg.get("parameters", {}) or {}

        # 深度合并 body
        body = dict(base_params.get("body", {}) or {})
        body.update(agent_params.get("body", {}) or {})

        url = base_params.get("url") or api_cfg.get("parameters", {}).get("url")
        if not url:
            raise ValueError("URL not found in api.yaml / agent.yaml")

        headers = dict(base_params.get("headers", {}) or {})
        headers.update(agent_params.get("headers", {}) or {})

        return url, headers, body

    def inference(self, history: List[Dict[str, Any]], tools: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
        """
        单轮调用：给定完整 history（system+user+assistant...），返回一条 assistant 消息。
        """
        body: Dict[str, Any] = {
            **(self.base_body or {}),
            "messages": history,
        }
        # 支持 function calling（tools）
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        # 简单串行重试逻辑：429/500/超时/网络错误一直重试；非可重试错误直接抛出
        data: Dict[str, Any] | None = None
        attempt = 0

        while True:
            try:
                # 单次请求超时设置为 250 秒，避免单个样本阻塞过久
                resp = requests.post(self.url, headers=self.headers, json=body, timeout=250)
                # Too Many Requests / 500: 一直重试（线性递增，最大 60 秒）
                if resp.status_code in (429, 500):
                    # 线性递增：5 * (attempt + 1) 秒（5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 60, ...），最大 60 秒
                    wait_sec = min(5 * (attempt + 1), 60)
                    logging.warning(
                        f"LLM API HTTP {resp.status_code} (attempt {attempt + 1}), "
                        f"retrying after {wait_sec}s (linear backoff, max 60s)..."
                    )
                    time.sleep(wait_sec)
                    attempt += 1
                    continue
                # 对于其他 HTTP 错误（如 400 Bad Request），直接抛出
                resp.raise_for_status()
                data = resp.json()
                break
            except (ReadTimeout, Timeout) as e:
                # 超时错误：一直重试（线性递增，最大 60 秒）
                # 线性递增：5 * (attempt + 1) 秒（5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 60, ...），最大 60 秒
                wait_sec = min(5 * (attempt + 1), 60)
                logging.warning(
                    f"LLM API timeout (attempt {attempt + 1}), retrying after {wait_sec}s (linear backoff, max 60s)..."
                )
                time.sleep(wait_sec)
                attempt += 1
                continue
            except requests.exceptions.RequestException as e:
                # 其他网络错误（如连接错误）：一直重试（线性递增，最大 60 秒）
                # 线性递增：5 * (attempt + 1) 秒（5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 60, ...），最大 60 秒
                wait_sec = min(5 * (attempt + 1), 60)
                logging.warning(
                    f"LLM API network error (attempt {attempt + 1}): {str(e)}, retrying after {wait_sec}s (linear backoff, max 60s)..."
                )
                time.sleep(wait_sec)
                attempt += 1
                continue
            except Exception as e:
                # 其他错误（如 400 Bad Request）：不重试，直接抛出
                raise RuntimeError(
                    f"LLM API error {getattr(e, 'status_code', 'unknown')}: {str(e)}. "
                    f"Request snippet: {json.dumps(body)[:4000]}"
                ) from e

        # 如果成功跳出循环，data 一定不为 None（因为 break 前会赋值）
        # 这个检查实际上永远不会执行，但保留作为防御性编程
        if data is None:
            raise RuntimeError("LLM API call failed: no response data parsed (unexpected state)")
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"Empty choices from LLM API: {data}")
        message = choices[0].get("message") or {}
        # 确保至少有 role/content 字段
        if "role" not in message:
            message["role"] = "assistant"
        if "content" not in message:
            message["content"] = ""
        # 保存推理内容（如果存在）
        if "reasoning_content" in message:
            # reasoning_content 已经在 message 中，直接返回即可
            pass
        return message
