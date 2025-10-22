# llm_call.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json
from openai import OpenAI

@dataclass
class OpenaiModel:
    api_key: str
    base_url: str = "https://api.openai.com"
    model: str = "gpt-4o-mini"
    schema: Optional[Dict[str, Any]] = None
    system_prompt: Optional[str] = None
    timeout: float = 60.0

    def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        schema: Optional[Dict[str, Any]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        极简 Chat 调用（OpenAI SDK）
        - 二选一：prompt 文本 或 messages 列表
        - 若传 schema：使用 response_format.json_schema 进行结构化输出
        返回：
          {"text": str, "json": dict|None, "response": dict}
        """
        if (prompt is None) and (not messages):
            raise ValueError("Either prompt or messages is required.")
        if (prompt is not None) and messages:
            raise ValueError("Pass either prompt or messages, not both.")

        eff_schema = schema if schema is not None else self.schema

        # 构造 messages
        if messages is None:
            messages = [{"role": "user", "content": prompt}]  # type: ignore[arg-type]
        if self.system_prompt and not any(m.get("role") == "system" for m in messages):
            messages = [{"role": "system", "content": self.system_prompt}] + messages

        # 构造 response_format
        response_format = None
        if eff_schema is not None:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "auto_schema",
                    "schema": eff_schema,
                    "strict": True
                }
            }

        client = OpenAI(api_key=self.api_key, base_url=self.base_url.rstrip("/"))
        cclient = client.with_options(timeout=self.timeout) if hasattr(client, "with_options") else client

        params: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if response_format is not None:
            params["response_format"] = response_format
        if extra_params:
            params.update(extra_params)

        resp = cclient.chat.completions.create(**params)

        try:
            text = resp.choices[0].message.content or ""
        except Exception:
            text = ""

        out: Dict[str, Any] = {"text": text, "json": None, "response": resp.model_dump()}

        if eff_schema is not None:
            try:
                out["json"] = json.loads(text)
            except Exception:
                out["json"] = None
        return out
