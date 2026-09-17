"""Thin, auditable wrapper around the Anthropic SDK.

Every model call goes through `call_structured` or `call_text`, which log stage, tokens and
cost to the run directory (DESIGN.md §3.6). No data rows ever enter a prompt (§3.8): callers
pass schemas, aggregates and paper pages only.
"""
from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel

MODEL = os.environ.get("REPLICATOR_MODEL", "claude-opus-5")
CHEAP_MODEL = os.environ.get("REPLICATOR_CHEAP_MODEL", "claude-sonnet-5")

# $/MTok, from the claude-api skill table (2026-06). Used for the cost cap only.
PRICES = {
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-fable-5-1": (10.0, 50.0),
}

T = TypeVar("T", bound=BaseModel)


@dataclass
class CallLog:
    stage: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    wall_seconds: float
    cost_usd: float


@dataclass
class LLM:
    """One instance per run; accumulates cost against the plan's cap."""
    log_dir: Path
    max_cost_usd: float = 150.0
    calls: list[CallLog] = field(default_factory=list)
    _client: Any = None

    @property
    def client(self):
        if self._client is None:
            import anthropic
            # A stalled stream must fail fast: 5-minute cap per call, one retry. A builder turn
            # that produces nothing is cheaper than a turn that eats the stage budget.
            self._client = anthropic.Anthropic(timeout=300.0, max_retries=1)
        return self._client

    @property
    def cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    def _record(self, stage: str, model: str, resp: Any, t0: float) -> None:
        u = resp.usage
        pin, pout = PRICES.get(model, (5.0, 25.0))
        cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
        cost = (u.input_tokens * pin + u.output_tokens * pout + cache_read * pin * 0.1) / 1e6
        rec = CallLog(stage, model, u.input_tokens, u.output_tokens, cache_read, time.time() - t0, cost)
        self.calls.append(rec)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with (self.log_dir / "llm_calls.jsonl").open("a") as f:
            f.write(json.dumps(rec.__dict__) + "\n")
        if self.cost_usd > self.max_cost_usd:
            raise CostCapExceeded(f"cost {self.cost_usd:.2f} > cap {self.max_cost_usd:.2f}")

    # ------------------------------------------------------------------ content helpers
    @staticmethod
    def pdf_block(pdf_path: Path) -> dict:
        data = base64.standard_b64encode(pdf_path.read_bytes()).decode()
        return {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": data},
                "cache_control": {"type": "ephemeral"}}

    @staticmethod
    def text_document_block(txt_path: Path, title: str = "paper") -> dict:
        return {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": txt_path.read_text()},
                "title": title, "cache_control": {"type": "ephemeral"}}

    @staticmethod
    def paper_block(path: Path) -> dict:
        return LLM.pdf_block(path) if path.suffix.lower() == ".pdf" else LLM.text_document_block(path)

    @staticmethod
    def image_block(png_path: Path) -> dict:
        data = base64.standard_b64encode(png_path.read_bytes()).decode()
        return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}

    # ------------------------------------------------------------------ calls
    GRAMMAR_LIMIT_BYTES = 2500  # above this the API rejects the compiled grammar; use prompt-guided JSON

    def call_structured(self, stage: str, system: str, content: list[dict], schema: Type[T],
                        model: Optional[str] = None, effort: str = "high", max_tokens: int = 32000) -> T:
        model = model or MODEL
        strict = _strict_schema(schema)
        if len(json.dumps(strict)) > self.GRAMMAR_LIMIT_BYTES:
            return self._call_json_guided(stage, system, content, schema, strict, model, effort, max_tokens)
        t0 = time.time()
        def _go():
            with self.client.with_options(timeout=900.0, max_retries=2).messages.stream(
                model=model,
                max_tokens=max_tokens,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": content}],
                thinking={"type": "adaptive"},
                output_config={"effort": effort, "format": {"type": "json_schema", "schema": _strict_schema(schema)}},
            ) as stream:
                return stream.get_final_message()
        resp = _with_transport_retry(_go, stage=stage)
        self._record(stage, model, resp, t0)
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"model refused at stage {stage}: {resp.stop_details}")
        text = next(b.text for b in resp.content if b.type == "text")
        return schema.model_validate(json.loads(text))

    def _call_json_guided(self, stage, system, content, schema: Type[T], strict: dict, model, effort, max_tokens) -> T:
        """Large schemas: the schema goes in the prompt, the reply must be one JSON object, pydantic
        validates, and one repair round-trip is allowed on a validation error."""
        sys_text = (system + "\n\nRespond with exactly one JSON object and nothing else (no prose, no code fence). "
                    "It must validate against this JSON schema:\n" + json.dumps(strict))
        msgs = [{"role": "user", "content": content}]
        last_err = None
        for attempt in range(3):
            t0 = time.time()
            def _go(msgs=msgs):
                with self.client.with_options(timeout=900.0, max_retries=2).messages.stream(
                    model=model, max_tokens=max_tokens,
                    system=[{"type": "text", "text": sys_text, "cache_control": {"type": "ephemeral"}}],
                    messages=msgs, thinking={"type": "adaptive"}, output_config={"effort": effort},
                ) as stream:
                    return stream.get_final_message()
            resp = _with_transport_retry(_go, stage=stage)
            self._record(stage if attempt == 0 else stage + ".repair", model, resp, t0)
            if resp.stop_reason == "refusal":
                raise RuntimeError(f"model refused at stage {stage}: {resp.stop_details}")
            text = "".join(b.text for b in resp.content if b.type == "text").strip()
            if text.startswith("```"):
                text = text.strip("`")
                text = text[text.find("{"):]
            body = text[text.find("{"): text.rfind("}") + 1]
            try:
                return schema.model_validate(_lenient_json(body))
            except Exception as e:  # noqa: BLE001
                last_err = e
                snippet = ""
                if isinstance(e, json.JSONDecodeError):
                    snippet = f"\nNear: ...{body[max(0, e.pos - 200): e.pos + 200]}..."
                msgs = msgs + [{"role": "assistant", "content": resp.content},
                               {"role": "user", "content": f"That JSON failed to parse or validate:\n{str(e)[:2000]}{snippet}\n"
                                                           "Return the complete corrected JSON object only, no prose."}]
        raise RuntimeError(f"stage {stage}: JSON did not validate after repair: {last_err}")

    def call_text(self, stage: str, system: str, content: list[dict], model: Optional[str] = None,
                  effort: str = "medium", max_tokens: int = 16000) -> str:
        model = model or MODEL
        t0 = time.time()
        with self.client.messages.stream(
            model=model, max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": content}],
            thinking={"type": "adaptive"}, output_config={"effort": effort},
        ) as stream:
            resp = stream.get_final_message()
        self._record(stage, model, resp, t0)
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"model refused at stage {stage}: {resp.stop_details}")
        return "".join(b.text for b in resp.content if b.type == "text")

    def tool_turn(self, stage: str, system: str, messages: list[dict], tools: list[dict],
                  model: Optional[str] = None, effort: str = "high", max_tokens: int = 16000):
        """One assistant turn of a manual tool loop. The orchestrator owns the loop and the stop."""
        model = model or MODEL
        t0 = time.time()
        import anthropic
        try:
            with self.client.messages.stream(
                model=model, max_tokens=max_tokens,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=messages, tools=tools,
                thinking={"type": "adaptive"}, output_config={"effort": effort},
            ) as stream:
                resp = stream.get_final_message()
        except Exception as e:  # noqa: BLE001
            if not _is_transport_error(e):
                raise
            self.log_dir.mkdir(parents=True, exist_ok=True)
            with (self.log_dir / "llm_calls.jsonl").open("a") as f:
                f.write(json.dumps({"stage": stage, "model": model, "error": type(e).__name__, "wall_seconds": time.time() - t0}) + "\n")
            return None
        self._record(stage, model, resp, t0)
        return resp


def _lenient_json(body: str):
    """json.loads with the usual model slips repaired: trailing commas, control characters in
    strings, a stray trailing code fence."""
    import re
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        pass
    fixed = re.sub(r",\s*([}\]])", r"\1", body)            # trailing commas
    fixed = fixed.replace("\r", "").replace("\t", " ")
    try:
        return json.loads(fixed, strict=False)
    except json.JSONDecodeError:
        pass
    # unescaped newlines inside strings: join lines that fall inside an open string
    out, in_str, esc = [], False, False
    for ch in fixed:
        if in_str and ch == "\n":
            out.append("\\n"); continue
        out.append(ch)
        if esc:
            esc = False
        elif ch == "\\":
            esc = True
        elif ch == '"':
            in_str = not in_str
    return json.loads("".join(out), strict=False)


def _is_transport_error(e: Exception) -> bool:
    name = type(e).__name__
    return name in ("RemoteProtocolError", "ReadError", "ConnectError", "APIConnectionError", "APITimeoutError",
                    "IncompleteRead", "ProtocolError") or "peer closed connection" in str(e)


def _with_transport_retry(fn, attempts: int = 3, stage: str = ""):
    """A connection that drops mid-stream surfaces as httpx RemoteProtocolError after the request
    already succeeded, which the SDK's own retry does not cover. Retry the whole call."""
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if not _is_transport_error(e):
                raise
            last = e
            time.sleep(min(60, 5 * 2 ** i))
    raise RuntimeError(f"stage {stage}: transport failed {attempts} times: {type(last).__name__}: {str(last)[:160]}")


class CostCapExceeded(RuntimeError):
    pass


def _strict_schema(model: Type[BaseModel]) -> dict:
    """Pydantic JSON schema with additionalProperties=false everywhere, as structured outputs require."""
    schema = model.model_json_schema()

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"].keys())
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(schema)
    return schema
