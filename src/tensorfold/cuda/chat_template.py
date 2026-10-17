# Modified for the H200 variant (2026); see docs/H200-ATTRIBUTION.md.
"""The CUDA server's chat format: checkpoint Jinja or the DeepSeek V4.1 reference encoder."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from tensorfold.server.messages import _normalize_tool_call_arguments, late_system_role, normalize_messages
from tensorfold.server.request_options import effort_levels


class ChatTemplate:
    """The model's own Jinja chat template, rendered the way Hugging Face's apply_chat_template does."""

    def __init__(self, model_dir: Path):
        import jinja2
        import jinja2.ext
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        cfg = json.loads((model_dir / "tokenizer_config.json").read_text())
        source_path = model_dir / "chat_template.jinja"
        source = source_path.read_text() if source_path.exists() else cfg.get("chat_template")
        self.specials = {k: (v.get("content") if isinstance(v, dict) else v)
                         for k, v in cfg.items() if k in ("bos_token", "eos_token", "pad_token", "unk_token")}
        if not source and json.loads((model_dir / "config.json").read_text()).get("model_type") == "deepseek_v41":
            self.template = _DeepSeekV41Template()
            self.efforts = frozenset({"minimal", "low", "medium", "high", "xhigh", "max"})
            self.late_system = "system"
            return
        if not source:
            raise ValueError(f"No chat template is available in {model_dir}")

        def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
            return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

        def raise_exception(message):
            raise jinja2.exceptions.TemplateError(message)

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                            extensions=[jinja2.ext.loopcontrols])
        env.filters["tojson"] = tojson
        env.globals["raise_exception"] = raise_exception
        env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
        self.template = env.from_string(source)
        self.efforts = effort_levels(source)          # the reasoning efforts it names (the Mac reads the same ones)
        self.late_system = late_system_role(
            lambda messages: self.template.render(**self.specials, messages=messages, add_generation_prompt=False))

    def render(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None,
               enable_thinking: bool, extra: dict[str, Any] | None = None, allow_images: bool = False) -> str:
        messages = _normalize_tool_call_arguments(normalize_messages(messages, late_system=self.late_system,
                                                                     allow_images=allow_images))
        kwargs = dict(self.specials, messages=messages, tools=tools or None, add_generation_prompt=True,
                      enable_thinking=enable_thinking)
        kwargs.update(extra or {})
        return self.template.render(**kwargs)


class _DeepSeekV41Template:
    """Jinja-compatible render surface over DeepSeek's MIT reference encoder.

    The 1017 checkpoint ships encoding.py instead of a Jinja template. Keep the
    reference implementation in-tree so serving does not execute model-directory code.
    """

    def render(self, *, messages, tools=None, enable_thinking=False,
               add_generation_prompt=True, reasoning_effort=None, **kwargs):
        from copy import deepcopy
        from tensorfold.families.deepseek_v41.vendor.encoding import encode_messages

        messages = deepcopy(messages)
        if tools:
            if not messages or messages[0].get("role") != "system":
                messages.insert(0, {"role": "system", "content": ""})
            messages[0]["tools"] = tools
        efforts = {"minimal": 25, "low": 50, "medium": 62, "high": 75, "xhigh": 88, "max": 100}
        text = encode_messages(messages, thinking_mode="thinking" if enable_thinking else "chat",
                               reasoning_effort=efforts.get(reasoning_effort, reasoning_effort))
        if not add_generation_prompt:
            # The reference adds a generation header after a final user/tool/system
            # turn. History/marker discovery needs the same prompt without that header.
            for suffix in ("<｜Assistant｜><think>", "<｜Assistant｜></think>"):
                if text.endswith(suffix):
                    text = text[:-len(suffix)]
                    break
        return text
