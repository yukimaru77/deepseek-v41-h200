import copy
import json

import pytest

from tensorfold.cuda.chat_template import ChatTemplate
from tensorfold.families.deepseek_v41.vendor.encoding import encode_messages


@pytest.fixture
def template(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "deepseek_v41"}))
    (tmp_path / "tokenizer_config.json").write_text("{}")
    return ChatTemplate(tmp_path)


@pytest.mark.parametrize("thinking,effort", [(False, None), (True, "low"), (True, "high"), (True, "max")])
def test_chat_matches_official_encoding(template, thinking, effort):
    messages = [{"role": "system", "content": "日本語で答える。"},
                {"role": "user", "content": "17×19は？"},
                {"role": "assistant", "content": "323", "reasoning_content": "17×20−17。"},
                {"role": "system", "content": "次は答えだけ。"},
                {"role": "user", "content": "7×8は？"}]
    expected = encode_messages(messages, thinking_mode="thinking" if thinking else "chat", reasoning_effort=effort)
    assert template.render(messages, tools=None, enable_thinking=thinking,
                           extra={"reasoning_effort": effort}) == expected


def test_tools_roundtrip_and_order(template):
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {
        "type": "object", "properties": {"name": {"type": "string"}}}}}]
    messages = [{"role": "user", "content": "葵と蓮を調べて。"},
                {"role": "assistant", "content": None, "reasoning_content": "二人の記録を読む。", "tool_calls": [
                    {"id": "a", "type": "function", "function": {"name": "lookup", "arguments": '{"name":"葵"}'}},
                    {"id": "b", "type": "function", "function": {"name": "lookup", "arguments": '{"name":"蓮"}'}}]},
                {"role": "tool", "tool_call_id": "b", "content": "蓮=2"},
                {"role": "tool", "tool_call_id": "a", "content": "葵=1"}]
    before = copy.deepcopy(messages)
    text = template.render(messages, tools=tools, enable_thinking=True, extra={"reasoning_effort": "high"})
    expected = encode_messages([{"role": "system", "content": "", "tools": tools}, *messages],
                               thinking_mode="thinking", reasoning_effort="high")
    assert text == expected
    assert text.index("<tool_result>葵=1") < text.index("<tool_result>蓮=2")
    assert '<｜DSML｜ invoke name="lookup">' in text
    assert text.endswith("<｜Assistant｜><think>")
    assert messages == before


def test_history_excludes_only_generation_header(template):
    messages = [{"role": "user", "content": "hello"}]
    generated = template.template.render(messages=messages, enable_thinking=False, add_generation_prompt=True)
    history = template.template.render(messages=messages, enable_thinking=False, add_generation_prompt=False)
    assert generated == history + "<｜Assistant｜></think>"
    messages.append({"role": "assistant", "content": "こんにちは"})
    history = template.template.render(messages=messages, add_generation_prompt=False)
    assert history.endswith("こんにちは<｜end▁of▁sentence｜>")
