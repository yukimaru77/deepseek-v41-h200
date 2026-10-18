"""DeepSeek's HTTP image frontend, exercised on CPU with synthetic images and no model weights."""

import base64
import copy
import io
import json
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")

from tensorfold.cuda.server import App
from tensorfold.families.deepseek_v41.cuda.app import DsApp
from tensorfold.families.deepseek_v41.cuda.vision import DsVision, PLACEHOLDER, VisionConfig, decode
from tensorfold.families.deepseek_v41.vendor.encoding import encode_messages
from tensorfold.server.errors import ContextLengthError, RequestError
from tensorfold.vision.images import ImageInput, ImageLimits, ImageSource, load_images


CONFIG = VisionConfig(layers=1, dim=12, heads=2, inter=24, patch=14, theta=10000, ratio=3,
                      max_tokens=994, min_pixels=42 * 42, max_wh_ratio=None, image_token_id=999, model_dim=12)


class Tokenizer:
    """Only image tokenization matters here; all text codepoints retain their ordering exactly."""

    @staticmethod
    def from_file(path):
        return Tokenizer()

    def encode(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        parts = text.split(PLACEHOLDER)
        ids = [ord(c) + 1000 for c in parts[0]]
        for part in parts[1:]:
            ids.extend([CONFIG.image_token_id, *(ord(c) + 1000 for c in part)])
        return NS(ids=ids)


@pytest.fixture(params=[App, DsApp], ids=["shared-app", "deepseek-app"])
def app(tmp_path, monkeypatch, request):
    tokenizers = pytest.importorskip("tokenizers")
    monkeypatch.setattr(tokenizers, "Tokenizer", Tokenizer)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "deepseek_v41",
                                                     "max_position_embeddings": 8192}))
    (tmp_path / "tokenizer_config.json").write_text("{}")
    engine = NS(vision=DsVision(CONFIG), generate=lambda *args, **kwargs: None)
    return request.param(engine, tmp_path, "test-model", sampling={"temperature": 0}, max_tokens=8)


def png(color="red", size=(42, 42)):
    data = io.BytesIO()
    Image.new("RGB", size, color).save(data, format="PNG")
    return data.getvalue()


def image_part(data):
    return {"type": "image_url", "image_url": {
        "url": "data:image/png;base64," + base64.b64encode(data).decode()}}


def user_message(data):
    return {"role": "user", "content": [{"type": "text", "text": "before"}, image_part(data),
                                        {"type": "text", "text": "after"}]}


def assert_official_prompt(app, messages, prepared):
    reference, media = encode_messages(messages, thinking_mode="chat", return_multi_modal_data=True)
    pictures = [decode(base64.b64decode(record["url"].split(",", 1)[1]), CONFIG)
                for record in media["images"]]
    expected = app.vision.expand(app.tok.encode(reference).ids, pictures)
    assert prepared.prompt == expected.token_ids == prepared.vision.token_ids
    assert prepared.vision.positions() == expected.positions()
    for (_, actual), (_, expected_pic) in zip(prepared.vision.spans, expected.spans):
        assert torch.equal(actual.patches, expected_pic.patches)
    return reference


def test_user_image_matches_the_official_encoder_and_legacy_preprocessing(app):
    messages = [user_message(png())]
    before = copy.deepcopy(messages)
    prepared = app.prepare({"messages": messages, "max_tokens": 8}, True)
    assert app.vision.tokenizer is app.tok  # the actual App constructor connects the frontend
    assert_official_prompt(app, messages, prepared)
    assert len(prepared.vision.spans) == 1
    assert messages == before


def test_parallel_tool_result_images_follow_the_official_call_order(app):
    def call(identifier):
        return {"id": identifier, "type": "function", "function": {"name": "screenshot", "arguments": "{}"}}

    messages = [{"role": "user", "content": "inspect both"},
                {"role": "assistant", "content": "", "tool_calls": [call("first"), call("second")]},
                {"role": "tool", "tool_call_id": "second", "content": [image_part(png("blue", (84, 42)))]},
                {"role": "tool", "tool_call_id": "first", "content": [image_part(png("red"))]},
                {"role": "user", "content": "compare"}]
    before = copy.deepcopy(messages)
    prepared = app.prepare({"messages": messages, "max_tokens": 8}, True)
    assert_official_prompt(app, messages, prepared)
    assert [picture.tokens for _, picture in prepared.vision.spans] == [4, 5]
    assert prepared.vision.spans[0][1].patches[0, :, 0, 0].tolist() == [1, -1, -1]
    assert prepared.vision.spans[1][1].patches[0, :, 0, 0].tolist() == [-1, -1, 1]
    assert messages == before


def test_prepare_uses_the_shared_loaders_oriented_rgb(app):
    from PIL import ImageOps

    source = Image.new("RGB", (14, 42), "red")
    source.paste("blue", (0, 0, 14, 14))
    exif = Image.Exif()
    exif[274] = 6  # a portrait stored on disk, displayed as landscape
    data = io.BytesIO()
    source.save(data, format="JPEG", exif=exif)
    url = "data:image/jpeg;base64," + base64.b64encode(data.getvalue()).decode()
    images = load_images([ImageSource(url)])
    assert (images[0].width, images[0].height) == (42, 14)
    prepared = app.vision.prepare(PLACEHOLDER, images)
    with Image.open(io.BytesIO(data.getvalue())) as encoded:
        expected_rgb = ImageOps.exif_transpose(encoded).convert("RGB")
    normalized = io.BytesIO()
    expected_rgb.save(normalized, format="PNG")
    expected = decode(normalized.getvalue(), CONFIG)
    assert torch.equal(prepared.spans[0][1].patches, expected.patches)


def test_visual_budget_and_expanded_context_are_enforced_before_patching(app, monkeypatch):
    from tensorfold.families.deepseek_v41.cuda import vision

    def unexpected(*args):
        pytest.fail("oversized image request must fail before creating patch tensors")

    monkeypatch.setattr(vision, "_preprocess", unexpected)
    messages = [user_message(png())]
    app.image_limits = ImageLimits(max_visual_tokens=3)
    with pytest.raises(RequestError, match="visual-token budget"):
        app.prepare({"messages": messages, "max_tokens": 8}, True)
    app.image_limits = ImageLimits()
    reference = encode_messages(messages, thinking_mode="chat")
    app.context_window = len(app.tok.encode(reference).ids) + 2  # image adds three positions
    with pytest.raises(ContextLengthError, match="expanded image prompt") as caught:
        app.prepare({"messages": messages, "max_tokens": 8}, True)
    assert caught.value.code == "context_length_exceeded"


def test_image_and_reply_together_must_fit(app):
    messages = [user_message(png())]
    prepared = app.prepare({"messages": messages, "max_tokens": 8}, True)
    app.context_window = len(prepared.prompt) + 7
    with pytest.raises(ContextLengthError, match="8 reply tokens"):
        app.prepare({"messages": messages, "max_tokens": 8}, True)


@pytest.mark.parametrize("prompt", ["no image", PLACEHOLDER * 2])
def test_missing_or_extra_placeholders_are_refused(app, prompt):
    with pytest.raises(ValueError, match="one placeholder per image"):
        app.vision.prepare(prompt, [ImageInput(42, 42, bytes(42 * 42 * 3))])


def test_literal_placeholder_in_user_text_is_refused(app):
    message = user_message(png())
    message["content"][0]["text"] = PLACEHOLDER
    with pytest.raises(RequestError, match="image placeholder"):
        app.prepare({"messages": [message], "max_tokens": 8}, True)


def test_tokenizer_must_encode_the_configured_single_image_token():
    frontend = DsVision(CONFIG)
    with pytest.raises(ValueError, match="server tokenizer"):
        frontend.prepare(PLACEHOLDER, [ImageInput(1, 1, bytes(3))])
    with pytest.raises(ValueError, match="does not match"):
        frontend.bind_tokenizer(NS(encode=lambda *args, **kwargs: NS(ids=[1, 2])))


def test_text_only_and_legacy_continuations_are_unchanged(app):
    messages = [{"role": "user", "content": "hello"}]
    prepared = app.prepare({"messages": messages, "max_tokens": 8}, True)
    assert prepared.vision is None
    assert prepared.prompt == app.tok.encode(encode_messages(messages, thinking_mode="chat")).ids
    legacy_messages, pictures = app.vision.split([user_message(png())])
    assert PLACEHOLDER in legacy_messages[0]["content"]
    prompt = app.vision.expand([10, CONFIG.image_token_id, 11], pictures)
    continued = app.vision.continued(prompt, [*prompt.token_ids, 12])
    assert continued.spans is prompt.spans and continued.token_ids == [*prompt.token_ids, 12]


def test_unlimited_image_count_still_enforces_expanded_context(app):
    app.image_limits = ImageLimits(max_images=0, max_visual_tokens=1048576)
    messages = [user_message(png())] * 65
    prepared = app.prepare({"messages": messages, "max_tokens": 8}, True)
    assert len(prepared.vision.spans) == 65
    app.context_window = len(prepared.prompt) + 7
    with pytest.raises(ContextLengthError, match="8 reply tokens"):
        app.prepare({"messages": messages, "max_tokens": 8}, True)


def test_legacy_split_zero_means_no_count_limit():
    frontend = DsVision(CONFIG, max_images=0)
    _, pictures = frontend.split([user_message(png())] * 65)
    assert len(pictures) == 65
    with pytest.raises(ValueError, match="nonnegative integer"):
        DsVision(CONFIG, max_images=-1)
