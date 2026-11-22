import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

fake_torch = types.ModuleType("torch")
fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
fake_torch.float16 = "float16"
fake_torch.float32 = "float32"
fake_torch.device = lambda value: value
fake_torch.inference_mode = lambda: types.SimpleNamespace(__enter__=lambda self: None, __exit__=lambda self, exc_type, exc, tb: None)

fake_peft = types.ModuleType("peft")
fake_peft.PeftModel = object

fake_transformers = types.ModuleType("transformers")
fake_transformers.AutoModelForCausalLM = object
fake_transformers.AutoTokenizer = object

fake_fastapi = types.ModuleType("fastapi")
fake_fastapi.FastAPI = object
fake_fastapi.Response = object
fake_fastapi.HTTPException = type("HTTPException", (Exception,), {})
fake_fastapi.HTMLResponse = str

fake_fastapi_responses = types.ModuleType("fastapi.responses")
fake_fastapi_responses.HTMLResponse = str

fake_pydantic = types.ModuleType("pydantic")
fake_pydantic.BaseModel = object
fake_pydantic.Field = lambda default_factory=None, **kwargs: default_factory

fake_uvicorn = types.ModuleType("uvicorn")
fake_uvicorn.run = lambda *args, **kwargs: None

sys.modules.setdefault("torch", fake_torch)
sys.modules.setdefault("peft", fake_peft)
sys.modules.setdefault("transformers", fake_transformers)
sys.modules.setdefault("fastapi", fake_fastapi)
sys.modules.setdefault("fastapi.responses", fake_fastapi_responses)
sys.modules.setdefault("pydantic", fake_pydantic)
sys.modules.setdefault("uvicorn", fake_uvicorn)

from src.serve_llm import build_chat_messages, format_chatml, normalize_chat_messages


def test_build_chat_messages_keeps_history_and_appends_new_prompt():
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]

    messages = build_chat_messages(history=history, prompt="what now")

    assert messages == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "what now"},
    ]

    rendered = format_chatml(messages)
    assert "<|im_start|>user\nhi<|im_end|>\n" in rendered
    assert "<|im_start|>assistant\nhello<|im_end|>\n" in rendered
    assert rendered.endswith("<|im_start|>assistant\n")


def test_normalize_chat_messages_accepts_browser_payload_shape():
    history = [
        {"id": "m-1", "role": "user", "text": "hi"},
        {"id": "m-2", "role": "model", "text": "hello"},
    ]

    normalized = normalize_chat_messages(history)

    assert normalized == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
