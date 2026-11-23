import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "postproc"))
from util import sanitize_tokenizer_config  # noqa: E402


def test_sanitize_list_to_dict(tmp_path):
    p = tmp_path / "tokenizer_config.json"
    p.write_text(json.dumps({
        "model_max_length": 131072,
        "extra_special_tokens": ["<|im_start|>", "<|im_end|>"],
    }))
    sanitize_tokenizer_config(tmp_path)
    cfg = json.loads(p.read_text())
    assert cfg["extra_special_tokens"] == {}
    assert cfg["model_max_length"] == 131072


def test_sanitize_idempotent_and_absent(tmp_path):
    p = tmp_path / "tokenizer_config.json"
    cfg = {"extra_special_tokens": {"foo": "<|x|>"}}   # dict form kept
    p.write_text(json.dumps(cfg))
    sanitize_tokenizer_config(tmp_path)
    assert json.loads(p.read_text()) == cfg
    sanitize_tokenizer_config(tmp_path / "nope")       # absent dir: no-op


def test_export_scripts_importable_shape():
    """Scripts are argparse-only at import (torch imports happen inside
    main), so CI lint-catches syntax errors without heavy deps."""
    for f in ("merge_adapter.py", "export_onnx.py", "quantize_awq.py",
              "quantize_onnx.py"):
        src = (Path(__file__).resolve().parents[1] / "postproc" /
               f).read_text()
        compile(src, f, "exec")
