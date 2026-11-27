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
              "quantize_onnx.py", "holdout_loss.py"):
        src = (Path(__file__).resolve().parents[1] / "postproc" /
               f).read_text()
        compile(src, f, "exec")


def test_adapter_namespace_reads_safetensors_header(tmp_path):
    """The load-shape choice comes from the adapter's OWN tensor names
    (r3's 1.4059==1.4059: env-flag guessing attached 372 tensors to a
    subtree the forward never routed through). Pure function — exec
    the slice, fabricate headers, no torch."""
    import re
    import struct
    src = (Path(__file__).resolve().parents[1] / "postproc" /
           "holdout_loss.py").read_text()
    m = re.search(r"def adapter_namespace\(.*?\n(?=\n\S)", src, re.S)
    assert m, "adapter_namespace gone/refactored — update this test"
    ns = {"struct": struct, "json": json, "Path": Path}
    exec(m.group(0), ns)
    fn = ns["adapter_namespace"]

    def mk(d, keys):
        d.mkdir(parents=True, exist_ok=True)
        hdr = json.dumps({k: {"dtype": "BF16", "shape": [1],
                              "data_offsets": [0, 2]} for k in keys})\
            .encode()
        (d / "adapter_model.safetensors").write_bytes(
            struct.pack("<Q", len(hdr)) + hdr + b"\x00\x00")

    lm = tmp_path / "lm"
    mk(lm, {"base_model.model.model.language_model.layers.0.mlp"
            ".up_proj.lora_A.default.weight": 0})
    pl = tmp_path / "pl"
    mk(pl, {"base_model.model.model.layers.0.mlp.up_proj"
            ".lora_A.default.weight": 0})
    assert fn(lm) == "language_model"
    assert fn(pl) == "plain"
    assert fn(tmp_path / "missing") == "plain"      # merged dir: plain


def test_remap_adapter_dir_toggles_namespace(tmp_path):
    """The r3 costume self-heal: keys toggle, BYTES and offsets don't
    move — a remapped file must round-trip back to the original."""
    import re
    import shutil
    import struct
    src = (Path(__file__).resolve().parents[1] / "postproc" /
           "holdout_loss.py").read_text()
    m = re.search(r"def remap_adapter_dir\(.*?\n(?=\ndef )", src, re.S)
    assert m, "remap_adapter_dir gone/refactored — update this test"
    ns = {"struct": struct, "json": json, "Path": Path, "shutil": shutil}
    exec(m.group(0), ns)
    md = tmp_path / "ad"
    md.mkdir()
    keys = {"base_model.model.model.language_model.layers.0.mlp"
            ".up_proj.lora_A.default.weight": {"dtype": "BF16",
                                               "shape": [1],
                                               "data_offsets": [0, 2]},
            "__metadata__": {"format": "pt"}}
    hdr = json.dumps(keys).encode()
    (md / "adapter_model.safetensors").write_bytes(
        struct.pack("<Q", len(hdr)) + hdr + b"\x00\x00")
    (md / "adapter_config.json").write_text('{"r": 32}')
    out = tmp_path / "alt"
    assert ns["remap_adapter_dir"](md, out) is True
    with (out / "adapter_model.safetensors").open("rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        new = json.loads(fh.read(n))
        body = fh.read()
    nk = [k for k in new if k != "__metadata__"]
    assert len(nk) == 1 and "language_model" not in nk[0]
    assert "model.layers.0.mlp" in nk[0]
    assert body == b"\x00\x00"                       # values untouched
    assert (out / "adapter_config.json").is_file()
    # toggle back == original keys
    back = tmp_path / "back"
    assert ns["remap_adapter_dir"](out, back) is True
    with (back / "adapter_model.safetensors").open("rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        rk = json.loads(fh.read(n))
    assert "base_model.model.model.language_model.layers.0.mlp" \
        ".up_proj.lora_A.default.weight" in [k for k in rk
                                             if k != "__metadata__"]
