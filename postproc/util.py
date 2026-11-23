"""Shared postproc helpers."""


def sanitize_tokenizer_config(model_dir) -> None:
    """Normalize tokenizer_config.json for modern transformers.

    Older transformers serialize ``extra_special_tokens`` as a LIST of
    token strings; >= ~4.55 requires a dict and crashes on load
    ('list' object has no attribute 'keys'). The tokens themselves live
    in tokenizer.json's added_tokens regardless, so reducing the list to
    an empty dict is lossless for encode/decode AND fixes loadability on
    the serving side (vLLM/ORT). Idempotent; no-op when clean.
    """
    import json
    from pathlib import Path

    p = Path(model_dir) / "tokenizer_config.json"
    if not p.is_file():
        return
    cfg = json.loads(p.read_text())
    est = cfg.get("extra_special_tokens")
    if isinstance(est, list):
        cfg["extra_special_tokens"] = {}
        p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
