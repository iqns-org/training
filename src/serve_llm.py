import argparse
import socket
from pathlib import Path

import torch
import uvicorn
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"


def find_free_port(host: str, start_port: int, max_tries: int = 20) -> int:
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
                return port
            except OSError:
                continue

    raise RuntimeError(
        f"No free port found starting at {start_port}"
    )


def discover_base_model(domain: str) -> str:
    root = Path(__file__).resolve().parents[1]
    yaml_candidates = sorted(
        (root / "axolotl" / domain).glob("*.yaml")
    )

    for yaml_path in yaml_candidates:
        text = yaml_path.read_text(encoding="utf-8")

        for line in text.splitlines():
            if line.strip().startswith("base_model:"):
                return line.split(":", 1)[1].strip().strip("\"'")

    return DEFAULT_BASE_MODEL


def normalize_chat_messages(history: list[dict[str, str]] | None) -> list[dict[str, str]]:
    """Accept both browser and server-style message payloads."""
    normalized: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for item in history or []:
        if not isinstance(item, dict):
            continue

        role = str(item.get("role", item.get("sender", ""))).strip().lower()
        if role == "model":
            role = "assistant"

        content = str(item.get("content", item.get("text", ""))).strip()
        if role not in {"user", "assistant"} or not content:
            continue

        key = (role, content)
        if key not in seen:
            normalized.append({"role": role, "content": content})
            seen.add(key)

    return normalized


def build_chat_messages(
    history: list[dict[str, str]] | None = None,
    prompt: str | None = None,
) -> list[dict[str, str]]:
    """Return a valid chat sequence while preserving prior conversation state."""
    messages = normalize_chat_messages(history)

    if prompt is not None:
        normalized_prompt = str(prompt).strip()
        if normalized_prompt:
            last_user = messages[-1] if messages and messages[-1]["role"] == "user" else None
            if last_user is None or last_user["content"] != normalized_prompt:
                messages.append({"role": "user", "content": normalized_prompt})

    return messages


def format_chatml(messages: list[dict[str, str]]) -> str:
    """
    Matches Axolotl's tokenizer_default_fallback_chatml format.
    """
    parts = []

    for message in messages:
        parts.append(
            f"<|im_start|>{message['role']}\n"
            f"{message['content']}"
            f"<|im_end|>\n"
        )

    # Generation prompt: assistant turn is opened but not closed.
    parts.append("<|im_start|>assistant\n")

    return "".join(parts)


def make_app(base_model: str, adapter_dir: str, use_lora: bool = True):
    print(f"Loading base model: {base_model}")
    print(f"Loading adapter: {adapter_dir}")
    print(f"LoRA enabled: {use_lora}")

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"chat_template: {tokenizer.chat_template!r}")
    print(f"bos_token: {tokenizer.bos_token!r} ({tokenizer.bos_token_id})")
    print(f"eos_token: {tokenizer.eos_token!r} ({tokenizer.eos_token_id})")
    print(f"pad_token: {tokenizer.pad_token!r} ({tokenizer.pad_token_id})")

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------

    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    print(f"device: {device}")
    print(f"dtype: {dtype}")

    # ------------------------------------------------------------------
    # Base model
    # ------------------------------------------------------------------

    base_model_obj = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
    )

    base_model_obj.to(device)

    # ------------------------------------------------------------------
    # LoRA adapter
    # ------------------------------------------------------------------

    if use_lora:
        model = PeftModel.from_pretrained(
            base_model_obj,
            adapter_dir,
        )
        model.to(device)
        model.eval()
        print("LoRA model ready.")
    else:
        model = base_model_obj
        model.to(device)
        model.eval()
        print("Base model ready without LoRA.")

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    app = FastAPI(
        title=f"{adapter_dir} inference"
    )

    class GenerateRequest(BaseModel):
        prompt: str | None = None
        messages: list[dict[str, Any]] = Field(default_factory=list)
        max_new_tokens: int = 512
        temperature: float = 0.0
        top_p: float = 0.9
        repetition_penalty: float = 1.1
        do_sample: bool = True

        @property
        def normalized_messages(self) -> list[dict[str, str]]:
            return normalize_chat_messages(self.messages)

    @app.get("/")
    def root():
        root_path = Path(__file__).resolve().parents[1]
        index_file = root_path / "src" / "index.html"

        if index_file.exists():
            return HTMLResponse(
                index_file.read_text(encoding="utf-8")
            )

        return HTMLResponse(
            "<html><body></body></html>"
        )

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "base_model": base_model,
            "adapter_dir": adapter_dir,
            "device": str(device),
        }

    @app.get("/favicon.ico")
    def favicon():
        return Response(status_code=204)

    @app.post("/generate")
    def generate(req: GenerateRequest):
        history = req.normalized_messages
        prompt = req.prompt.strip() if req.prompt else ""
        messages = build_chat_messages(history=history, prompt=prompt)

        if not messages:
            raise HTTPException(status_code=400, detail="No prompt or conversation history provided.")

        # --------------------------------------------------------------
        # Chat formatting
        #
        # SmolLM2 tokenizer currently has no chat_template, so this
        # explicitly matches Axolotl's:
        #
        #   chat_template: tokenizer_default_fallback_chatml
        # --------------------------------------------------------------

        prompt_text = format_chatml(messages)

        print(f"\nPROMPT: {messages[-1]['content']!r}")
        print(f"TEMPLATE: {prompt_text!r}")

        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        input_length = inputs["input_ids"].shape[1]

        print(f"INPUT TOKENS: {input_length}")

        # --------------------------------------------------------------
        # Generate
        # --------------------------------------------------------------
        CHAT_END_TOKEN = "<|im_end|>"
        CHAT_END_TOKEN_ID = tokenizer.convert_tokens_to_ids(CHAT_END_TOKEN)

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=req.max_new_tokens,
                do_sample=req.do_sample,
                temperature=req.temperature,
                top_p=req.top_p,
                repetition_penalty=req.repetition_penalty,
                eos_token_id=CHAT_END_TOKEN_ID,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Only decode newly generated tokens.
        new_tokens = generated[0, input_length:]

        text = tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()

        print(f"OUTPUT: {text!r}")

        return {
            "text": text,
            "prompt_tokens": input_length,
        }

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Serve a trained Axolotl adapter locally."
    )

    parser.add_argument(
        "domain",
        nargs="?",
        default="queer",
    )

    parser.add_argument(
        "--host",
        default="0.0.0.0",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
    )

    parser.add_argument(
        "--adapter-dir",
        default=None,
    )

    parser.add_argument(
        "--base-model",
        default=None,
    )

    parser.add_argument(
        "--base",
        action="store_true",
        help="Serve the base model without applying the LoRA adapter.",
    )

    parser.add_argument(
        "--lora",
        action="store_true",
        help="Serve the LoRA adapter on top of the base model (default).",
    )

    args = parser.parse_args()

    if args.base and args.lora:
        parser.error("Choose either --base or --lora, not both.")

    # ------------------------------------------------------------------
    # Port
    # ------------------------------------------------------------------

    port = args.port

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((args.host, port))

    except OSError:
        port = find_free_port(
            args.host,
            args.port,
        )

        print(
            f"Port {args.port} is already in use; "
            f"using port {port} instead."
        )

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    root = Path(__file__).resolve().parents[1]

    domain = args.domain

    adapter_dir = (
        args.adapter_dir
        or str(root / "outputs" / domain)
    )

    base_model = (
        args.base_model
        or discover_base_model(domain)
    )

    # ------------------------------------------------------------------
    # Server
    # ------------------------------------------------------------------

    use_lora = not args.base
    if args.lora:
        use_lora = True

    app = make_app(
        base_model=base_model,
        adapter_dir=adapter_dir,
        use_lora=use_lora,
    )

    uvicorn.run(
        app,
        host=args.host,
        port=port,
    )
