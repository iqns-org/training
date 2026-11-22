#!/usr/bin/env python3

"""
SmolLM2 + LoRA diagnostic / evaluation script.

Purpose:
  1. Inspect tokenizer special tokens.
  2. Inspect the adapter/tokenizer configuration.
  3. Verify ChatML token IDs.
  4. Test the model with several controlled prompts.
  5. Compare generation using:
       A. <|im_end|> as EOS
       B. <|endoftext|> as EOS
  6. Detect generation that crosses conversational boundaries.
  7. Show raw generated tokens and decoded output.

Usage:

    python diagnose_model.py

Optional:

    BASE_MODEL=HuggingFaceTB/SmolLM2-360M-Instruct \
    ADAPTER=/developer/training-data/outputs/queer \
    python diagnose_model.py
"""

import os
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_MODEL = os.getenv(
    "BASE_MODEL",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
)

ADAPTER = os.getenv(
    "ADAPTER",
    "/developer/training-data/outputs/queer",
)

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "128"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title: str):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def token_info(tokenizer, token: str):
    token_id = tokenizer.convert_tokens_to_ids(token)

    if token_id == tokenizer.unk_token_id and token != tokenizer.unk_token:
        return {
            "token": token,
            "id": token_id,
            "decoded": None,
            "exists": False,
        }

    decoded = tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
    )

    return {
        "token": token,
        "id": token_id,
        "decoded": decoded,
        "exists": True,
    }


def format_chatml(messages):
    parts = []

    for message in messages:
        parts.append(
            f"<|im_start|>{message['role']}\n"
            f"{message['content']}"
            f"<|im_end|>\n"
        )

    parts.append("<|im_start|>assistant\n")

    return "".join(parts)


def show_tokens(tokenizer, ids, label="TOKENS"):
    print()
    print(f"--- {label} ---")

    for i, token_id in enumerate(ids):
        token = tokenizer.convert_ids_to_tokens(token_id)
        decoded = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
        )

        print(
            f"{i:4d}  "
            f"id={token_id:5d}  "
            f"token={token!r:30}  "
            f"decoded={decoded!r}"
        )


def generate_test(
    model,
    tokenizer,
    prompt,
    eos_token_id,
    label,
):
    section(f"GENERATION TEST: {label}")

    print("Prompt:")
    print(repr(prompt))

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )

    inputs = {
        k: v.to(model.device)
        for k, v in inputs.items()
    }

    input_len = inputs["input_ids"].shape[1]

    print()
    print("Input token count:", input_len)

    show_tokens(
        tokenizer,
        inputs["input_ids"][0].tolist(),
        "INPUT TOKENS",
    )

    print()
    print("Generation EOS token:", eos_token_id)
    print(
        "EOS token:",
        tokenizer.convert_ids_to_tokens(eos_token_id),
    )

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=False,
        )

    all_ids = output.sequences[0].tolist()
    generated_ids = all_ids[input_len:]

    print()
    print("Generated token count:", len(generated_ids))

    show_tokens(
        tokenizer,
        generated_ids,
        "GENERATED TOKENS",
    )

    decoded = tokenizer.decode(
        generated_ids,
        skip_special_tokens=False,
    )

    decoded_clean = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    print()
    print("--- RAW DECODE ---")
    print(repr(decoded))

    print()
    print("--- CLEAN DECODE ---")
    print(repr(decoded_clean))

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    endoftext_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    print()
    print("--- BOUNDARY ANALYSIS ---")

    print(
        "<|im_end|> occurrences:",
        generated_ids.count(im_end_id),
    )

    print(
        "<|endoftext|> occurrences:",
        generated_ids.count(endoftext_id),
    )

    if generated_ids:
        first_token = generated_ids[0]
        last_token = generated_ids[-1]

        print(
            "First generated token:",
            first_token,
            tokenizer.convert_ids_to_tokens(first_token),
        )

        print(
            "Last generated token:",
            last_token,
            tokenizer.convert_ids_to_tokens(last_token),
        )

        if last_token == im_end_id:
            print("RESULT: generation terminated on <|im_end|>")
        elif last_token == endoftext_id:
            print("RESULT: generation terminated on <|endoftext|>")
        else:
            print(
                "RESULT: generation did NOT terminate on a known "
                "conversation boundary"
            )

    return {
        "label": label,
        "prompt": prompt,
        "generated_ids": generated_ids,
        "decoded": decoded,
        "decoded_clean": decoded_clean,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    section("ENVIRONMENT")

    print("Python:", os.sys.version)
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))

    print("Base model:", BASE_MODEL)
    print("Adapter:", ADAPTER)
    print("Device:", DEVICE)


    # -----------------------------------------------------------------------
    # Adapter inspection
    # -----------------------------------------------------------------------

    section("ADAPTER DIRECTORY")

    adapter_path = Path(ADAPTER)

    print("Exists:", adapter_path.exists())

    if adapter_path.exists():
        for filename in [
            "adapter_config.json",
            "adapter_model.safetensors",
            "tokenizer_config.json",
            "tokenizer.json",
            "special_tokens_map.json",
        ]:
            path = adapter_path / filename

            print(
                f"{filename:30} "
                f"{'YES' if path.exists() else 'NO'}"
            )

        adapter_config = adapter_path / "adapter_config.json"

        if adapter_config.exists():
            print()
            print("--- adapter_config.json ---")

            try:
                print(
                    json.dumps(
                        json.loads(
                            adapter_config.read_text()
                        ),
                        indent=2,
                    )
                )
            except Exception as exc:
                print("Could not read adapter config:", exc)


    # -----------------------------------------------------------------------
    # Tokenizer
    # -----------------------------------------------------------------------

    section("TOKENIZER")

    adapter_tokenizer_config = adapter_path / "tokenizer_config.json"

    if adapter_tokenizer_config.exists():
        tokenizer_source = ADAPTER
        print(
            "Tokenizer source: ADAPTER "
            "(tokenizer_config.json exists)"
        )
    else:
        tokenizer_source = BASE_MODEL
        print(
            "Tokenizer source: BASE MODEL "
            "(adapter has no tokenizer_config.json)"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=True,
    )

    print()
    print("Tokenizer class:", type(tokenizer).__name__)
    print("Tokenizer name/source:", tokenizer_source)

    print()
    print("bos_token:", repr(tokenizer.bos_token))
    print("bos_token_id:", tokenizer.bos_token_id)

    print("eos_token:", repr(tokenizer.eos_token))
    print("eos_token_id:", tokenizer.eos_token_id)

    print("unk_token:", repr(tokenizer.unk_token))
    print("unk_token_id:", tokenizer.unk_token_id)

    print("pad_token:", repr(tokenizer.pad_token))
    print("pad_token_id:", tokenizer.pad_token_id)

    print()
    print("chat_template exists:", tokenizer.chat_template is not None)

    if tokenizer.chat_template:
        print()
        print("--- CHAT TEMPLATE ---")
        print(tokenizer.chat_template)

    print()
    print("--- SPECIAL TOKENS ---")

    print(
        json.dumps(
            tokenizer.special_tokens_map,
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print("--- ALL SPECIAL TOKENS ---")

    for token, token_id in zip(
        tokenizer.all_special_tokens,
        tokenizer.all_special_ids,
    ):
        print(
            f"id={token_id:4d}  token={token!r}"
        )


    # -----------------------------------------------------------------------
    # Critical token inspection
    # -----------------------------------------------------------------------

    section("CHATML TOKEN INSPECTION")

    tokens_to_check = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
    ]

    for token in tokens_to_check:
        print(token_info(tokenizer, token))


    im_start_id = tokenizer.convert_tokens_to_ids(
        "<|im_start|>"
    )

    im_end_id = tokenizer.convert_tokens_to_ids(
        "<|im_end|>"
    )

    endoftext_id = tokenizer.convert_tokens_to_ids(
        "<|endoftext|>"
    )

    print()
    print("IMPORTANT:")
    print("  tokenizer.eos_token_id =", tokenizer.eos_token_id)
    print("  <|im_end|>             =", im_end_id)
    print("  <|endoftext|>          =", endoftext_id)

    if tokenizer.eos_token_id == im_end_id:
        print()
        print("GOOD: tokenizer EOS is <|im_end|>")

    else:
        print()
        print(
            "WARNING: tokenizer EOS is NOT <|im_end|>."
        )

        print(
            "For ChatML SFT, conversational stopping should "
            "normally be tested with <|im_end|>."
        )


    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------

    section("LOAD MODEL")

    print("Loading base model...")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    print("Base model loaded.")

    print()
    print("Loading LoRA adapter...")

    model = PeftModel.from_pretrained(
        model,
        ADAPTER,
    )

    print("LoRA adapter loaded.")

    model.eval()

    if not torch.cuda.is_available():
        model.to(DEVICE)


    # -----------------------------------------------------------------------
    # Test prompt construction
    # -----------------------------------------------------------------------

    section("CHATML PROMPT")

    messages = [
        {
            "role": "user",
            "content": "hi",
        }
    ]

    prompt = format_chatml(messages)

    print(repr(prompt))

    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0].tolist()

    show_tokens(
        tokenizer,
        prompt_ids,
        "CHATML PROMPT TOKENS",
    )


    # -----------------------------------------------------------------------
    # Test 1: Correct ChatML EOS
    # -----------------------------------------------------------------------

    result_im_end = generate_test(
        model,
        tokenizer,
        prompt,
        eos_token_id=im_end_id,
        label="<|im_end|> EOS",
    )


    # -----------------------------------------------------------------------
    # Test 2: Original EOS
    # -----------------------------------------------------------------------

    result_endoftext = generate_test(
        model,
        tokenizer,
        prompt,
        eos_token_id=endoftext_id,
        label="<|endoftext|> EOS",
    )


    # -----------------------------------------------------------------------
    # Additional controlled prompts
    # -----------------------------------------------------------------------

    prompts = [
        "hi",
        "hello",
        "how are you?",
        "what do you want?",
        "tell me something.",
        "good morning.",
    ]

    section("CONTROLLED PROMPTS")

    results = []

    for text in prompts:

        prompt = format_chatml(
            [
                {
                    "role": "user",
                    "content": text,
                }
            ]
        )

        result = generate_test(
            model,
            tokenizer,
            prompt,
            eos_token_id=im_end_id,
            label=f"CONTROL: {text}",
        )

        results.append(result)


    # -----------------------------------------------------------------------
    # Multi-turn test
    # -----------------------------------------------------------------------

    section("MULTI-TURN TEST")

    messages = [
        {
            "role": "user",
            "content": "hi",
        },
        {
            "role": "assistant",
            "content": "hello.",
        },
        {
            "role": "user",
            "content": "how are you?",
        },
    ]

    prompt = format_chatml(messages)

    print(repr(prompt))

    generate_test(
        model,
        tokenizer,
        prompt,
        eos_token_id=im_end_id,
        label="MULTI-TURN",
    )


    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    section("DIAGNOSTIC SUMMARY")

    print("Base model:", BASE_MODEL)
    print("Adapter:", ADAPTER)
    print()

    print(
        "tokenizer.eos_token_id:",
        tokenizer.eos_token_id,
    )

    print(
        "<|im_start|>:",
        im_start_id,
    )

    print(
        "<|im_end|>:",
        im_end_id,
    )

    print(
        "<|endoftext|>:",
        endoftext_id,
    )

    print()
    print("Interpretation:")

    if tokenizer.chat_template is None:
        print(
            "1. Tokenizer has NO native chat_template."
        )
        print(
            "   Manual ChatML formatting is therefore being used."
        )
    else:
        print(
            "1. Tokenizer HAS a chat_template."
        )

    if tokenizer.eos_token_id != im_end_id:
        print(
            "2. tokenizer.eos_token_id != <|im_end|>."
        )
        print(
            "   The diagnostic explicitly tests <|im_end|> as EOS."
        )
    else:
        print(
            "2. tokenizer.eos_token_id == <|im_end|>."
        )

    print()
    print(
        "Most important evidence to send back:"
    )
    print(
        "  - TOKENIZER section"
    )
    print(
        "  - CHATML TOKEN INSPECTION"
    )
    print(
        "  - CHATML PROMPT TOKENS"
    )
    print(
        "  - both generation tests"
    )
    print(
        "  - CONTROLLED PROMPTS"
    )
    print(
        "  - MULTI-TURN TEST"
    )


if __name__ == "__main__":
    main()

