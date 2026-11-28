# RUNTIMES — serving the loop's artifacts across heterogeneous hardware

Status: ANALYSIS/PLAN (2026-11-28 night, farm-paused reading). Facts
cited from repo files; open questions marked. Nothing here is executed
until a parity gate exists (§5).

## 1. What we actually serve today

The loop's artifact is NOT a model file — it is a (base, adapter) pair
served as one callable voice, plus a grammar:

  base    Qwen/Qwen3.5-0.8B          BF16, ~1.6 GB weights
  adapter LoRA r=32 (cognition, kandi)  hot-attached, multiple aliases
          at ONE vLLM process (--lora-modules nargs trap: SINGLE
          occurrence, space-separated — AS-BUILT; two flags silently
          evict all but the last)
  grammar GBNF/structural_tag for guided lanes (R3) — engine-coupled
  runtime vllm/vllm-openai:latest on nano0 (Jetson Orin Nano 8 GiB
          unified, arm64, sm_87), tuned flags (infra
          scripts/serve-qwen3.5-0.8b-tuned.sh):
            --gpu-memory-utilization 0.3  (~2.4 GiB of 8)
            --kv-cache-memory-bytes 650M  --kv-cache-dtype int4_per_token_head
            --max-model-len 131072 (nominal) --max-num-seqs 32
            --language-model-only --cudagraph-capture-sizes 1 2 4
            --reasoning-parser qwen3 --tool-call-parser qwen3_coder
          + image-level GDN patch (qwen_gdn linear-attn kernels) and
          eager mode where graphs OOM the memcg (AS-BUILT §7)
  edge    Pi nodes (arm64, no CUDA): NO LLM today; ONNX = fleet's
          established edge format for SMALL PREDICTORS (D-114:
          "onnx = model format at the edge (deterministic artifact)";
          registry/onnx.yaml + mcp-panoptes/iq onnx mounts, DECISIONS
          1111: onnx stays on pi00 with the fact DB)

So the loop's truth table lives in vLLM. Every runtime question below
is really: which OTHER machines may carry which PART of that truth,
and what certification gate must pass before a persona is served from
anywhere else? (Precedent: D-472 — persona split GPU-orin vs
CPU-pi; the 4B AWQ VL persona runs where sm_87 does, the 0.8B router
wherever there's an engine. That split is already a runtime decision
made once, in the right order: capability first, then hardware.)

## 2. Hardware law of the farm (from infra configs, not vibes)

  machine   silicon               arch law                                  can run
  nano0/1   Orin Nano, sm_87      Ampere-gen; NO fp8, NO fp4; 8 GiB unified  vLLM bf16/int4-kv;
          (CUDA graphs memcg-fragile — eager mode default)                   llama.cpp/GGUF q4;
                                                                             TRT-LLM (native path)
  pi00/01   Raspberry Pi, arm64   CPU-only, 8 GiB                             llama.cpp GGUF (slow);
                                                                             ONNX predictors (today)
  cogs00    RTX PRO 4000          workstation, fp8-capable                     vLLM (dev/CI engine);
          (also the training GPU)                                              anything CUDA
  cloud     RunPod spot (A4000→)  varies by offer; pin --max-price            vLLM for training/eval

Key asymmetry: the ONE place we'd want fp8 (Orin, memory-bound) is
the ONE place the ISA forbids it. The fp8 target the peer just added
to the registry is therefore a workstation/cloud lever, not a nano0
lever — worth stating explicitly before anyone "upgrades" a persona.

## 3. The runtime landscape, scored against THIS workload

Workload properties that decide everything:
  (a) LoRA-at-birth — the loop produces ADAPTERS; a runtime that
      cannot apply our adapter format (PEFT/HF, target_modules with
      the gdn.* naming) forces a MERGE, and a merge is a new model
      with its own corpus-of-one behavior (r1/kandi precedent: merged
      artifacts need re-verification; adapters do not)
  (b) small model (0.8B) — memory/latency per token is not the
      bottleneck; QUALITY margin is. Quantization hurts a 0.8B far
      more than a 40B (less redundancy to trade). Our own
      quantization registry: awq "untested → TRIGGER PRESENT", fp8
      "untested, NO STAGE BUILT", GDN linears "no proven int4
      kernels -> expect PARTIAL-quant products", and the named
      hazard: "quantized-endpoint benches BEFORE register — adapter
      holdout does not certify the product" (#50059). The registry
      already legislated this space; the analysis must not re-litigate
      it, only map it.
  (c) deterministic oracles everywhere — grammar-constrained decode,
      exact NLL holdout, chat-template byte-matters rules. A runtime
      that changes tokenizer/template handling (or its own sampler
      numerics) changes measured BEHAVIOR, not just speed. The
      360→0.8B jinja crutch story (template pinned voice) is the
      local proof.

  runtime        adapter?   oracles?        arm64/orin?     verdict
  -------------- ---------- --------------- --------------- ----------------
  vLLM (today)   YES        grammar, logprobs CUDA only (arm64 wheels     KEEP: measurement ground
                 (attach,  exist; chat tmpl  exist; sm_87 OK              truth; multi-LoRA = the
                  hot-swap) controllable)    graphs fragile -> eager      loop's serving substrate
  SGLang         YES        weaker grammar;  YES-ish (aarch64 wheels     watch, don't adopt: no
                            logprob parity   exist; perf on Jetson        feature we lack in vLLM
                            unproven         UNMEASURED here)             except speed we haven't needed
  llama.cpp/GGUF NO-attach  grammars EXIST   YES (CPU; NEON;  THE EDGE LANE: 0.8B q4
                 (merge     (GBNF port      decent on Pi 5;  q8 ≈ 0.5–0.9 GB
                 first);    already used    Jetson faster    RAM — pi/nano1 offline
                 LoRA->    by our own       with CUDA       persona+router; adapters
                 merge)    gbnf); NLL via   build)          MUST be merged first
                 --logprobs; CHAT-TEMPLATE
                 REWRITES TOKENIZATION-ADJACENT behavior —
                 D-479 law applies verbatim: never compare v1-trained
                 artifacts across a format change without re-cert.
  TensorRT-LLM   YES        no HF-grammar    YES (Jetson   NOT NOW: engine = build
                 (per-shape equivalence     is its home    step per artifact; sm_87
                  engines)  unproven        platform       is its weakest NVIDIA SKU;
                            for our tags)                 our throughput need is
                                                         zero at 32 seqs eager
  ONNX RT        merge+     logprobs: yes   YES           WRONG CLASS for LLMs at
                 trace;     but trace bakes               0.8B: no paged KV, CPU
                 STATIC     shapes/context                decode of 0.8B is fine BUT
                 graph                                   grammar-constrained decode
                                                         effectively unsupported —
                                                         kills the oracle lane
  torchtitan /   n/a        n/a              n/a         training-side only, ignore
  TF exports                                      for serving
  Triton server  wraps      inherits         CUDA         irrelevant until N runtimes
  (KServe)       runtimes   themselves                   must coexist — one gateway
                                                        we already have (ai-router)

## 4. The strategic frame: the loop is runtime-agnostic, the ORACLES
are not.

Every asset the loop produces survives a runtime change IF the
certification travels with it:
  adapters    -> merge tool exists in the lane (r6 merge algebra was
                 specced exactly for this); a merged GGUF is just
                 another product of the same corpus
  grammars    -> gbnf is a runtime-neutral grammar format (ours is
                 already compiled for xgrammar/vLLM; llama.cpp reads
                 the same GBNF — same grammar file, two engines; the
                 parity check is free)
  holdout/NLL -> engine-reported logprobs let the SAME split certify
                 on any runtime — but only cross-runtime if the
                 tokenizer chat-template path is byte-identical
                 (D-479: format changes need re-cert, never re-use)
  bench v2    -> runtime-neutral HTTP; works unchanged against a
                 llama.cpp OpenAI shim (its server mode speaks
                 /v1/chat/completions — preflight + lint still fire)

So "which runtime" is not an infrastructure preference — it is
WHICH CERTIFICATION REGIME a product ships under. A GGUF persona on
the Pi is the same corpus + a different measurement; the honest gate
is to treat every runtime change as a model change: re-run bench v2
attribution against the vLLM reference. The registry line "quantized
endpoint benches before register" is this law in one place; generalize
it: **any endpoint, any format — bench-before-register.**

## 5. What I'd actually do (farm-free first, in order)

A. (FREE, now) runtime fingerprint in the ledger + bench report:
   record engine id, version, quantization, template hash per
   measurement row (orchestrator side: eval rows gain engine: and
   quant: columns; qtrain bench already reads /v1/models — capture
   it at run time). Without the fingerprint, cross-runtime numbers
   silently mix — the one failure mode this doc exists to prevent.
B. (FREE, now) GGUF parity harness, vLLM-side zero risk: a script
   that (1) merges cognition-r3 (tool exists), (2) exports gguf q4/q8
   via convert (llama.cpp is CPU-side — runs on the workstation or a
   Pi), (3) serves llama-server --port locally, (4) runs bench v2 +
   holdout NLL vs the vLLM reference with a PASS/FAIL delta budget
   (behavioral: flips<=0 net; NLL: +<=0.15 — thresholds pre-
   registered, then tuned on the r3 reference). Numbers-only until it
   has run; the harness is falsifiable before any GPU hour.
C. (FREE) write the registry-side lesson: PARTIAL-quant hazard for
   GDN models belongs in REGIMES' quantization law (cognizes only
   the dense half of the model → the "cognition" behavior may live
   precisely in the un-quantizable linear-attention part; a q4 that
   passes perplexity can fail behavioral benches). Predict, don't
   discover.
D. (LATER, farm) the real use-cases worth a runtime slot:
   1. nano1 offline persona (GGUF merged, grammar-guided, NO
   cluster) — a genuinely air-gapped product;
   2. Pi-side fallback router (0.8B q4 GGUF CPU, persona-only,
   degraded cognition = honest, labeled fallback) — the loop's
   "capability tiers" doc wants this tier named, not improvised;
   3. workstation CI engine (vLLM fp8 IF the model ever fp8-works —
   sm_87 law says NOT on nano0; RTX-class only).
   Each is a lane-assignment decision (D-472 precedent), not an
   engine war.
E. (AVOID) TensorRT-LLM until a concrete Orin throughput requirement
   exists; AVOID SGLang/ONNX for LLMs — no capability we lack, and
   ONNX's static-shape story fights our grammar lane directly.

## 6. Open questions (answerable locally)

1. llama.cpp Qwen3.5 GDN support: RESOLVED in the affirmative by the
   fleet registry itself ("llama.cpp implements hybrid GDN on CPU;
   single-file artifact fits the OCI/registry law" — gguf: deferred,
   waiting a CONSUMER not a capability). Verify the convert script
   name for the qwen3_5 arch when the harness is written.
2. Does llama.cpp honor --logprobs with the same tokenizer byte
   semantics as vLLM's? (Their OpenAI shim does; the NLL parity gate
   depends on this. Local experiment, CPU.)
3. Grammar parity: our gbnf compiled by xgrammar vs llama.cpp's
   gbnf engine — same grammar files, same accepted set? (Free to
   test; a mismatch is a certification finding, not a bug.)
4. Does merging r3 (adapter trained with zero-B injection — the
   attached-but-not-executed lesson) survive merge? The wiring guard
   (postproc) proves adapter effect at RUNTIME; merge must prove it
   again at a NEW runtime: merge != attach, and this repo already
   knows the difference (r1 era).

## 7. Verdict

vLLM stays the measurement ground truth AND the cluster engine —
nothing beats it while LoRA hot-attach + grammar + logprobs are the
loop's requirements. GGUF/llama.cpp is the only runtime that earns a
second lane here (arm64 everywhere, our gbnf portable, Pi/nano1
offline = real product), but ONLY under the bench-before-register law
the peer's quantization registry already states. ONNX stays what the
fleet decided it is — the edge predictor format (D-114), not an LLM
serving play. TRT-LLM/sglang: parked until a throughput ticket
exists. The immediate, free, high-value work is the runtime
FINGERPRINT in the ledger — until every number knows the engine it
came from, cross-runtime progress is a rumor.
