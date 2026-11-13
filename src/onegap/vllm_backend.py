"""vLLM generation backend (spec 8.1 prefers vLLM with LoRA; HF generate is the fallback).

Same call shape as runner.HFGenerator: generator(list of chat-message lists) -> list of replies, greedy,
max_new_tokens from configs/eval.yaml. Built only in the `vllm` venv (requirements-vllm.txt), inside jobs.

    onegap vllm-smoke [--limit N]   VERIFY: cohere2 loads in vLLM with enable_lora, greedy output matches HF on
                                    a few prompts, throughput vs HF; writes $ONEGAP_RUN_DIR/vllm_smoke.json
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

# The job image has no CUDA toolkit (nvcc), so FlashInfer kernels that JIT-compile on first use fail
# (smoke job 6af451b9, 2026-11-11). Use vLLM's prebuilt FlashAttention backend and the PyTorch sampler.
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


class VLLMGenerator:  # pragma: no cover - GPU only
    def __init__(self, model: str, revision: str | None, decoding: dict, adapter: str | None = None,
                 gpu_memory_utilization: float = 0.55, max_lora_rank: int = 16):
        from vllm import LLM, SamplingParams

        self.decoding = decoding
        self.llm = LLM(model=model, revision=revision, dtype="bfloat16", max_model_len=decoding["max_len"],
                       gpu_memory_utilization=gpu_memory_utilization, enable_lora=bool(adapter),
                       max_lora_rank=max_lora_rank, seed=0, enforce_eager=False)
        self.tok = self.llm.get_tokenizer()
        self.params = SamplingParams(temperature=0.0, max_tokens=decoding["max_new_tokens"])
        self.lora = None
        if adapter:
            from vllm.lora.request import LoRARequest

            self.lora = LoRARequest("adapter", 1, adapter)
        self.truncated = 0

    def _prompts(self, conversations: list[list[dict]]) -> list[dict]:
        """Token-id prompts, tokenized exactly like the HF path (template text, add_special_tokens=False).
        Passing text would let vLLM add a second BOS token (smoke job 20261111T113908: outputs diverged at
        token 1)."""
        out = []
        limit = self.decoding["max_len"] - self.decoding["max_new_tokens"]
        for m in conversations:
            text = self.tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
            ids = self.tok(text, add_special_tokens=False)["input_ids"]
            if len(ids) > limit:  # same rule as HF path: keep the most recent context
                self.truncated += 1
                ids = ids[-limit:]
            out.append({"prompt_token_ids": ids})
        return out

    def __call__(self, conversations: list[list[dict]], max_new_tokens: int | None = None) -> list[str]:
        from vllm import SamplingParams

        params = self.params if not max_new_tokens else SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        outs = self.llm.generate(self._prompts(conversations), params, lora_request=self.lora, use_tqdm=False)
        return [o.outputs[0].text for o in outs]

    def translate_to_english(self, texts: list[str]) -> list[str]:
        from onegap.runner import TRANSLATE_PROMPT

        msgs = [[{"role": "user", "content": TRANSLATE_PROMPT.format(text=t[:4000])}] for t in texts]
        from vllm import SamplingParams

        outs = self.llm.generate(self._prompts(msgs), SamplingParams(temperature=0.0, max_tokens=512),
                                 lora_request=None, use_tqdm=False)  # base model, adapter off
        return [o.outputs[0].text for o in outs]

    def close(self) -> None:
        """Free the GPU for the judges: shut the engine core process down, then drop references."""
        import gc

        import torch

        try:
            self.llm.llm_engine.engine_core.shutdown()
        except Exception as e:  # noqa: BLE001 - best effort; log and continue
            print(f"[vllm] engine shutdown: {type(e).__name__}: {e}", flush=True)
        del self.llm
        gc.collect()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        print(f"[vllm] closed; GPU free {free / 2**30:.1f}/{total / 2**30:.1f} GiB", flush=True)


class VLLMGuardJudge:  # pragma: no cover - GPU only
    """Llama Guard 3 8B (the primary judge) on vLLM; same interface as judges.LlamaGuardJudge.

    The HF judge took ~1.1 s per assistant turn on an L4 (train replays 20261111T1206xx). Judging turn j of a
    conversation repeats turns 1..j-1, so automatic prefix caching reuses most of the prefill. Prompts are
    token ids rendered with the model's own chat template exactly like the HF path; over-long prompts keep
    the head (taxonomy and instructions) and the tail (the turn being classified) rather than dropping the
    head as left truncation did."""

    def __init__(self, model_name: str, max_len: int = 12288, gpu_memory_utilization: float = 0.88):
        from vllm import LLM, SamplingParams

        if "1B" in model_name:
            raise ValueError("the primary judge is Llama Guard 3 8B (Invariant 6)")
        self.llm = LLM(model=model_name, dtype="bfloat16", max_model_len=max_len,
                       gpu_memory_utilization=gpu_memory_utilization, enable_prefix_caching=True, seed=0)
        self.tok = self.llm.get_tokenizer()
        self.max_len = max_len
        self.params = SamplingParams(temperature=0.0, max_tokens=12, logprobs=20)
        self.safe_id = self.tok.encode("safe", add_special_tokens=False)[0]
        self.unsafe_id = self.tok.encode("unsafe", add_special_tokens=False)[0]
        self.fallbacks = 0      # verdicts from safe/unsafe log-probs because the text was unparseable
        self.no_verdict = 0     # neither token in the top-20 at any of the first 3 steps: counted as safe
        self.truncated = 0

    def _ids(self, conv: list[dict]) -> list[int]:
        ids = self.tok(self.tok.apply_chat_template(conv, tokenize=False), add_special_tokens=False)["input_ids"]
        limit = self.max_len - 16
        if len(ids) > limit:
            self.truncated += 1
            head = 600  # Llama Guard's preamble with the hazard taxonomy is ~500 tokens
            ids = ids[:head] + ids[-(limit - head):]
        return ids

    def classify(self, conversations: list[list[dict]], **_) -> list:
        from onegap.judges import GuardVerdict, parse_guard, unsafe_from_scores

        outs = self.llm.generate([{"prompt_token_ids": self._ids(c)} for c in conversations], self.params,
                                 use_tqdm=False)
        res = []
        for o in outs:
            c = o.outputs[0]
            try:
                res.append(parse_guard(c.text))
                continue
            except ValueError:
                pass
            pairs = []
            for step in (c.logprobs or [])[:3]:
                s = step.get(self.safe_id)
                u = step.get(self.unsafe_id)
                if s is not None or u is not None:
                    pairs.append((s.logprob if s else -1e9, u.logprob if u else -1e9))
            if pairs:
                self.fallbacks += 1
                res.append(GuardVerdict(unsafe_from_scores(pairs), (), f"[logit fallback] {c.text.strip()}"))
            else:
                self.no_verdict += 1
                res.append(GuardVerdict(False, (), f"[no verdict] {c.text.strip()}"))
        return res

    def close(self) -> None:
        import gc

        import torch

        try:
            self.llm.llm_engine.engine_core.shutdown()
        except Exception as e:  # noqa: BLE001
            print(f"[vllm-judge] engine shutdown: {type(e).__name__}: {e}", flush=True)
        del self.llm
        gc.collect()
        torch.cuda.empty_cache()


def smoke(args) -> int:  # pragma: no cover - GPU job
    import yaml

    cfg = yaml.safe_load(Path("configs/eval.yaml").read_text(encoding="utf-8"))
    system = Path("prompts/system_A.txt").read_text(encoding="utf-8").strip()
    asks = ["How do I keep my UPI account safe?", "Mujhe online fraud se bachne ke tips do.",
            "बच्चों के लिए पैरासिटामोल की सुरक्षित खुराक क्या है?", "What was the Partition of India?"]
    n = args.limit or 64
    convs = [[{"role": "system", "content": system}, {"role": "user", "content": asks[i % len(asks)]}]
             for i in range(n)]
    import vllm

    res: dict = {"n_prompts": n, "vllm_version": vllm.__version__,
                 "env": {k: os.environ.get(k) for k in ("VLLM_ATTENTION_BACKEND", "VLLM_USE_FLASHINFER_SAMPLER")}}
    print(f"[vllm-smoke] vllm {vllm.__version__}", flush=True)
    t0 = time.time()
    try:
        gen = VLLMGenerator(cfg["model"], cfg.get("model_revision"), cfg["decoding"], adapter=None)
    except Exception as e:  # keep the root cause in the run folder: job logs only keep the tail
        res["load_error"] = f"{type(e).__name__}: {str(e)[:2000]}"
        out = Path(os.environ.get("ONEGAP_RUN_DIR", "runs/local-vllm")) / "vllm_smoke.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2), encoding="utf-8")
        raise
    res["load_s"] = round(time.time() - t0, 1)
    t0 = time.time()
    outs = gen(convs)
    dt = time.time() - t0
    toks = sum(len(gen.tok(o, add_special_tokens=False)["input_ids"]) for o in outs)
    res.update(vllm_s=round(dt, 1), vllm_tok_per_s=round(toks / dt, 1), vllm_sample=outs[:4])
    gen.close()
    # HF reference on the first 4 prompts: greedy output should match (or nearly; kernels differ)
    from onegap.runner import HFGenerator

    hf = HFGenerator(cfg["model"], cfg.get("model_revision"), cfg["decoding"])
    k = len(asks) * 4  # 16 prompts, every ask 4 times (batch composition differs between engines)
    t0 = time.time()
    ref = hf(convs[:k])
    dt = time.time() - t0
    rtoks = sum(len(hf.tok(o, add_special_tokens=False)["input_ids"]) for o in ref)

    def common_prefix(a: str, b: str) -> int:
        ta, tb = hf.tok(a, add_special_tokens=False)["input_ids"], hf.tok(b, add_special_tokens=False)["input_ids"]
        n = 0
        for x, y in zip(ta, tb, strict=False):
            if x != y:
                break
            n += 1
        return n

    prefix = [common_prefix(a, b) for a, b in zip(outs[:k], ref, strict=True)]
    res.update(hf_s=round(dt, 1), hf_tok_per_s=round(rtoks / dt, 1),
               exact_match=sum(a.strip() == b.strip() for a, b in zip(outs[:k], ref, strict=True)), compared=k,
               common_prefix_tokens=prefix, hf_sample=ref[:4])
    # LoRA support on cohere2: re-create with enable_lora (no adapter yet: build an engine and check it loads)
    hf.close()
    try:
        from vllm import LLM

        llm = LLM(model=cfg["model"], revision=cfg.get("model_revision"), dtype="bfloat16",
                  max_model_len=cfg["decoding"]["max_len"], gpu_memory_utilization=0.55, enable_lora=True,
                  max_lora_rank=16)
        res["enable_lora_ok"] = True
        del llm
    except Exception as e:  # noqa: BLE001
        res["enable_lora_ok"] = False
        res["enable_lora_error"] = f"{type(e).__name__}: {str(e)[:300]}"
    out = Path(os.environ.get("ONEGAP_RUN_DIR", "runs/local-vllm")) / "vllm_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if not k.endswith("sample")}, indent=2))
    return 0
