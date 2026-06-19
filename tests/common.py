"""
tests/common.py — 测试脚本共享工具

prepare_baseline_manifest.py 和 vllm_baseline_runner.py 的共同依赖。
避免 prompt 构造 / tokenizer / 截断逻辑在两处重复维护。
"""

from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
try:
    from transformers import AutoTokenizer

    def load_tokenizer(path: str) -> Any:
        return AutoTokenizer.from_pretrained(
            path, trust_remote_code=True, use_fast=True)
except ImportError:
    AutoTokenizer = None

    def load_tokenizer(path: str) -> Any:
        raise SystemExit("pip install transformers")


def count_tokens(text: str, tok: Any) -> int:
    """返回 text 的 token 数，失败返回 0。"""
    if tok is None:
        return 0
    try:
        return len(tok.encode(text))
    except Exception:
        return 0


def truncate_by_tokens(text: str, max_tokens: int, tok: Any) -> str:
    """按 token 截断文本（保留前 max_tokens 个 token）。"""
    if tok is None or max_tokens is None:
        return text
    try:
        ids = tok.encode(text)
        if len(ids) <= max_tokens:
            return text
        return tok.decode(ids[:max_tokens], skip_special_tokens=True)
    except Exception:
        return text[:max_tokens * 4]


# ---------------------------------------------------------------------------
# 合成文本生成
# ---------------------------------------------------------------------------
def repeated_text(chars: int, topic: str) -> str:
    """生成指定字符数的重复文本（用于构建稳定的长上下文基准）。"""
    chunk = (
        f"{topic} This paragraph is repeated for stable long-context baseline "
        "testing. It is semantically boring but structurally consistent.\n"
    )
    repeat_count = (chars // len(chunk)) + 1
    return "".join([chunk] * repeat_count)[:chars]


# ---------------------------------------------------------------------------
# LongBench prompt 构造
# ---------------------------------------------------------------------------
def prompt_from_longbench(sample: dict, max_chars: int) -> str:
    """从 LongBench 样本构造 prompt。"""
    ctx = sample.get("context", "") or ""
    instr = sample.get("input", "") or ""
    return (
        "You are given a long-context task.\n\n"
        f"Context:\n{ctx}\n\nQuestion:\n{instr}\n\nAnswer:"
    )[:max_chars]


# ---------------------------------------------------------------------------
# NeedleBench prompt 构造
# ---------------------------------------------------------------------------
def needlebench_prompt_from_sample(sample: dict, max_chars: int) -> tuple[str, dict]:
    """从 NeedleBench 样本构造 prompt，返回 (prompt, extra_meta)。"""
    # pick answer
    answer = None
    for k in ["answer", "answers", "target", "label"]:
        v = sample.get(k)
        if v not in (None, ""):
            answer = v if isinstance(v, list) else [v]
            break
    # pick question
    question = None
    for k in ["question", "input", "query", "prompt", "instruction",
              "needle", "retrieval_question"]:
        v = sample.get(k)
        if v not in (None, ""):
            question = v
            break
    # pick context
    context = None
    for k in ["context", "text", "contents", "content", "document",
              "documents", "haystack", "haystack_text", "English", "Chinese"]:
        v = sample.get(k)
        if v not in (None, ""):
            context = v
            break
    if isinstance(context, list):
        context = "\n\n".join(str(x) for x in context)
    if isinstance(question, list):
        question = "\n".join(str(x) for x in question)
    if not context:
        parts = []
        for k, v in sample.items():
            if k in {"answer", "answers", "target", "label"}:
                continue
            if isinstance(v, str) and v.strip():
                parts.append(f"{k}:\n{v}")
        context = "\n\n".join(parts)
    if not question:
        question = "Answer the question based on the long context above."

    prompt = (
        "You are given a long-context retrieval task.\n\n"
        f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    )
    return prompt[:max_chars], {
        "answers": answer or [],
        "fields": sorted(sample.keys()),
    }
