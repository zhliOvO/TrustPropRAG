from __future__ import annotations

from typing import Callable, Dict, List, Optional, Protocol


class LLMClient(Protocol):
    def __call__(self, messages: List[Dict[str, str]], **kwargs) -> str: ...


def format_baseline_context(contexts: List[str], max_chars: int = 500) -> str:
    parts = [f"[Document {i + 1}]: {ctx[:max_chars]}" for i, ctx in enumerate(contexts)]
    return " ".join(parts)


def build_baseline_messages(query: str, context_block: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "user",
            "content": (
                "Answer the following question based on the provided documents. "
                "Give a short, direct answer.\n"
                f"Documents: {context_block}\n"
                f"Question: {query}\n"
                "Short answer:"
            ),
        }
    ]


def generate_baseline_answer(
    query: str,
    contexts: List[str],
    llm_generate: Optional[LLMClient] = None,
    temperature: float = 0.0,
    max_tokens: int = 80,
) -> str:
    context_block = format_baseline_context(contexts)
    messages = build_baseline_messages(query, context_block)

    if llm_generate is None:
        return messages[-1]["content"]

    return llm_generate(messages, temperature=temperature, max_tokens=max_tokens)
