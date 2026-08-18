"""LLM generation clients. Ollama (open-source, local) is the default provider;
Anthropic is supported as an alternative. Both fail gracefully — any error
(no server running, no API key, network unreachable) falls back to an
extractive synthesizer rather than raising or fabricating an answer.
"""
import requests

from . import config


def call_ollama(prompt, model=None, host=None, timeout=30):
    model = model or config.OLLAMA_MODEL
    host = host or config.OLLAMA_HOST
    resp = requests.post(
        f"{host}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def call_anthropic(prompt, model=None, api_key=None, max_tokens=150):
    import anthropic
    model = model or config.ANTHROPIC_MODEL
    api_key = api_key or config.ANTHROPIC_API_KEY
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(model=model, max_tokens=max_tokens,
                                   messages=[{"role": "user", "content": prompt}])
    return "".join(b.text for b in resp.content if hasattr(b, "text")).strip()


def generate_answer(question, context_edges, fallback_answer, provider=None):
    """Routes to the configured LLM_PROVIDER; falls back to extractive synthesis
    on any failure (no server, no key, network error) — never raises, never fakes."""
    provider = provider or config.LLM_PROVIDER
    context = "; ".join(context_edges)
    prompt = (f"Context (retrieved knowledge subgraph): {context}\n\n"
              f"Question: {question}\nAnswer concisely, using only the given context:")

    try:
        if provider == "ollama":
            return {"mode": "live_ollama", "text": call_ollama(prompt), "provider": f"ollama/{config.OLLAMA_MODEL}"}
        elif provider == "anthropic":
            if not config.ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY is empty")
            return {"mode": "live_anthropic", "text": call_anthropic(prompt), "provider": f"anthropic/{config.ANTHROPIC_MODEL}"}
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {provider}")
    except Exception as e:
        return {
            "mode": "fallback_extractive",
            "text": f"Answer: {fallback_answer}\nContext used: {context}",
            "provider": None,
            "error": f"{type(e).__name__}: {e}",
        }
