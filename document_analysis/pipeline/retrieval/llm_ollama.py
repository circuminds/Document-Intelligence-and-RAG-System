from __future__ import annotations
import requests
from typing import Dict, Any, List

OLLAMA_URL = "http://localhost:11434/api/generate"


def build_prompt(query: str, evidence: List[Dict[str, Any]]) -> str:
    """
    evidence items should contain:
      - doc_id, page_number, chunk_id, text
    """
    # Create a compact evidence pack with explicit citation keys
    blocks = []
    for i, e in enumerate(evidence, start=1):
        cite = f"[S{i}: doc={e['doc_id']}, page={e['page_number']}, chunk={e['chunk_id']}]"
        text = (e["text"] or "").strip()
        blocks.append(f"{cite}\n{text}")

    evidence_pack = "\n\n---\n\n".join(blocks)

    prompt = f"""
You are a precise assistant. Answer the user ONLY using the SOURCES below.
Rules:
- If the sources do not contain enough information, say: "I don't have enough information in the provided documents."
- Every claim must include at least one citation tag like [S2].
- Do not invent facts.
- Keep the answer concise but complete.

User question:
{query}

SOURCES:
{evidence_pack}

Return format:
Answer:
<answer with citations>

Citations used:
- [S#] doc=<doc_id> page=<page> chunk=<chunk_id>
""".strip()
    return prompt


def ollama_generate(model: str, prompt: str, temperature: float = 0.2) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()
