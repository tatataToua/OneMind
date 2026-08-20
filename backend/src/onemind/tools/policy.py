"""Compliance data plane - the policy and regulation corpus.

Only the Compliance specialist holds this tool.

Retrieval is lexical: IDF-weighted term overlap with a phrase bonus, scored over
section-level chunks. For a six-document corpus that outperforms a naive
embedding search and, more usefully here, it is deterministic - the same query
returns the same citations in a test as in the demo. `nomic-embed-text` is
already available locally if this corpus grows enough to need dense retrieval;
the tool signature would not change.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Any

from . import store
from .base import obj_schema, tool, tools

_WORD = re.compile(r"[a-z0-9]+")

_STOP = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "is",
    "are",
    "for",
    "on",
    "we",
    "our",
    "do",
    "does",
    "it",
    "that",
    "this",
    "with",
    "be",
    "must",
    "what",
    "how",
    "when",
    "if",
    "any",
    "can",
    "you",
    "need",
    "there",
}


def _terms(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1]


@lru_cache(maxsize=1)
def _index() -> tuple[list[dict[str, Any]], dict[str, float]]:
    chunks = store.policies()
    docs = [_terms(f"{c['title']} {c['section']} {c['text']}") for c in chunks]
    n = len(docs)
    df: dict[str, int] = {}
    for terms in docs:
        for term in set(terms):
            df[term] = df.get(term, 0) + 1
    idf = {term: math.log(1 + n / count) for term, count in df.items()}
    enriched = [
        {**chunk, "_terms": terms, "_len": max(len(terms), 1)}
        for chunk, terms in zip(chunks, docs, strict=True)
    ]
    return enriched, idf


@tool(
    tools,
    name="policy_search",
    description=(
        "Search internal HIPAA, SOC 2, and ISO 27001 policy documents. Returns "
        "the most relevant policy sections with their source document, so the "
        "answer can cite them."
    ),
    parameters=obj_schema(
        {
            "query": {
                "type": "string",
                "description": "The compliance question, in plain language",
            },
            "limit": {"type": "integer", "description": "Max sections, default 5"},
        },
        required=["query"],
    ),
)
def policy_search(query: str, limit: int = 5) -> dict[str, Any]:
    # Default of 5, not 3. Specialists rephrase the user's question before
    # searching, and a rephrase can push the decisive section from rank 2 to
    # rank 4 - which produced a confidently wrong "the policy does not say"
    # answer on the BAA question. Recall is worth more than precision here
    # because the model still has to justify its answer from what comes back.
    chunks, idf = _index()
    wanted = _terms(query)
    if not wanted:
        return {"query": query, "count": 0, "results": []}

    needle = query.lower().strip()
    scored: list[tuple[float, dict[str, Any]]] = []
    for chunk in chunks:
        counts: dict[str, int] = {}
        for term in chunk["_terms"]:
            counts[term] = counts.get(term, 0) + 1
        score = sum(idf.get(term, 0.0) * (counts.get(term, 0) / chunk["_len"]) for term in wanted)
        # Reward a literal phrase hit; "business associate agreement" as a unit
        # should beat three scattered mentions.
        if len(needle) > 12 and needle in chunk["text"].lower():
            score *= 1.8
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    top = scored[: max(1, min(limit, 8))]
    return {
        "query": query,
        "count": len(top),
        "results": [
            {
                "doc_id": chunk["doc_id"],
                "title": chunk["title"],
                "section": chunk["section"],
                "citation": f"{chunk['title']} / {chunk['section']}",
                "text": chunk["text"][:1200],
                "score": round(score, 4),
            }
            for score, chunk in top
        ],
    }
