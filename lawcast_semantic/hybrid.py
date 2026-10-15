"""Stage 4b: lexical evidence for hybrid retrieval and no-result gating.

A bi-encoder compresses every query into a high cosine band, so no absolute
similarity floor separates unrelated queries from vague-but-valid ones (see
agent_memories/20-semantic-query-quality-eval/ §10.4). This module owns the
complementary signal: shared surface tokens between the (alias-expanded) query
and each candidate chunk's `subject` + `text`.

It intentionally stays dependency-free (no torch/faiss) and ranking-free: it
scores and filters candidate windows the dense index already returned. Fusion
helpers (RRF) live here as well so a future dense+FTS retriever shares one
implementation.
"""

from __future__ import annotations

import re

from .aliases import expand_query
from .preprocess import normalize_text

# Hangul syllables, latin letters and digits form content tokens. Single
# characters (`법`, `에`) match everywhere, so they are excluded below.
_TOKEN_RE = re.compile(r'[0-9A-Za-z가-힣]{2,}')

# Function words that co-occur with everything in legislative prose. They add
# no evidence that a candidate is about the query, so overlap on these alone
# must not keep a window alive.
_STOP_TOKENS = frozenset(
    {
        '관한',
        '위한',
        '대한',
        '통한',
        '따른',
        '등에',
        '등의',
        '및',
        '또는',
        '그리고',
        '관련',
        '주요',
        '내용',
        '제안이유',
        '일부개정법률안',
        '전부개정법률안',
        '제정법률안',
        '법률안',
    }
)


def lexical_tokens(text: str | None) -> list[str]:
    """Split text into deduplicated content tokens for overlap scoring.

    The query path (alias expansion + normalization) is applied first so the
    tokens match what the dense path embeds. Order follows first occurrence.
    """
    expanded = expand_query(normalize_text(text or '')).text
    seen: list[str] = []
    for token in _TOKEN_RE.findall(expanded.casefold()):
        if token in _STOP_TOKENS or token in seen:
            continue
        seen.append(token)
    return seen


def overlap_count(query_tokens: list[str], doc_tokens: list[str]) -> int:
    """Number of query tokens with a surface match in the document tokens.

    Korean compounds are rarely space-separated (`주택임대차보호법` holds
    `임대차`), so a query token matches when it is a substring of a document
    token or vice versa. Single characters never reach this comparison because
    `lexical_tokens` already drops them.
    """
    count = 0
    for query_token in query_tokens:
        for doc_token in doc_tokens:
            if query_token in doc_token or doc_token in query_token:
                count += 1
                break
    return count


def chunk_tokens(chunk: dict) -> list[str]:
    """Content tokens of one chunk record (`subject` + `text`)."""
    subject = chunk.get('subject') or ''
    text = chunk.get('text') or ''
    return lexical_tokens(f'{subject}\n{text}')


def has_lexical_evidence(query: str | None, chunk: dict) -> bool:
    """Whether the query shares at least one content token with the chunk."""
    query_tokens = lexical_tokens(query)
    if not query_tokens:
        return False
    return overlap_count(query_tokens, chunk_tokens(chunk)) > 0


def filter_by_lexical_evidence(query: str | None, hits: list, chunks_by_id: dict) -> list:
    """Keep only hits whose chunk shares a content token with the query.

    Hits carry a `chunk_id` attribute (see `search.SearchResult`); chunks are
    looked up in `chunks_by_id`. Order is preserved. An unrelated query keeps
    nothing, which is the explicit no-result signal an absolute cosine floor
    cannot produce.
    """
    kept = []
    for hit in hits:
        chunk = chunks_by_id.get(getattr(hit, 'chunk_id', None))
        if chunk is not None and has_lexical_evidence(query, chunk):
            kept.append(hit)
    return kept


def rrf_fuse(rank_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal-rank fusion over candidate id rankings.

    Each input list ranks ids best-first (e.g. dense order, lexical order).
    Score per id is `sum(1 / (k + rank))`; higher is better. Rank positions —
    not raw scores — are fused, so dense cosine and lexical counts need no
    common scale.
    """
    if k < 1:
        raise ValueError(f'k must be >= 1, got {k}')
    fused: dict[str, float] = {}
    for ranking in rank_lists:
        for rank, chunk_id in enumerate(ranking, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return fused
