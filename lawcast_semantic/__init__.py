"""LawCast semantic search pipeline package.

Stages:
  0. datasource            - load learning records (DB `proposalReason` / snapshots)
  1. preprocess / chunking - clean legislation text, split into chunks
  2. embedding             - tokenize and extract embeddings (Korean model)
  3. indexing              - build and persist a FAISS index
  4. search                - query processing and similarity ranking

Public names are re-exported lazily (PEP 562): `from lawcast_semantic import
SemanticSearcher` still works, but importing a light submodule (e.g.
`lawcast_semantic.datasource`) no longer loads torch/faiss/sentence-transformers.
Offline CLI and library consumers therefore share one package without the data
tools depending on the model stack.
"""

from __future__ import annotations

from importlib import import_module

# Public API: attribute name -> defining submodule (loaded on first access).
_EXPORTS = {
    'Chunk': '.chunking',
    'chunk_notices': '.chunking',
    'KoreanEmbedder': '.embedding',
    'VectorIndex': '.indexing',
    'detect_sections': '.preprocess',
    'normalize_text': '.preprocess',
    'SearchResult': '.search',
    'SearchResults': '.search',
    'SemanticSearcher': '.search',
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value  # cache so later lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *__all__])
