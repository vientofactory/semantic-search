"""LawCast semantic search pipeline package.

Stages:
  1. preprocess / chunking  - clean legislation text, split into chunks
  2. embedding              - tokenize and extract embeddings (Korean model)
  3. indexing               - build and persist a FAISS index
  4. search                 - query processing and similarity ranking
"""

from .chunking import Chunk, chunk_notices
from .embedding import KoreanEmbedder
from .indexing import VectorIndex
from .preprocess import detect_sections, normalize_text
from .search import SearchResult, SemanticSearcher

__all__ = [
    'Chunk',
    'chunk_notices',
    'detect_sections',
    'KoreanEmbedder',
    'normalize_text',
    'SearchResult',
    'SemanticSearcher',
    'VectorIndex',
]
