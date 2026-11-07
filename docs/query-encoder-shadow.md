# Query encoder shadow replay

The HTTP service always uses its existing full query encoder. BGE20 is loaded
only by the one-shot replay script. The runner reads fixed, operator-supplied
queries and the same FAISS generation, alternates the encoder order, and
reports one JSON log record per pair. It does not bind a port, read a live
request, update the index, or send a response to a user. The default FAISS
builder still writes FP32.

The projection was trained against the dense embedding space of
[`dragonkue/BGE-m3-ko`](https://huggingface.co/dragonkue/BGE-m3-ko), revision
`7074d66aa46562342193ca4feb3d89bf9dad71b4` (Apache 2.0). The checkpoint is
the projection only; the pinned backbone remains in the HF cache. The
teacher and student share the backbone sequentially in this isolated process.
Layer selection is never changed in a request-serving process.

For a production run, use the released semantic-search image, the serving
artifacts and HF cache mounted read-only, the replay queries mounted
read-only, no Docker network, a CPU limit and a timeout. Example container
command (after supplying these mounts):

```sh
python scripts/11_shadow_replay.py --queries /probe/queries.json --k 50 --repeat 2
```

The logs include encoding, FAISS retrieval, reranking (zero unless
`--rerank` is explicitly passed), annotation/serialization, and total sidecar
compute. They exclude the backend HTTP hop, archive hydration, and WAN.
Measure those separately using the real backend origin, with BGE20 still
disconnected. Do not compare a rerank-off run to a rerank-on run. For a
replay with reranking, use the same cached reranker, top-K and max length
for both encoders.
