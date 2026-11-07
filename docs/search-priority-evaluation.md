# Search Priority Evaluation

The evaluation corpus combines the existing labeled and unrelated probes with
46 fixed cases in `data/eval_search_priorities.jsonl`. The new cases include
formal law names (multiple valid bills), complete bill identifiers, supported
single-jamo typos, protected input, out-of-domain queries, and recent bills
with evidence terms checked against their original notice text. These are
curated regression fixtures, not independent user-study judgments or training
data. Treat section evidence terms as a narrow automated proxy for evidence quality.

## Reproduce the vector comparison

Use one frozen chunk generation, one query-vector file and the same thread count.
Keep the candidate files outside the serving artifact directory:

```sh
cd semantic-search
work=$(mktemp -d /tmp/lawcast-search-ab.XXXXXX)
export OMP_NUM_THREADS=1
.venv/bin/python scripts/03_build_index.py --index-type sq8 \
  --out "$work/faiss.index" --id-map "$work/id_map.json"
.venv/bin/python scripts/10_search_index_ab.py encode --out "$work/vectors.npz"
.venv/bin/python scripts/10_search_index_ab.py rank --vectors "$work/vectors.npz" \
  --index artifacts/faiss.index --id-map artifacts/id_map.json --out "$work/flat.json"
.venv/bin/python scripts/10_search_index_ab.py rank --vectors "$work/vectors.npz" \
  --index "$work/faiss.index" --id-map "$work/id_map.json" --out "$work/sq8.json"
```

The rank step measures the actual FAISS call with a fixed chunk window and
exports the raw hits. It does not include query embedding, relevance gating,
HTTP or archive hydration. Compare those separately on the same stack. A
changed index may move a hit across a relevance boundary even when the
relevant notices in the first page are unchanged.

`flat` remains the default. An experimental SQ8 generation uses the identical
FP32 source embeddings and chunk metadata, and incremental writes preserve its
declared format. For a full comparison, copy the candidate's matching chunk
and embedding files to the same isolated directory; never run the builder
against only one file of a live generation. Roll back by pointing the service
to the unchanged flat generation. Do not reuse an SQ8 index with a flat ID map.
