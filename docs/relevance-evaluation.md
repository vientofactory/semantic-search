# Embedding-only relevance evaluation

The classifier is **opt-in**. Set `LAWCAST_SEMANTIC_RELEVANCE_ENABLED=true`
only on the calibrated `dragonkue/BGE-m3-ko` FlatIP setup. The default still
uses the legacy cosine tiers, but the cross-encoder loader, scorer, backend
cutoff, and internal score field are removed in both modes. Deploy the new
backend before enabling the option on the sidecar so it can distinguish a
short filtered result from an exhausted search window. No index rebuild is
required. Turning the option off restores the cosine tiers, not the removed
model.

## Fixed-index comparison

CPU replay on 122,268 indexed chunks (artifact stamp
`2026-11-06T18:00:35.586602+00:00`). Each comparison used the same top 100
dense candidates and collapsed results to five distinct notices per tier.
This is a fixed-window experiment, not a replay of the public automatic
search or a production latency benchmark.

The calibration set used 40 legacy noise probes, eight positive controls
and 16 labeled queries. Six ambiguous legacy probes were excluded from
threshold selection. The separate final set used 40 newly written
out-of-scope queries and 42 previously labeled positives, with exact
normalized calibration queries and known overlapping target cases removed.
The negatives were labeled by query intent, not by a failure to retrieve
a hit; they are not a representative sample of production traffic or
independent human judgments of all candidate documents.

- **Both tiers empty on negatives:** legacy cosine 0/40; legacy lexical
  enforcement 9/40; higher cosine floors alone 39/40; higher floors plus
  background 39/40; full classifier 39/40 (Wilson 95% CI 87.1–99.6%).
  Paired exact McNemar p=3.64e-12 vs legacy cosine and p=1.86e-9 vs the
  legacy lexical variant. These are query-level sample tests; queries
  within an intent category can be correlated.
- **Strong empty on negatives:** 36/40 to 39/40 vs legacy cosine
  (three wins, zero losses, p=0.25). The significant change was in
  emptying *both* tiers, not in Strong alone.
- **Known positive hit in Strong@5:** 30/42 legacy, 28/42 with higher
  cosine floors alone, 25/42 full classifier. **Known hit in either tier's
  top five:** 31/42 legacy and 31/42 full classifier (three gains and
  three losses); only 38/42 had any labeled target in the raw candidates.
  Any visible result, even if not a labeled target, went from 42/42 to
  39/42. In calibration the full classifier emptied 38/40 negatives
  and retained all 13 legacy Strong@5 target hits.
- The surviving final negative asked how to check a personal pension
  projection in an app. It matched a pension bill by topic but not by
  intent. Raising cosine floors explained the measured final-set rejection
  gain; the experiment does not establish incremental rejection value for
  background or IDF. The Strong recall loss is why the option is off.

Candidate judgments are not exhaustive, so these results do not estimate
Strong precision or prove a production false-exposure target. Exact bill
lookup and keyword retrieval keep their existing paths; no BM25 candidate
union or embedding-model replacement is part of this change.

## Reproduce

From the semantic-search checkout with its existing model cache and artifacts:

```sh
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 LAWCAST_SEMANTIC_RELEVANCE_ENABLED=true \
  .venv/bin/python scripts/12_eval_relevance.py --out /tmp/relevance-holdout.json
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 LAWCAST_SEMANTIC_RELEVANCE_ENABLED=true \
  .venv/bin/python scripts/12_eval_relevance.py --split calibration --out /tmp/relevance-calibration.json
```
