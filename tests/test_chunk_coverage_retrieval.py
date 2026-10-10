"""Search-layer guard: a clause the chunk floor used to drop must be retrievable.

`tests/test_chunk_coverage.py` proves the text is inside the chunk SET. That is
not the same claim as "a user can find it": retrieval adds query embedding, FAISS
ranking, the chunk -> notice mapping and the notice-deduplicated order. A clause
that never made it into a chunk passes every chunk-count, size and fingerprint
check and still cannot be returned by any query — so the property is asserted
where it matters, through the real `SemanticSearcher`.

The retrieval work runs in `tests/retrieval_probe.py`, a subprocess: a faiss
search performed before the embedding stack is imported into the same process
aborts on this host (`OMP: Error #15`; see that module's docstring). This module
therefore builds the task, runs the probe and asserts on its JSON, and touches
neither engine — so being collected before `test_entrypoints.py` cannot
destabilize the suite the way an in-process index would.

Fixtures (built together by `_workspace/build_chunk_coverage_fixture.py`):
  - `notices-chunk-coverage.jsonl` — 11 real notices, 8 of which lost content
    under the pre-merge floor, plus 3 controls.
  - `notices-chunk-coverage-queries.jsonl` — one query per dropped fragment: the
    source SENTENCE the fragment sits in (a hard-split fragment is a character
    slice of its sentence, so the sentence is the smallest text that must reach
    it). Each row also carries the fragment itself.

Two layers, because CI has neither the corpus nor a model cache:

  1. always (stub layer): the probe indexes the fixture notices with a
     deterministic hashing embedder (bag of character 3-grams, no model) and the
     real search path has to rank every clause query's notice first, with the
     clause itself inside the ranked window `/search` serves by default. The query
     is a substring of the text it must reach, so n-gram overlap carries it.
  2. when a cached model, the corpus DB and the live artifacts are all present:
     `scripts/05_evaluate.py` (the production retrieval script, real model, real
     index) must surface every clause query's notice within the first page of
     distinct notices. It skips while the live index predates the merge, which is
     the honest state of an unshipped fix.

Layer 2 does NOT assert recall@1. Measured on the rebuilt live index (2026-10-10,
chunks fingerprint 2c3e03179db0): 10 of 11 clause sentences rank their notice
first, and the eleventh (notice 2218244) ranks 8th — its sentence is generic
accounting-oversight boilerplate whose closest neighbours are identical
parallel-bill chunks (score 0.619 vs 0.574), so no semantic ranker can promise
rank 1 for it. The merge's own claim is that the clause text is IN the index and
reachable through the real stack; ranking quality is measured separately by
`scripts/05_evaluate.py` on the standalone eval sets.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from lawcast_semantic import config
from lawcast_semantic.chunking import chunk_notice, load_chunks_jsonl

FIXTURE_DIR = Path(__file__).resolve().parent / 'fixtures'
NOTICES_PATH = FIXTURE_DIR / 'notices-chunk-coverage.jsonl'
QUERIES_PATH = FIXTURE_DIR / 'notices-chunk-coverage-queries.jsonl'
PROBE_PATH = Path(__file__).resolve().parent / 'retrieval_probe.py'
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = Path(__file__).resolve().parents[2] / 'backend' / 'lawcast.db'
PROBE_TIMEOUT = 600
# The window a user actually sees: the sidecar's `/search` default top-k. Kept
# local so importing the app (fastapi + engine wiring) is not a collection-time
# cost of this guard, with `test_window_matches_the_sidecar_default` pinning it.
QUERY_WINDOW = 5
# First page of distinct notices the real-model layer accepts. Measured ceiling
# on the rebuilt live index is 8 (the boilerplate sentence of 2218244); 10 keeps
# one slot of headroom without turning the claim into "somewhere in the index".
SURFACE_BOUND = 10

_WHITESPACE_RE = re.compile(r'\s+')


def squash(text: str) -> str:
    return _WHITESPACE_RE.sub('', text)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


FIXTURE_NOTICES = load_jsonl(NOTICES_PATH)
FIXTURE_QUERIES = load_jsonl(QUERIES_PATH)
LOST_NOTICES = {entry['notice_num'] for entry in FIXTURE_QUERIES}


def clause_probe(index: int, entry: dict) -> dict:
    """The probe that asks for one formerly dropped fragment's own notice.

    Labels carry the fixture index because one notice can contribute several
    clauses (three rows share notice 2221062), and `run_probe` keys its results
    by label — a notice-number-only label would collapse them.
    """
    return {
        'label': f'clause {index} notice {entry["notice_num"]}',
        'query': entry['query'],
        'notice_num': entry['notice_num'],
    }


def control_probes() -> list[dict]:
    """Probes for the notices the floor never touched: still retrievable."""
    probes = []
    for notice in FIXTURE_NOTICES:
        if notice['notice_num'] in LOST_NOTICES:
            continue
        longest = max(chunk_notice(notice), key=lambda chunk: chunk.char_count)
        probes.append(
            {
                'label': f'control notice {notice["notice_num"]}',
                'query': longest.text[:120],
                'notice_num': notice['notice_num'],
            }
        )
    return probes


def run_probe(artifacts_dir: Path, probes: list[dict]) -> dict[str, dict]:
    """Index the fixture notices out of process and answer every probe there.

    Returns probe results keyed by label. One probe process serves the whole
    module (the index is built once), and its stdout is parsed strictly: a probe
    that cannot answer is a failure of this guard, not a skip.
    """
    assert len({probe['label'] for probe in probes}) == len(probes), (
        'probe labels must be unique: results are keyed by label, so a collision would hand '
        "one entry another entry's ranking"
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    task_path = artifacts_dir.parent / 'retrieval-task.json'
    task_path.write_text(
        json.dumps({'notices': FIXTURE_NOTICES, 'probes': probes}, ensure_ascii=False),
        encoding='utf-8',
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(PROBE_PATH),
            '--artifacts-dir',
            str(artifacts_dir),
            '--task',
            str(task_path),
            '--k',
            str(QUERY_WINDOW),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=PROBE_TIMEOUT,
    )
    assert completed.returncode == 0, (
        f'retrieval probe failed (exit {completed.returncode}):'
        f'\n{completed.stdout}\n{completed.stderr}'
    )
    results = [
        json.loads(line[len('RESULT ') :])
        for line in completed.stdout.splitlines()
        if line.startswith('RESULT ')
    ]
    assert len(results) == len(probes), (
        f'the probe answered {len(results)} of {len(probes)} questions:\n{completed.stdout}'
    )
    return {result['label']: result for result in results}


@pytest.fixture(scope='module')
def stub_retrieval(tmp_path_factory) -> dict[str, dict]:
    """One probe run for the module: the stub index is built once for all probes."""
    directory = tmp_path_factory.mktemp('retrieval-guard')
    probes = [clause_probe(index, entry) for index, entry in enumerate(FIXTURE_QUERIES)]
    return run_probe(directory / 'artifacts', probes + control_probes())


CLAUSE_CASES = list(enumerate(FIXTURE_QUERIES))


@pytest.mark.parametrize(
    'index, entry',
    CLAUSE_CASES,
    ids=[f'{entry["notice_num"]}-{index}' for index, entry in CLAUSE_CASES],
)
def test_dropped_clause_query_returns_its_notice(stub_retrieval: dict, index: int, entry: dict):
    """The clause's notice comes back first, carrying the clause, inside the window.

    Both halves are asserted on the k=5 window `/search` serves by default, and
    they fail differently: the ranked window has to hold a hit from the notice
    that CONTAINS the fragment (before the merge the fragment existed nowhere in
    the index, so no query could reach it — a chunk unreachable through search is
    the same loss from the user's side), and the query's first distinct notice has
    to be the clause's own notice.
    """
    query = entry['query'][:40]
    result = stub_retrieval[clause_probe(index, entry)['label']]
    hits = result['hits']
    assert hits, f'query returned nothing: {query!r}'
    fragment = squash(entry['dropped_fragment'])
    carrier = next(
        (
            hit
            for hit in hits
            if hit['notice_num'] == entry['notice_num'] and fragment in squash(hit['text'])
        ),
        None,
    )
    assert carrier is not None, (
        f'no hit among the {len(hits)} for {query!r} carries the dropped clause '
        f'{entry["dropped_fragment"][:40]!r} in notice {entry["notice_num"]}: the clause sits in '
        f'the index but no query reaches it (notice order: {result["notice_order"]})'
    )
    assert result['notice_order'][0] == entry['notice_num'], (
        f'notice {entry["notice_num"]} is not the first distinct notice for {query!r}; '
        f'order starts {result["notice_order"][:5]}'
    )


def test_every_control_fixture_notice_is_still_retrievable(stub_retrieval: dict):
    """The notices the floor never touched must keep ranking first for their own text."""
    labels = [label for label in stub_retrieval if label.startswith('control notice ')]
    assert len(labels) >= 3
    for label in labels:
        result = stub_retrieval[label]
        assert result['notice_order'][0] == result['notice_num'], (
            f'control notice {result["notice_num"]} is no longer retrievable from its own '
            f'text; order starts {result["notice_order"][:5]}'
        )


def test_window_matches_the_sidecar_default():
    """The guard must probe the window the API serves, not a wider one."""
    from service.app import DEFAULT_K

    assert DEFAULT_K == QUERY_WINDOW


def test_query_fixture_is_floor_shaped():
    """The queries must probe what the floor dropped, not ordinary text."""
    assert len(FIXTURE_QUERIES) >= 8
    for entry in FIXTURE_QUERIES:
        fragment = entry['dropped_fragment']
        assert fragment, 'a query without its dropped fragment cannot prove anything'
        assert len(fragment) < config.CHUNK_MIN_CHARS, (
            f'{fragment[:30]!r} is {len(fragment)} chars: not a below-floor fragment'
        )
        # The query has to be more than the fragment, otherwise the test would
        # only be checking that a string equals itself.
        assert len(entry['query']) >= len(fragment)
        assert squash(fragment) in squash(entry['query'])


def _cached_model_available() -> bool:
    """True when the embedding model is already in the local HF cache (no download)."""
    from huggingface_hub import constants as hub_constants

    model_dir = f'models--{config.MODEL_NAME.replace("/", "--")}'
    return (Path(hub_constants.HF_HUB_CACHE) / model_dir).exists()


def _live_index_covers_the_fragments() -> bool:
    """True when the artifacts in `config.ARTIFACTS_DIR` already carry the fragments.

    A live index built before the merge cannot answer these queries at all, so
    the check below skips instead of reporting a code failure against a stale
    deployment.
    """
    if not config.CHUNKS_PATH.exists():
        return False
    records = load_chunks_jsonl(config.CHUNKS_PATH)
    text_by_notice: dict[int, str] = {}
    for record in records:
        notice_num = int(record['notice_num'])
        text_by_notice[notice_num] = text_by_notice.get(notice_num, '') + record['text']
    return all(
        squash(entry['dropped_fragment']) in squash(text_by_notice.get(entry['notice_num'], ''))
        for entry in FIXTURE_QUERIES
    )


@pytest.mark.skipif(
    not (_cached_model_available() and CORPUS_PATH.exists()),
    reason='needs the cached embedding model and the corpus DB',
)
def test_real_index_surfaces_every_dropped_clause_within_a_page():
    """Production path: real model + real artifacts, via `scripts/05_evaluate.py`.

    `05_evaluate.py` uses `config` paths (so `LAWCAST_SEMANTIC_ARTIFACTS_DIR`
    selects the artifact set) and reports notice-deduplicated rankings. It runs
    in a subprocess because it must load the model and FAISS in a process that
    installs the single-threaded OpenMP environment first
    (`lawcast_semantic.omp_env`) — loading both in the pytest process aborts.

    The assertion is bounded surface, not recall@1: measured on the rebuilt live
    index, 10/11 clause sentences rank their notice first and the boilerplate
    sentence of 2218244 ranks 8th behind identical parallel-bill text (see the
    module docstring). Every clause's own notice must still appear within the
    first page of distinct notices — that fails loudly if the merge or the
    ranking regresses, without demanding a semantic guarantee the corpus's
    shared drafting language cannot give.
    """
    if not _live_index_covers_the_fragments():
        pytest.skip(
            'the artifacts in config.ARTIFACTS_DIR predate the chunk-floor merge, so these '
            'queries cannot be answered yet; rebuild with scripts/06_incremental_update.py'
        )
    completed = subprocess.run(
        [sys.executable, 'scripts/05_evaluate.py', '--eval', str(QUERIES_PATH)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    summary = next(
        (line for line in completed.stdout.splitlines() if line.startswith('all')),
        None,
    )
    assert summary, f'no summary line in evaluation output:\n{completed.stdout}'
    # Every miss line ends with ', rank N' (a fully unretrieved query prints
    # 'rank miss' and must not parse as a small number).
    ranks = [
        int(match.group(1))
        for line in completed.stdout.splitlines()
        if (match := re.search(r'-> notice \d+, rank (\d+)$', line))
    ]
    # Cross-check the parse against the summary's recall@1 so an output-format
    # change cannot turn the surface assertion into a vacuous pass.
    fields = summary.split()
    total, recall_at_1 = int(fields[1]), float(fields[2])
    expected_misses = total - round(recall_at_1 * total)
    assert len(ranks) == expected_misses, (
        f'parsed {len(ranks)} ranked misses but recall@1 {recall_at_1} over {total} queries '
        f'implies {expected_misses} (05_evaluate.py output format changed?):\n'
        f'{completed.stdout}'
    )
    assert max(ranks, default=1) <= SURFACE_BOUND, (
        'a formerly dropped clause fell out of the first page of the live ranking:\n'
        f'{completed.stdout}'
    )
