"""Contract tests for the scheduled update runner (design §2/§5/§6).

Exercised at the real surface: real plan/apply/reload against a tiny corpus
in a temp artifacts dir plus a LawCast-shaped sqlite DB (read mode=ro), with a
deterministic stub embedder (no model download). Tables pin the policy
boundaries (§6 shrink guard, §6.1 repair approval); scenario tests pin the
swap semantics (§5 readiness held across a generation swap, old generation
preserved on failure, §5.3 fingerprint self-heal), the scheduler gates, and
§5.2.1 `POST /reload` boundaries over HTTP.
"""

from __future__ import annotations

import fcntl
import json
import sqlite3
import threading
from datetime import datetime
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from test_datasource import make_db
from test_incremental import StubEmbedder, artifact_paths, make_notice, run_update

import service.app as app_module
import service.update_runner as runner
from lawcast_semantic import config
from lawcast_semantic.chunking import compute_chunks_fingerprint
from lawcast_semantic.datasource import load_notices_from_db
from lawcast_semantic.incremental import UpdatePlan
from lawcast_semantic.search import SemanticSearcher

NOTICE_COUNT = 120  # > 100 so the shrink guard's "deletions > 100" arm is reachable


def db_row(notice: dict) -> tuple:
    return (
        notice['notice_num'],
        notice['subject'],
        notice['proposer_category'],
        notice['committee'],
        notice['proposal_reason'],
        None,
    )


def make_notices(count: int = NOTICE_COUNT) -> list[dict]:
    reason = '제안이유 본문입니다. 이 법안은 국민의 권익 보호와 사회 안정을 도모합니다.'
    return [make_notice(1000 + index, reason=f'{reason} ({index})') for index in range(count)]


def load_args() -> dict:
    return {
        'chunks_path': config.CHUNKS_PATH,
        'index_path': config.FAISS_INDEX_PATH,
        'id_map_path': config.ID_MAP_PATH,
    }


def execute(db_path, sql: str, params: tuple = ()) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.execute(sql, params)


def artifact_snapshot(paths: dict) -> dict:
    return {name: path.read_bytes() if path.exists() else None for name, path in paths.items()}


def ready_state(env):
    """Boot-equivalent state: real load with the fingerprint recorded (§5.2)."""
    searcher = SemanticSearcher.load(env.embedder, **load_args())
    state = app_module.create_state()
    state.mark_ready(
        searcher,
        env.embedder.model_name,
        compute_chunks_fingerprint(searcher.chunks_by_id.values()),
    )
    return state


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """Temp artifacts + LawCast-shaped DB, config pointed at both."""
    paths = artifact_paths(tmp_path / 'artifacts')
    notices = make_notices()
    db_path = make_db(tmp_path, [db_row(notice) for notice in notices])
    for name, value in {
        'ARTIFACTS_DIR': paths['chunks_path'].parent,
        'CHUNKS_PATH': paths['chunks_path'],
        'EMBEDDINGS_PATH': paths['embeddings_path'],
        'FAISS_INDEX_PATH': paths['index_path'],
        'ID_MAP_PATH': paths['id_map_path'],
        'DB_PATH': str(db_path),
        'ALLOW_LARGE_DELETE': False,
    }.items():
        monkeypatch.setattr(config, name, value)
    # Baseline artifacts from the DB-read corpus (same reader the runner uses).
    run_update(load_notices_from_db(db_path), StubEmbedder(), paths)
    return SimpleNamespace(db_path=db_path, paths=paths, embedder=StubEmbedder())


# --- §6 shrink guard: deletions > 20% AND > 100 notices (either alone is fine) ---


@pytest.mark.parametrize(
    'deleted, old_total, allow_large, refused',
    [
        (0, 120, False, False),
        (25, 120, False, False),  # 21% but <= 100 notices
        (100, 120, False, False),  # boundary: not > 100
        (110, 120, False, True),  # both conditions
        (110, 600, False, False),  # > 100 but <= 20%
        (121, 600, False, True),  # just past 20%
        (110, 120, True, False),  # operator override (§4.2)
    ],
)
def test_shrink_guard_thresholds(deleted, old_total, allow_large, refused, monkeypatch):
    monkeypatch.setattr(config, 'ALLOW_LARGE_DELETE', allow_large)
    plan = UpdatePlan(
        new_records=tuple({'notice_num': num} for num in range(old_total - deleted)),
        embed_records=(),
        reused_chunk_ids=(),
        dropped_chunk_ids=(),
        added_notices=(),
        updated_notices=(),
        deleted_notices=tuple(range(deleted)),
        records_changed=False,
    )
    assert runner._shrink_refused(plan) is refused


# --- §6.1 boot repair: approve iff the cycle embeds nothing ---


def test_boot_repair_approves_torn_set_with_zero_embeds(env):
    """Crash-window set (new npz/id_map/index, stale chunks.jsonl) repairs in place."""
    stale_chunks = env.paths['chunks_path'].read_bytes()
    notice = make_notice(9999, reason='신규 의안의 제안이유입니다. 본안의 필요성을 설명합니다.')
    execute(env.db_path, 'INSERT INTO notice_archives VALUES (?,?,?,?,?,?)', db_row(notice))
    run_update(load_notices_from_db(env.db_path), StubEmbedder(), env.paths)  # writes v2...
    env.paths['chunks_path'].write_bytes(stale_chunks)  # ...but the commit point never landed

    with pytest.raises(ValueError):
        SemanticSearcher.load(StubEmbedder(), **load_args())  # boot would fail on this set

    repair_embedder = StubEmbedder()
    assert runner.run_boot_repair(repair_embedder, env.db_path) is True
    assert repair_embedder.embedded_texts == []  # 0-embed repair (§6.1)

    searcher = SemanticSearcher.load(repair_embedder, **load_args())
    assert 9999 in {record['notice_num'] for record in searcher.chunks_by_id.values()}


@pytest.mark.parametrize('mutation', ['edit', 'shrink', 'missing'])
def test_boot_repair_refuses_and_leaves_artifacts_untouched(env, mutation):
    """Needs-embedding (no baseline) and guarded shrink both refuse (§6.1/§6)."""
    if mutation == 'edit':
        execute(
            env.db_path,
            'UPDATE notice_archives SET proposalReason = ? WHERE noticeNum = ?',
            ('전부 새로 작성된 제안이유입니다. 기존 본문과 전혀 다른 내용을 담고 있습니다.', 1005),
        )
    elif mutation == 'shrink':
        execute(env.db_path, 'DELETE FROM notice_archives WHERE noticeNum < 1110')  # 110 of 120
    else:
        for path in env.paths.values():
            path.unlink(missing_ok=True)

    before = artifact_snapshot(env.paths)
    assert runner.run_boot_repair(env.embedder, env.db_path) is False
    assert artifact_snapshot(env.paths) == before


# --- §5.2 cycle: swap semantics ---


def test_cycle_reports_unchanged_when_in_sync(env):
    state = ready_state(env)
    assert runner.run_update_cycle(state, env.embedder) == 'unchanged'
    assert state.generation == 1  # no reload, no generation bump
    assert state.last_update_result == 'unchanged'
    assert state.last_update_error is None
    assert state.last_update_at is not None


def test_cycle_swaps_generation_while_old_keeps_serving(env, monkeypatch):
    """New set is validated while status stays 'ready' and the old searcher serves."""
    state = ready_state(env)
    old = state.searcher
    notice = make_notice(9999, reason='신규 의안의 제안이유입니다. 본안의 필요성을 설명합니다.')
    execute(env.db_path, 'INSERT INTO notice_archives VALUES (?,?,?,?,?,?)', db_row(notice))

    observed = []
    real_load = SemanticSearcher.load

    def spy(cls, embedder, **kwargs):
        observed.append((state.status, state.searcher))
        return real_load(embedder, **kwargs)

    monkeypatch.setattr(SemanticSearcher, 'load', classmethod(spy))

    assert runner.run_update_cycle(state, env.embedder) == 'changed'
    assert observed == [('ready', old)]  # ready + old generation during validation
    assert state.status == 'ready'
    assert state.generation == 2
    assert state.searcher is not old
    assert state.reload_error is None
    assert 9999 in {record['notice_num'] for record in state.searcher.chunks_by_id.values()}
    # The tick wrote new artifacts and the swap adopted their stamp.
    written = json.loads(env.paths['id_map_path'].read_text(encoding='utf-8'))
    assert state.last_update_at == written['updated_at']


def test_cycle_keeps_previous_generation_on_reload_failure_then_self_heals(env, monkeypatch):
    """§5.2 rollback: failed swap keeps serving; §5.3: fingerprint mismatch retries."""
    state = ready_state(env)
    old, generation = state.searcher, state.generation
    notice = make_notice(9999, reason='신규 의안의 제안이유입니다. 본안의 필요성을 설명합니다.')
    execute(env.db_path, 'INSERT INTO notice_archives VALUES (?,?,?,?,?,?)', db_row(notice))

    real_load = SemanticSearcher.load
    calls = []

    def fail_once(cls, embedder, **kwargs):
        calls.append(embedder)
        if len(calls) == 1:
            raise ValueError('boom')
        return real_load(embedder, **kwargs)

    monkeypatch.setattr(SemanticSearcher, 'load', classmethod(fail_once))

    assert runner.run_update_cycle(state, env.embedder) == 'failed'
    assert state.status == 'ready'  # still serving, never back to loading/failed
    assert state.searcher is old
    assert state.generation == generation  # preserved
    assert 'boom' in state.reload_error
    assert state.last_update_result == 'failed'

    # Artifacts are already updated on disk; the next tick reloads on
    # fingerprint mismatch even though the corpus itself is unchanged (§5.3).
    assert runner.run_update_cycle(state, env.embedder) == 'changed'
    assert state.generation == generation + 1
    assert state.searcher is not old
    assert state.reload_error is None


def test_cycle_refuses_large_delete_and_keeps_serving(env):
    state = ready_state(env)
    before = artifact_snapshot(env.paths)
    execute(env.db_path, 'DELETE FROM notice_archives WHERE noticeNum < 1110')  # 110 of 120

    assert runner.run_update_cycle(state, env.embedder) == 'failed'
    assert 'shrink guard' in state.last_update_error
    assert state.status == 'ready'
    assert state.generation == 1
    assert artifact_snapshot(env.paths) == before


@pytest.mark.parametrize('scenario, expected', [('lock', 'skipped'), ('no_db', 'failed')])
def test_cycle_skip_and_failure_outcomes(env, monkeypatch, scenario, expected):
    state = ready_state(env)
    if scenario == 'lock':
        handle = open(config.ARTIFACTS_DIR / '.update.lock', 'a')
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            result = runner.run_update_cycle(state, env.embedder)
        finally:
            handle.close()
    else:
        monkeypatch.setattr(config, 'DB_PATH', str(env.db_path) + '.missing')
        result = runner.run_update_cycle(state, env.embedder)
    assert result == expected
    assert state.last_update_result == expected


# --- §6.1 wiring: lifespan hands run_boot_repair to load_engine's searcher phase ---


@pytest.mark.parametrize(
    'db_path, interval, scheduler_starts',
    [('', 60, False), ('/data/lawcast.db', 0, False), ('/data/lawcast.db', 60, True)],
)
def test_lifespan_wires_boot_repair_hook_and_gates_scheduler(
    monkeypatch, db_path, interval, scheduler_starts
):
    """Hook is connected unconditionally (load_engine gates on DB_PATH); the
    scheduler thread starts only for DB_PATH set AND interval > 0 (§4.2)."""
    loaded = threading.Event()
    scheduler_ran = threading.Event()
    captured = {}

    def spy_loader(state, boot_repair=None):
        captured['boot_repair'] = boot_repair
        loaded.set()

    monkeypatch.setattr(app_module, 'STATE', app_module.create_state())
    monkeypatch.setattr(app_module, 'load_engine', spy_loader)
    monkeypatch.setattr(runner, 'start_scheduler', lambda state: scheduler_ran.set())
    monkeypatch.setattr(config, 'DB_PATH', db_path)
    monkeypatch.setattr(config, 'UPDATE_INTERVAL_MINUTES', interval)

    with TestClient(app_module.app):
        assert loaded.wait(timeout=5)
    if scheduler_starts:
        assert scheduler_ran.wait(timeout=5)
    else:
        assert not scheduler_ran.wait(timeout=0.2)
    assert captured['boot_repair'] is runner.run_boot_repair


# --- §4.2 scheduler: ready-gated ticks, stopped by failure ---


def test_scheduler_gates_on_status_and_stops_when_failed(monkeypatch):
    state = app_module.create_state()
    searcher = SimpleNamespace(chunk_ids=[], embedder='live-embedder', index_updated_at=None)
    ticks = []
    monkeypatch.setattr(runner, 'run_update_cycle', lambda state_, embedder: ticks.append(embedder))
    script = [
        lambda: None,  # still loading -> no tick
        lambda: state.mark_ready(searcher, 'stub-model'),
        lambda: state.mark_failed('boom'),
    ]
    delays = []

    def fake_sleep(seconds):
        delays.append(seconds)
        if not script:
            raise AssertionError('scheduler kept running after failure')
        script.pop(0)()

    monkeypatch.setattr(runner, 'sleep', fake_sleep)
    runner.start_scheduler(state)  # returns once the state is failed

    assert ticks == ['live-embedder']  # embedder taken from the served searcher
    assert len(delays) == 3
    interval = config.UPDATE_INTERVAL_MINUTES * 60
    assert all(interval * 0.9 <= delay <= interval * 1.1 for delay in delays)  # ±10% jitter


# --- §5.2.1 POST /reload: the scheduler's swap path over HTTP ---


@pytest.fixture()
def reload_client(env, monkeypatch):
    """Ready engine (real artifacts) behind the real endpoint; no scheduler."""
    state = ready_state(env)
    monkeypatch.setattr(app_module, 'STATE', state)
    monkeypatch.setattr(app_module, 'load_engine', lambda *_: None)  # state already ready
    monkeypatch.setattr(config, 'UPDATE_INTERVAL_MINUTES', 0)
    with TestClient(app_module.app) as client:
        yield client, state


def test_reload_swaps_in_host_written_artifacts(reload_client, env):
    """Host pipeline 01→03 (or 06) wrote artifacts; one curl hot-swaps them."""
    client, state = reload_client
    notice = make_notice(9999, reason='신규 의안의 제안이유입니다. 본안의 필요성을 설명합니다.')
    execute(env.db_path, 'INSERT INTO notice_archives VALUES (?,?,?,?,?,?)', db_row(notice))
    run_update(load_notices_from_db(env.db_path), StubEmbedder(), env.paths)
    old, generation = state.searcher, state.generation

    response = client.post('/reload')

    assert response.status_code == 200
    body = response.json()
    assert body['status'] == 'ready'  # readiness held throughout the swap
    assert body['generation'] == generation + 1
    assert body['reloadError'] is None
    assert body['lastUpdateResult'] is None  # manual reloads leave tick fields alone
    assert state.searcher is not old
    assert 9999 in {record['notice_num'] for record in state.searcher.chunks_by_id.values()}


def test_reload_keeps_old_generation_on_validation_failure(reload_client, env):
    """503 + reloadError; the old generation keeps serving (§5.2.4)."""
    client, state = reload_client
    old, generation = state.searcher, state.generation
    env.paths['chunks_path'].unlink()  # torn disk set (e.g. crash mid host write)

    response = client.post('/reload')

    assert response.status_code == 503
    assert 'semantic reload failed' in response.json()['detail']
    health = client.get('/health').json()
    assert health['status'] == 'ready'
    assert health['generation'] == generation
    assert health['reloadError']  # recorded for the §5.3 next-tick retry
    assert state.searcher is old


@pytest.mark.parametrize('status', ['loading', 'failed'])
def test_reload_requires_ready_engine(monkeypatch, status):
    state = app_module.create_state()
    if status == 'failed':
        state.mark_failed('OSError: artifacts missing')
    monkeypatch.setattr(app_module, 'STATE', state)
    monkeypatch.setattr(app_module, 'load_engine', lambda *_: None)
    with TestClient(app_module.app) as client:
        response = client.post('/reload')
    assert response.status_code == 409
    assert f'status={status}' in response.json()['detail']
    assert state.generation == 0  # nothing was swapped


# --- index last-update time: id_map `updated_at` (VectorIndex.save) -> /health ---


def test_artifact_write_stamps_updated_at(env):
    """Every writer (03, 06, tick, boot repair) funnels through
    VectorIndex.save, the single owner of the id_map `updated_at` stamp."""
    payload = json.loads(env.paths['id_map_path'].read_text(encoding='utf-8'))
    stamp = datetime.fromisoformat(payload['updated_at'])  # valid ISO 8601
    assert stamp.tzinfo is not None  # UTC-aware


def test_responses_report_serving_index_updated_at(reload_client, env):
    """Boot load adopts the stamp; /health and /search expose it for the
    backend's single-request consumption."""
    client, state = reload_client
    stamp = json.loads(env.paths['id_map_path'].read_text(encoding='utf-8'))['updated_at']
    assert state.last_update_at == stamp
    for body in (
        client.get('/health').json(),
        client.get('/search', params={'query': '의안'}).json(),
    ):
        assert body['lastUpdateAt'] == stamp


def test_ticks_never_move_last_update_at(env):
    """Single ownership: tick outcomes (unchanged/failed) record result+error
    only — the time moves solely when artifacts are rewritten."""
    state = ready_state(env)
    before = state.last_update_at
    assert before is not None  # adopted from the fixture's baseline write

    assert runner.run_update_cycle(state, env.embedder) == 'unchanged'
    assert state.last_update_at == before

    execute(env.db_path, 'DELETE FROM notice_archives WHERE noticeNum < 1110')  # shrink guard
    assert runner.run_update_cycle(state, env.embedder) == 'failed'
    assert state.last_update_result == 'failed'
    assert state.last_update_at == before


def test_reload_adopts_host_written_artifacts_timestamp(reload_client, env):
    """Host pipeline (01->03 or 06) wrote artifacts -> one POST /reload:
    the sidecar reports the write time, not the reload time."""
    client, state = reload_client
    old_stamp = state.last_update_at
    notice = make_notice(9999, reason='신규 의안의 제안이유입니다. 본안의 필요성을 설명합니다.')
    execute(env.db_path, 'INSERT INTO notice_archives VALUES (?,?,?,?,?,?)', db_row(notice))
    run_update(load_notices_from_db(env.db_path), StubEmbedder(), env.paths)
    payload = json.loads(env.paths['id_map_path'].read_text(encoding='utf-8'))
    assert payload['updated_at'] != old_stamp  # fresh write, fresh stamp

    response = client.post('/reload')

    assert response.status_code == 200
    assert response.json()['lastUpdateAt'] == payload['updated_at'] != old_stamp


def test_legacy_id_map_without_stamp_reports_null(reload_client, env):
    """Pre-stamp artifact sets load fine and report lastUpdateAt=null."""
    client, _state = reload_client
    assert client.get('/health').json()['lastUpdateAt'] is not None
    payload = json.loads(env.paths['id_map_path'].read_text(encoding='utf-8'))
    payload.pop('updated_at')
    env.paths['id_map_path'].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8'
    )

    response = client.post('/reload')

    assert response.status_code == 200
    assert response.json()['lastUpdateAt'] is None
    assert client.get('/search', params={'query': '의안'}).json()['lastUpdateAt'] is None


def test_reload_duplicate_in_flight_call_conflicts(reload_client, env, monkeypatch):
    """Second POST while the first holds the artifact lock: 409, single swap."""
    client, state = reload_client
    started, release = threading.Event(), threading.Event()
    real_reload = state.reload

    def slow_reload(embedder):
        started.set()  # inside run_reload's _update_lock
        release.wait(timeout=5)
        return real_reload(embedder)

    monkeypatch.setattr(state, 'reload', slow_reload)
    outcomes = []
    thread = threading.Thread(target=lambda: outcomes.append(client.post('/reload')))
    thread.start()
    try:
        assert started.wait(timeout=5)
        duplicate = client.post('/reload')
        assert duplicate.status_code == 409
        assert 'already in progress' in duplicate.json()['detail']
    finally:
        release.set()
        thread.join(timeout=5)
    assert outcomes[0].status_code == 200
    assert state.generation == 2  # exactly one swap
