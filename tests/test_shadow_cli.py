import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / '11_shadow_replay.py'
FLAG = 'LAWCAST_SEMANTIC_DISTILLATION_SHADOW_ENABLED'


@pytest.mark.parametrize('value', [None, '', '0', 'false', 'true', '01', '1 '])
def test_shadow_requires_exact_process_environment_opt_in(tmp_path, value):
    dotenv = tmp_path / '.env'
    dotenv.write_text(f'{FLAG}=1\n', encoding='utf-8')
    env = dict(os.environ, LAWCAST_SEMANTIC_ENV_FILE=str(dotenv))
    if value is None:
        env.pop(FLAG, None)
    else:
        env[FLAG] = value
    result = subprocess.run(
        [sys.executable, str(SCRIPT), '--queries', str(tmp_path / 'missing.json')],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 2
    assert f'distillation shadow replay is disabled; set {FLAG}=1' in result.stderr
    assert 'FileNotFoundError' not in result.stderr


def test_shadow_opt_in_reaches_input_validation_without_loading_model(tmp_path):
    queries = tmp_path / 'queries.json'
    queries.write_text('[]', encoding='utf-8')
    env = dict(os.environ, LAWCAST_SEMANTIC_ENV_FILE='', **{FLAG: '1'})
    result = subprocess.run(
        [sys.executable, str(SCRIPT), '--queries', str(queries)],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 2
    assert 'use 1–50 fixed probes' in result.stderr
    assert 'distillation shadow replay is disabled' not in result.stderr
