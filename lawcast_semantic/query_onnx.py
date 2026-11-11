"""Self-contained, validated CPU BGE query runtime without PyTorch weights."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn.functional import normalize
from transformers import AutoTokenizer

from .timing import DensePhase, EmbeddingStages

MODEL = 'dragonkue/BGE-m3-ko'
DIMENSION = 1024
PROBES = ('민원 처리 절차', '정책 영향과 이해관계자', 'embedding parity 123')


def manifest_path(graph: Path) -> Path:
    return graph.with_name(f'{graph.name}.validation.json')


def file_digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class OnnxQueryEncoder:
    """Run the pinned FP32 graph and the canonical BGE CLS/normalization tail."""

    def __init__(self, graph: Path, threads: int, *, validation: dict | None = None) -> None:
        from onnxruntime import InferenceSession, SessionOptions

        if validation is None:
            validation = json.loads(manifest_path(graph).read_text(encoding='utf-8'))
        self.revision = validation['revision']
        self.dimension = validation['dimension']
        self.max_seq_length = validation['maxSeqLength']
        if (
            validation['version'] != 1
            or validation['model'] != MODEL
            or not re.fullmatch(r'[0-9a-f]{40}', self.revision)
            or self.dimension != DIMENSION
            or self.max_seq_length != 8192
        ):
            raise ValueError('unsupported query ONNX bundle')
        hashes = validation['sha256']
        required = {graph.name, f'{graph.name}_data', 'tokenizer/tokenizer.json'}
        if not required.issubset(hashes):
            raise ValueError('query bundle is missing graph, weights or tokenizer digests')
        for name, expected in hashes.items():
            path = Path(name)
            if name not in required and (len(path.parts) != 2 or path.parent != Path('tokenizer')):
                raise ValueError('unexpected query bundle file')
            if file_digest(graph.parent / path) != expected:
                raise ValueError(f'query bundle digest mismatch: {name}')
        tokenizer_files = {
            f'tokenizer/{path.name}'
            for path in (graph.parent / 'tokenizer').iterdir()
            if path.is_file()
        }
        if tokenizer_files != set(hashes) - {graph.name, f'{graph.name}_data'}:
            raise ValueError('query bundle tokenizer files differ from validation')

        self.tokenizer = AutoTokenizer.from_pretrained(
            graph.parent / 'tokenizer', local_files_only=True
        )
        options = SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.add_session_config_entry('session.intra_op.allow_spinning', '0')
        self.session = InferenceSession(
            str(graph), sess_options=options, providers=['CPUExecutionProvider']
        )
        inputs = {item.name: item.type for item in self.session.get_inputs()}
        outputs = {item.name: item.type for item in self.session.get_outputs()}
        if inputs != {
            'input_ids': 'tensor(int64)',
            'attention_mask': 'tensor(int64)',
        } or outputs != {'last_hidden_state': 'tensor(float)'}:
            raise ValueError('unexpected FP32 query graph signature')
        expected = np.asarray(validation['probeVectors'], dtype=np.float32)
        actual = np.stack([self.encode(text)[0] for text in PROBES])
        if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=0, atol=3e-5):
            raise ValueError('query graph failed offline PyTorch vector parity probes')
        self.probe_delta = float(np.max(np.abs(actual - expected)))

    def encode(self, text: str) -> tuple[np.ndarray, EmbeddingStages]:
        started = perf_counter()
        tokens = self.tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors='np',
        )
        tokenized = perf_counter()
        inputs = {name: tokens[name] for name in ('input_ids', 'attention_mask')}
        (hidden,) = self.session.run(['last_hidden_state'], inputs)
        if hidden.shape != (1, inputs['input_ids'].shape[1], self.dimension):
            raise ValueError('unexpected query graph output shape')
        # BGE pools the first token, then normalizes in the model and in encode().
        cls = torch.from_numpy(hidden[:, 0, :])
        vector = normalize(normalize(cls, dim=1), dim=1)[0]
        converted = perf_counter()
        output = vector.numpy()
        if not np.isfinite(output).all() or not np.any(output):
            raise ValueError('query graph returned a non-finite or zero embedding')
        finished = perf_counter()
        return output, EmbeddingStages(
            tokenization=DensePhase(started, tokenized),
            model=DensePhase(tokenized, converted),
            cpu_conversion=DensePhase(converted, finished),
        )
