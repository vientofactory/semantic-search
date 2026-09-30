# LawCast Semantic Search

LawCast의 법률안(입법예고) 데이터를 대상으로 의미(시맨틱) 검색 파이프라인을 구축한 독립 사이드 프로젝트입니다.
`backend/`, `frontend/` 와는 분리된 디렉토리이며 LawCast 코드베이스를 수정하지 않습니다.

4개 단계는 각각 독립적으로 실행 가능한 스크립트이며, 앞 단계의 산출물(artifact)을 다음 단계가 읽는 방식입니다.

```mermaid
flowchart LR
    db[("backend/lawcast.db")] -->|1. 전처리 + 청킹| chunks["chunks.jsonl<br/>96,754 청크"]
    db -.->|0. 샘플 추출(평가용)| sample["sample_notices.jsonl"]
    chunks -->|2. 토크나이징 + 임베딩| emb["embeddings.npz<br/>96,754 x 1,024 float32"]
    emb -->|3. FAISS 인덱싱| idx["faiss.index + id_map.json"]
    query(["query"]) --> search["4. 질의 처리 + 유사도"]
    idx --> search
    search --> result["top-k 랭킹"]
```

(2)와 (4)는 동일 모델(`nlpai-lab/KURE-v1`)로 임베딩하며, 산출물 수치는 전체 코퍼스
(공고 20,919건) 기준입니다. 샘플 추출(0)은 평가셋 작성용입니다.

## LawCast 프로젝트 구조 분석 (참고)

| 영역        | 위치                                        | 내용                                                                                   |
| ----------- | ------------------------------------------- | -------------------------------------------------------------------------------------- |
| 데이터 저장 | `notice_archives` (SQLite, TypeORM)         | 법률안 2만여 건: `noticeNum`, `subject`, `committee`, `proposalReason`, `aiSummary` 등 |
| 수집        | `backend/src/modules/crawling/`             | 국회 의안시스템(NSM) 브라우저 크롤링 → 아카이브 동기화                                 |
| 검색        | `notice_archives_fts` (SQLite FTS5)         | `subject`/`committee`/`proposalReason` 대상 키워드 전문검색 (트리거 동기화)            |
| API         | `backend/src/controllers/api.controller.ts` | `/api/notices/recent`, `/notices/archive`, `/notices/search` 등                        |
| 요약        | `backend/src/modules/ollama/`               | Ollama 기반 AI 요약 (`aiSummary`)                                                      |

현재 검색은 FTS5 키워드 매칭뿐이라 "세입자 보호" 같은 질의가 "임차인"이 들어간 법률안을 찾지 못합니다.
이 사이드 프로젝트는 그 위에 올릴 의미 검색 레이어의 프로토타입입니다.

핵심 텍스트 소스는 `proposalReason`(제안이유 및 주요내용)이며, `제안이유` / `주요내용` 헤더와
`가. 나. 다.` 항목이 개행으로 구분된 구조화된 한국어 법률 텍스트입니다.

## 임베딩 모델 선택: `nlpai-lab/KURE-v1`

최초 채택은 `jhgan/ko-sbert-sts`(유사도 회귀 모델)였으나, 홀드아웃 평가 A/B에서 검색 특화 모델
`nlpai-lab/KURE-v1`이 우위를 보여 교체했습니다.

| 후보 모델                      | 차원  | 최대 토큰  | 크기                               | 특징 / 판정                                                                                                                                            |
| ------------------------------ | ----- | ---------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **nlpai-lab/KURE-v1 (채택)**   | 1,024 | 8,192 토큰 | 가중치 ~2.2GB (HF 캐시 4.3GB 측정) | bge-m3 기반 한국어-영어 검색 특화, 2M 쿼리-문서-하드네거티브 학습, 모델 카드 기준 한국어 검색 벤치마크 평균 Recall@1 0.526(1위, 법률 도메인 포함), MIT |
| jhgan/ko-sbert-sts (이전 채택) | 768   | 128 토큰   | ~420MB                             | klue/bert-base 기반, KorSTS 파인튜닝 유사도 모델                                                                                                       |
| nlpai-lab/KoE5                 | 1,024 | 512 토큰   | ~2.2GB                             | multilingual-e5-large 파인튜닝 — 미검증                                                                                                                |
| dragonkue/BGE-m3-ko 등         | 1,024 | 8,192 토큰 | ~2.2GB                             | bge-m3 한국어 파인튜닝 계열 — 미검증                                                                                                                   |

**채택 근거** (차원·토큰·파라미터는 로드된 모델에서 실측)

1. **목적 정합**: 리트리벌(질의→문서 순위) 목표로 학습된 모델이라 의미 검색과 목적이 일치합니다.
2. **사용 검증**: HuggingFace에서 정상 다운로드되어 전 파이프라인에서 사용했습니다
   (`sentence-transformers 6.x` 네이티브 포맷, 커스텀 코드 없음). 실측 568M 파라미터.
3. **A/B 채택 규칙 충족**: 홀드아웃 개선 + 제목/본문 세트 무회귀 (아래 결과표).
   부수 효과로 8,192 토큰 윈도우에서 청크 절단 0건 (이전 모델은 4건).

기준선 재현: `LAWCAST_SEMANTIC_MODEL=jhgan/ko-sbert-sts`로 파이프라인을 재실행하면 이전 수치가
재현됩니다.

## 디렉토리 구조

```
semantic-search/
├── lawcast_semantic/             # 파이프라인 라이브러리 (자체 완결, 루트 모듈 의존 없음)
│   ├── config.py                 #     설정·아티팩트 경로 단일 소유 (env: LAWCAST_SEMANTIC_*)
│   ├── datasource.py             # 0.  학습 데이터 소스: DB `notice_archives.proposalReason` (읽기 전용)
│   ├── preprocess.py             # 1a. 텍스트 정규화 + 섹션(제안이유/주요내용) 감지
│   ├── chunking.py               # 1b. 섹션 인식 청킹 (문장 경계 + 오버랩)
│   ├── embedding.py              # 2.  KoreanEmbedder: 토크나이징 + 임베딩 추출
│   ├── indexing.py               # 3.  VectorIndex: FAISS 생성/저장/로드/검색
│   ├── search.py                 # 4.  SemanticSearcher: 질의 처리 + 유사도 랭킹
│   └── evaluation.py             #     검색 품질 지표 (notice_rank, recall@k/MRR)
├── scripts/                      # 독립 실행 CLI (각 단계)
│   ├── extract_sample_data.py    # 0.  LawCast DB에서 샘플 추출
│   ├── 01_preprocess_chunk.py    # 1.  전처리 + 청킹
│   ├── 02_extract_embeddings.py  # 2.  토크나이징 + 임베딩
│   ├── 03_build_index.py         # 3.  FAISS 인덱싱 + 저장
│   ├── 04_search.py              # 4.  질의 + 유사도 검색
│   ├── 05_evaluate.py            #     정량 평가 (recall@k / MRR)
│   └── 06_incremental_update.py  # 5.  증분 갱신 (신규/수정/삭제 공고만 반영)
├── data/
│   ├── sample_notices.jsonl       # 평가 코퍼스 (backend/lawcast.db에서 추출, 300건)
│   ├── eval_holdout_queries.jsonl # 홀드아웃 평가셋 (사용자 스타일, 24쌍)
│   └── eval_queries.jsonl         # 제목/본문 평가셋 (질의 -> 정답 공고, 16쌍)
├── artifacts/                    # 단계별 산출물 (gitignore 대상, 전체 코퍼스 기준)
│   ├── chunks.jsonl              #   96,754 청크 (stage 1 출력, 54 MiB)
│   ├── embeddings.npz            #   96,754 x 1,024 float32 (stage 2 출력, 352 MiB)
│   ├── faiss.index               #   IndexFlatIP (stage 3 출력, 378 MiB)
│   ├── id_map.json               #   행 -> chunk_id 매핑 (stage 3 출력)
│   └── backup-sample-300/        #   300건 샘플 시대 산출물 백업 (아래 교체 규칙 참조)
├── service/                      # 온라인 어댑터: 백엔드 연동용 HTTP 사이드카
│   └── app.py                    #   FastAPI (/health, /search) — 엔진은 백그라운드 1회 로딩
├── tests/                        # pytest 53종 (모델 다운로드 없이 동작)
│   └── conftest.py               #   테스트 스위트 공용 부트스트랩 (sys.path 1곳에서만 조정)
├── requirements.txt
└── .venv/                        # 로컬 가상환경 (직접 생성)
```

## 설치

```bash
cd semantic-search
python -m venv .venv          # Python 3.13 (macOS arm64) 기준
.venv/bin/pip install -r requirements.txt
```

모델(`nlpai-lab/KURE-v1`, 가중치 ~2.2GB)은 최초 실행 시 HuggingFace에서 자동 다운로드됩니다.

## 파이프라인 실행 (순서대로)

각 스크립트는 어느 디렉토리에서든 실행 가능하며(경로가 `lawcast_semantic/config.py` 기준 절대 해석),
이전 단계의 artifact를 기본 입력으로 읽습니다.

```bash
# 0. LawCast SQLite DB에서 평가 코퍼스 추출 (읽기 전용, 기본 소스: ../backend/lawcast.db)
.venv/bin/python scripts/extract_sample_data.py --per-bucket 100   # 300건 (길이 구간별 100건)

# 1. 전처리 + 청킹  -> artifacts/chunks.jsonl (JSONL 스냅샷 기준)
.venv/bin/python scripts/01_preprocess_chunk.py
#    또는 DB에서 직접 학습: notice_archives.proposalReason 전체를 청킹
.venv/bin/python scripts/01_preprocess_chunk.py --db ../backend/lawcast.db

# 2. 토크나이징 + 임베딩 추출 -> artifacts/embeddings.npz
.venv/bin/python scripts/02_extract_embeddings.py

# 3. FAISS 인덱싱 + 저장 -> artifacts/faiss.index + id_map.json
.venv/bin/python scripts/03_build_index.py

# 4. 질의 처리 + 유사도 계산
.venv/bin/python scripts/04_search.py --query "국가 연구시설과 장비의 공동 활용" --k 3
.venv/bin/python scripts/04_search.py --query "..." --query "..." --json  # 다중 질의/JSON 출력

# 5. 증분 갱신: 신규/수정/삭제 공고만 반영 (전체 재구축 없이)
.venv/bin/python scripts/06_incremental_update.py --db ../backend/lawcast.db
.venv/bin/python scripts/06_incremental_update.py --db ../backend/lawcast.db --plan-only  # 변경 계획만 JSON으로
```

## 아티팩트 교체 규칙 (샘플 ↔ 전체 코퍼스)

`artifacts/`는 항상 **마지막에 완주한 파이프라인(0/1→3)의 산출물 한 세트**를 나타냅니다.

- **일관성 원칙**: 파이프라인을 다시 돌리면 chunks → embeddings → faiss/id_map이 한 세트로
  재생성됩니다. `embeddings.npz`이 보관하는 `chunks_fingerprint`(청크 id + 임베딩 입력의 해시)와
  `model_name`이 청크·모델과 자동 정합되므로, `SemanticSearcher.load`가 섞인 세트(예: 새 청크에
  옛 임베딩)를 ValueError로 거부합니다. **파일을 손으로 부분 교체하는 것은 항상 금지**이며,
  일관된 세트를 만드는 방법은 두 가지입니다: 1→3 전체 재구축, 또는 아래 증분 갱신.
- **증분 갱신(운영 권장)**: 신규·수정·삭제 공고만 반영하려면 `scripts/06_incremental_update.py`를
  사용합니다. 산출물은 전체 재구축과 동일하며(검증됨), 모델 교체는 명시적 오류로 거부하고,
  갱신 중 크래시는 재실행만으로 복구됩니다. 설계 결정·검증 기록은
  `agent_memories/07-incremental-indexing/plan.md`가 단일 소유처입니다.
- **백업**: 300건 샘플 시대 산출물은 `artifacts/backup-sample-300/`에 있습니다(로컬 전용,
  gitignore). 샘플로 되돌리려면 백업 파일을 `artifacts/` 최상위로 복사하거나, 스크립트의
  `--chunks/--embeddings/--id-map` 인자로 백업 파일을 직접 지정합니다.
- **평가셋 주의**: `data/eval_*_queries.jsonl`의 정답 공고는 300건 샘플 코퍼스 기준입니다.
  전체 코퍼스(2만+ 건)에서는 주제 인접 공고가 다수 경합하므로 평가 지표는 **동일 코퍼스
  안에서만 비교**해야 합니다(샘플 지표 ↔ 전체 지표 직접 비교 금지).

## 백엔드 연동: HTTP 사이드카 (`service/app.py`)

LawCast 백엔드(NestJS)는 Python을 직접 임포트할 수 없으므로, 이 프로젝트는 FastAPI 사이드카로
배포되고 백엔드는 HTTP로 호출합니다 (Ollama 연동과 동일한 패턴). 백엔드 측 엔드포인트는
`GET /api/notices/semantic-search?query=...&k=...`이며 사이드카 미실행·모델 로딩 실패 시
기존 키워드 검색으로 자동 폴백합니다 (`mode: keyword_fallback`).

```bash
# 사이드카 실행 (semantic-search/ 디렉토리에서, 포트 8300)
.venv/bin/python -m uvicorn service.app:app --host 127.0.0.1 --port 8300

# 상태 확인 (엔진 로딩 중/실패/준비 완료를 status로 리포트)
curl http://127.0.0.1:8300/health

# 직접 검색
curl 'http://127.0.0.1:8300/search?query=임대차 계약에서 세입자 보호&k=3'
```

- `GET /health` → `{status: loading|ready|failed, model, indexedChunks, error}`
- `GET /search?query=&k=` → 청크 랭킹 결과. query 1~500자(공백만은 400), k 1~50 (기본 5).
  엔진 로딩 중/로딩 실패 시 503 + 사유. 모델은 시작 시 백그라운드에서 1회 로딩되며
  요청을 블로킹하지 않고, 로딩 실패는 프로세스 재시작 전까지 유지됩니다.
- 백엔드 설정: `SEMANTIC_SEARCH_ENABLED` / `SEMANTIC_SEARCH_API_URL`(기본
  `http://127.0.0.1:8300`) / `SEMANTIC_SEARCH_TIMEOUT` (기본 10초).

## 라이브러리 구조 및 데이터 흐름 (리팩터 기록)

`lawcast_semantic/`는 자체 완결적인 라이브러리로, 백엔드 연동 시 아래처럼 임포트해 사용합니다.
설정·아티팩트 경로는 `lawcast_semantic/config.py`가 단일 소유이며, 루트 레벨 설정 모듈은 없습니다.

```python
from lawcast_semantic import KoreanEmbedder, SemanticSearcher  # 공개 API (지연 로딩)

embedder = KoreanEmbedder()  # 기본값: config.MODEL_NAME / config.DEVICE
searcher = SemanticSearcher.load(embedder)  # 기본 경로: config의 artifacts/*
results = searcher.search('세입자 보호', k=5)  # 산출물 지문/모델 불일치 시 ValueError
```

- **책임 분리**: `datasource`(데이터) / `preprocess`·`chunking`(전처리·청킹) / `embedding`(엔진) /
  `indexing`(인덱스) / `search`(조회) / `evaluation`(지표) / `config`(설정) — 단방향 의존 DAG
  (`config` ← 각 모듈 ← `search`), 순환 없음.
- **상태 소유**: 공고 레코드 스키마·JSONL은 `datasource`, 청크 레코드 스키마·JSONL·지문은
  `chunking`, 모델 런타임은 `KoreanEmbedder`, 인덱스 런타임은 `VectorIndex`, 로드된 조회
  상태(artifact 검증 포함)는 `SemanticSearcher`가 소유.
- **CLI/라이브러리 분리**: `scripts/`는 argparse·출력(프레젠테이션)만 담당하고 라이브러리는
  CLI를 모릅니다. `__init__`의 공개 API는 PEP 562 지연 로딩이라 가벼운 모듈
  (`datasource` 등) 임포트가 torch/faiss를 로드하지 않습니다(테스트로 고정).
- **데이터 흐름**: `datasource` → `preprocess`/`chunking` → `embedding` → `indexing` → `search`.
  질의는 `search` → `normalize_text` → `embed_query` → FAISS top-k.

## 단계별 상세

### Stage 0 — 학습 데이터 소스 (`datasource.py`)

- 학습 코퍼스의 단일 원천은 DB `notice_archives.proposalReason` 컬럼입니다. `source_html`
  스냅샷은 일절 읽지 않으며 HTML에서 텍스트를 추출하지 않습니다.
- `load_notices_from_db`: DB에서 공고 레코드를 직접 로드(`--db` 학습 경로)
- `extract_sample`: `proposalReason` 길이 구간별 샘플링 (스트래티파이드)
- `load_notices_jsonl` / `write_notices_jsonl`: 동일 레코드의 JSONL 스냅샷 입출력 (오프라인 재현용)
- 전처리(정규화)는 preprocess.py의 책임이며, HTML 태그 제거는 잔여 마크업 가드일 뿐입니다.

### Stage 1 — 전처리 및 청킹

- **전처리** (`preprocess.py`): NFC 정규화, 잔여 마크업/제로폭 문자 제거, 가로 공백 정리.
  개행은 보존합니다. `proposalReason`의 줄바꿈은 문단 구조이며 의미를 가집니다.
- **섹션 감지**: `제안이유`, `제안이유 및 주요내용`, `주요내용` 등 독립 헤더 줄 기준으로 섹션 분할.
- **청킹** (`chunking.py`):
  1. 섹션별 문단을 문장 경계(`.!?`, 개행) 기준 유닛으로 분할
  2. 유닛을 청크로 그리디 패킹 (개행으로 결합해 문단 구조 유지). 상한 200자는 **임베딩
     입력(제목+본문) 전체** 기준이며, 본문 예산은 제목 길이만큼 자동 축소됩니다.
  3. 청크 경계에 이전 청크의 꼬리(~50자)를 오버랩해 문맥 손실 방지
  4. `proposal_reason`가 빈 공고는 제목 기반 단일 청크로 폴백 (모든 공고가 검색 가능)
- 청크 스키마: `chunk_id`, `notice_num`, `subject`, `committee`, `section`, `chunk_index`,
  `text`, `char_count`, `sentence_count`

### Stage 2 — 토크나이징 + 임베딩 추출

- `KoreanEmbedder`가 `nlpai-lab/KURE-v1` 로드 (sentence-transformers).
- **임베딩 입력 구성**: 공고 제목을 컨텍스트 프리픽스로 결합합니다(`"{제목}\n{본문}"`).
  제목에만 등장하는 어휘(예: "공급망 안정화")가 벡터에 반영되어 제목 어휘 질의가 매칭됩니다.
  (`compose_embedding_text`가 단일 소스이며 청크 지문도 동일 입력 기준으로 해싱)
- **토크나이징 단계**: 이 결합 입력을 모델 토크나이저로 변환해 토큰 수/절단 여부를 리포트.
  모델 윈도우(8,192 토큰) 초과 청크를 가드레일로 출력합니다(전체 코퍼스 96,754 청크 기준
  0건, 평균 86.6 / 최대 172 토큰). 이전 모델(ko-sbert, 128 토큰)에서는 4건이 초과했습니다.
- **임베딩 추출**: 1,024-dim float32, L2 정규화(코사인 유사도용).
  청크 상한 200자는 이전 모델의 128 토큰 윈도우 캘리브레이션이지만 청크 크기 정책으로
  그대로 유지합니다 (비교 가능성).

### Stage 3 — FAISS 인덱싱 및 저장

- `faiss.IndexFlatIP` + L2 정규화 벡터 = 코사인 유사도(정밀 탐색).
- 현재 규모(96,754 벡터)에서는 정밀 탐색이 충분하며(사이드카 질의당 ~0.11초 실측),
  대규모화 시 IVF/PQ로 교체 가능.
- `faiss.index`(바이너리)와 `id_map.json`(행 → chunk_id)을 함께 저장해 청크 메타데이터와 연결.

### Stage 4 — 질의 처리 + 유사도 계산

- 질의도 문서와 동일한 정규화(`normalize_text`)를 거쳐 동일 모델로 임베딩 (도메인 일관성).
  질의에는 공고 제목이 없으므로 질의 텍스트를 그대로 임베딩하고, 제목 컨텍스트는 청크 측에서 제공합니다.
- FAISS top-k 탐색 후 코사인 유사도 점수와 청크 메타데이터(공고 번호, 제목, 섹션, 원문 발췌)를 랭킹 출력.

## 데이터 원천 (`backend/lawcast.db`)

샘플 추출의 기본 소스는 `backend/lawcast.db`(백엔드 개발 DB)입니다. 두 DB를 실제로
비교 검증한 결과:

| 항목                         | `lawcast_prod.db` (루트)              | `backend/lawcast.db`                             |
| ---------------------------- | ------------------------------------- | ------------------------------------------------ |
| `notice_archives` 행 수      | 20,177                                | 20,895 (상위집합: 신규 718건)                    |
| 스키마                       | 구버전 (스크린샷 캡처 상태 컬럼 없음) | 최신 (`screenshot_capture_status/error` 등 포함) |
| 공유 20,177건 중 내용 불일치 | —                                     | 29건 (소스 갱신 반영)                            |

`lawcast_prod.db`는 루트의 프로덕션 스냅샷이며 `--db ../lawcast_prod.db`로 여전히 사용할
수 있습니다. 샘플 선택은 길이 구간별 최신순(`--per-bucket N`으로 구간당 건수 조절)이므로
데이터가 갱신되면 샘플 대상 공고가 바뀔 수 있습니다. 현재 평가 코퍼스는
`--per-bucket 100`으로 추출한 300건입니다.

## 실행 결과 예시

```
query: 버스회사를 인수한 사모펀드가 차고지를 팔면
  1. score=0.7666 notice=2221504 [body] 여객자동차 운수사업법 일부개정법률안 (김남근의원 등 13인)
     버스회사를 인수한 사모펀드의 투자전략계획서 분석 자료에 의하면 버스회사를 인수한 사모펀드들은 차고지를 매각하여 고배당을 하고 엑시트(투자금회수)...
  2. score=0.6631 notice=2221504 [body] 여객자동차 운수사업법 일부개정법률안 (김남근의원 등 13인)
     영업이익 이상의 과도한 배당을 위한 자금 마련을 위해 자본잉여금을 배당가능한 이익잉여금으로 전환, 차고지와 충전소 등 토지매각, 고이자의 사채를...
  3. score=0.6585 notice=2221504 [body] 여객자동차 운수사업법 일부개정법률안 (김남근의원 등 13인)
     서울시는 시가지 확대로 주요 주거, 상업지역화한 버스 차고지를 지하화하고 그 지상에 행복주택, 주민문화시설 등을 건설하는 개발사업을 추진 중이거...
```

구어체 질의("차고지를 팔면")가 공고 본문의 법률 서술("차고지 매각")과 매칭되어 정답 공고가
top-3를 석권합니다. 키워드 검색(FTS5)과 차별화되는 의미 검색의 동작입니다. 주제 인접
법안이 상위에 오던 사례("국가 연구시설과 장비의 공동 활용")도 KURE-v1 교체 후 정답
1위(0.6341)로 복원됐습니다.

## 정량 평가 (recall@k / MRR)

검색 품질을 숫자로 측정합니다. 평가셋은 두 종류입니다:

| 평가셋           | 파일                                                               | 용도                                                               |
| ---------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------ |
| 홀드아웃 (24쌍)  | [data/eval_holdout_queries.jsonl](data/eval_holdout_queries.jsonl) | 대규모 코퍼스 실사용 품질 측정 (주 평가셋)                         |
| 제목/본문 (16쌍) | [data/eval_queries.jsonl](data/eval_queries.jsonl)                 | 제목 컨텍스트 임베딩 회귀 검출 (`key_phrase` 분류를 테스트가 검증) |

측정은 공고 단위입니다: 청크 랭킹에서 공고별 첫 등장 순위로 중복 제거 후 계산합니다
(`recall@k` = 정답 공고가 상위 k개 공고 안에 있는 비율, `MRR` = 첫 등장 순위 역수의 평균).

**홀드아웃 평가셋 작성 원칙**: 코퍼스 문구 유도가 아닌 실제 사용자 표현.

- **abbr** (7): 법안명 약칭 — "예보법 개정", "보이스피싱 특별법", "5·18 유공자 위탁병원 진료" 등
- **colloquial** (8): 일상 질문 — "해외직구할 때 관세 얼마나 내야 해?" 등
- **synonym** (9): 동의어·일상어 — "디자인 특허"(디자인보호법), "자영업자"(소상공인) 등
- 정답은 단일 공고이며 동일 법안 복수 개정안이 경합하면 개정 내용으로 질의를 구별합니다
  (예: "한은 전북 이전" — 형제안은 외환보유액 공표). 각 쌍에 `intent`로 근거를 기록

### 측정 환경 및 재현

- 샘플 코퍼스: `backend/lawcast.db`에서 300건(길이 구간별 최신순 100건씩) 추출, 1,973 청크
- `data/sample_notices.jsonl`이 측정 기준 스냅샷이며 재추출하면 DB 갱신으로 달라질 수 있음
- **주의**: 아래 A/B 표는 300건 샘플 코퍼스 기준입니다. 전체 코퍼스 인덱스(2026-09-30 구축)에서는
  동일 평가셋이라도 지표가 크게 달라지며(정답 주변 경합 증가), 두 결과를 직접 비교하지 말고
  코퍼스를 고정한 채로만 비교해야 합니다 (아래 "전체 코퍼스 인덱스" 절 참조)
- 위 파이프라인 실행(0→3)으로 인덱싱한 뒤 측정:

```bash
.venv/bin/python scripts/05_evaluate.py --eval data/eval_holdout_queries.jsonl  # 홀드아웃
.venv/bin/python scripts/05_evaluate.py                                        # 제목/본문 세트
```

### 결과 (2026-09-29, 300건 코퍼스) — ko-sbert-sts → KURE-v1 A/B

동일 코퍼스·동일 청크(지문 동일)로 두 모델을 측정했습니다.

**홀드아웃 (24쌍, 사용자 스타일)**

| kind       | n   | k-sbert r@1 | r@3   | r@5   | MRR   | KURE r@1  | r@3   | r@5   | MRR       |
| ---------- | --- | ----------- | ----- | ----- | ----- | --------- | ----- | ----- | --------- |
| abbr       | 7   | 0.429       | 0.857 | 0.857 | 0.602 | 0.429     | 0.571 | 0.714 | 0.521     |
| colloquial | 8   | 0.500       | 0.875 | 0.875 | 0.647 | **0.750** | 0.750 | 0.875 | **0.776** |
| synonym    | 9   | 0.667       | 0.889 | 0.889 | 0.770 | **0.778** | 0.778 | 0.889 | **0.808** |
| **all**    | 24  | 0.542       | 0.875 | 0.875 | 0.680 | **0.667** | 0.708 | 0.833 | **0.714** |

**제목/본문 세트 (16쌍, 동일 코퍼스)**

| kind    | n   | k-sbert r@1 | r@3   | r@5   | MRR   | KURE r@1  | r@3       | r@5       | MRR       |
| ------- | --- | ----------- | ----- | ----- | ----- | --------- | --------- | --------- | --------- |
| body    | 10  | 0.700       | 0.900 | 1.000 | 0.808 | **0.900** | **1.000** | **1.000** | **0.950** |
| title   | 6   | 0.667       | 0.667 | 0.833 | 0.719 | 0.667     | **0.833** | **1.000** | **0.783** |
| **all** | 16  | 0.688       | 0.812 | 0.938 | 0.775 | **0.812** | **0.938** | **1.000** | **0.887** |

**어휘 간극 미스 개별 rank**: 보이스피싱 특별법 22→3, 통화 녹음 92→86, ODA 10→54

**채택 판정**: 홀드아웃 개선 + 제목/본문 전 지표 개선 → 채택. 트레이드오프도 있습니다. 홀드아웃 recall@3/5는 하락(0.875→0.708/0.833)했고 개별 악화가 있습니다: "KIC 관련법 개정"
3→137, "예보법 개정" 2→18, "공공임대주택 공급 확대" 2→4. 구어체·동의어는 대폭 개선,
극단 약어는 악화입니다.

### 미스 원인 분석 (KURE-v1 기준 홀드아웃 미스 8건)

1. **극단 약어 지식 부족**: "KIC 관련법 개정"(rank 137), "개발원조 ODA 예산"(rank 54),
   "예보법 개정"(rank 18). 극단적 축약어는 서브워드로 파편화되고 모델 지식도 부족
2. **법률 ↔ 일상어 간극 잔존**: "통화 녹음해서 증거로 써도 되나요?"(rank 86). 정답은
   '감청·통신비밀' 사용. 동의어 지식("보이스피싱" 22→3)은 개선됐지만 완전하지 않음
3. **형제안·주제 인접 경합**: "한은 전북 이전"(4위), "공공임대주택 공급 확대"(4위) 등
   같은 발의자 묶음·인접 주제 법안과의 경합
4. **단일 벡터 희석**: "공급망 안정화 지원"(5위). 제목 매칭이 본문 어휘에 희석

파이프라인 버그가 아닌 모델·아키텍처 한계입니다.

### 실험 기록: BM25 + RRF 하이브리드 (채택하지 않음, 2026-09-29)

문자 bigram BM25(형태소 분석기 없음)를 벡터 검색과 Reciprocal Rank Fusion으로 융합해
측정했으나 홀드아웃에서 회귀를 보여 전량 되돌렸습니다(모듈·산출물 삭제). 민감도
스윕(가중치 0.25~2.0 × RRF k 10~100, 20구성) 전 구성이 홀드아웃 recall@1 0.458~0.500으로
기준선(0.542) 미만이었고, 제목/본문 세트만 개선(최대 0.812)되는 교환 관계였습니다.

| 세트           | 지표           | 벡터 전용         | 하이브리드 (동일 가중, k=60) |
| -------------- | -------------- | ----------------- | ---------------------------- |
| 홀드아웃 (24)  | recall@1 / MRR | **0.542 / 0.680** | 0.500 / 0.630                |
| 제목/본문 (16) | recall@1 / MRR | 0.688 / 0.775     | 0.750 / 0.849                |

- 원인: 사용자 스타일 질의는 표면 어휘가 거의 없어 bigram 매칭이 랭킹을 오염
  ("예보법 개정" 2→9, "시골에 병원이 너무 없어요" 3→18). 어휘 간극 미스도 보이스피싱
  22→7만 개선되고 통화 녹음 92→125, ODA 10→14는 악화
- 기록을 남기는 이유: 재도입 시에는 plain bigram RRF가 아니라 질의 확장·리랭킹과 함께 검토

### 전체 코퍼스 인덱스 구축 기록 (2026-09-30)

DB 전체(`notice_archives.proposalReason`)로 인덱스를 구축해 `artifacts/`를 교체했습니다.
샘플 산출물은 `artifacts/backup-sample-300/`에 백업되어 있습니다 (교체 규칙은 "아티팩트
교체 규칙" 절 참조).

| 항목              | 값                                                        |
| ----------------- | --------------------------------------------------------- |
| 공고 / 청크       | 20,919건 / 96,754 청크 (평균 129자)                       |
| 토큰              | 총 8.38M, 평균 86.6 / 최대 172, 모델 윈도우 초과 0건      |
| 임베딩 매트릭스   | 96,754 x 1,024 float32 (L2 정규화, norm 1.0)              |
| 소요 시간         | stage 1 3.9초 / stage 2 39.6분(MPS) / stage 3 수 초       |
| 자원              | stage 2 peak RSS ~1.9GB, 산출물 총 ~785 MiB               |

- **디바이스 선택**: 동일 배치 벤치마크(`scripts/benchmark_device.py`, 96청크) 기준
  cpu 21.0 청크/초 vs mps 38.6 청크/초 → MPS 채택 (예상 시간 76.7분 vs 41.7분)

**전체 코퍼스 홀드아웃 (24쌍)** — `search depth: 96,754 chunks (full ranking)`

| kind       | n  | recall@1 | recall@3 | recall@5 | MRR  |
| ---------- | -- | -------- | -------- | -------- | ---- |
| abbr       | 7  | 0.000    | 0.143    | 0.286    | 0.121 |
| colloquial | 8  | 0.125    | 0.250    | 0.250    | 0.208 |
| synonym    | 9  | 0.000    | 0.333    | 0.333    | 0.175 |
| **all**    | 24 | 0.042    | 0.250    | 0.292    | 0.170 |

**전체 코퍼스 제목/본문 (16쌍)**

| kind    | n  | recall@1 | recall@3 | recall@5 | MRR  |
| ------- | -- | -------- | -------- | -------- | ---- |
| body    | 10 | 0.500    | 1.000    | 1.000    | 0.717 |
| title   | 6  | 0.167    | 0.333    | 0.500    | 0.296 |
| **all** | 16 | 0.375    | 0.750    | 0.812    | 0.559 |

**해석상 주의**: 평가셋의 정답 공고는 300건 샘플 코퍼스 기준으로 라벨링되어 있어, 2만 건
코퍼스에서는 동일 주제의 형제안·후속 개정안이 정답보다 위에 오는 경우가 많습니다(예:
"예보법 개정" rank 1,455 — 기상 예보 계열 법안 다수가 상위권). 따라서 이 표는 파이프라인의
정상 동작 확인용이며 **샘플 시대 지표와의 비교·회귀 판단에는 사용할 수 없습니다**. 품질
측정이 목표라면 전체 코퍼스 대상의 새 홀드아웃 라벨링이 선행되어야 합니다.

실검색 검증(2026-09-30): 사이드카 `/health`가 `indexedChunks: 96754`를 보고하고, `/search`
질의당 ~0.11초로 서로 다른 공고에서 결과가 폭넓게 나오며, 오타 질의("소상공없이 자업
힐들어요")도 소상공인 법안 계열을 반환합니다.400/422 검증 경로도 정상입니다.

## 린트 / 포맷

ruff로 코드 품질을 관리합니다 ([ruff.toml](ruff.toml): E/F/W/I/UP 규칙, 100자, single quote).

```bash
.venv/bin/ruff check --fix . && .venv/bin/ruff format .   # 자동 수정 + 포맷
.venv/bin/ruff check . && .venv/bin/ruff format --check . # 통과 검증
```

## 테스트

```bash
.venv/bin/python -m pytest tests/ -q    # 53 passed
```

- `test_preprocess.py` — 정규화(개행 보존, NFC, HTML 제거), 섹션 감지
- `test_datasource.py` — DB `proposalReason` 로드(`source_html` 무시 증명), 길이 구간 샘플링,
  JSONL 스냅샷 라운드트립
- `test_chunking.py` — 청크 크기 상한(제목 컨텍스트 예산 포함), 오버랩, 섹션 전파, chunk_id 유일성,
  중복 notice_num 거부, 폴백, 제목 컨텍스트 결합(`compose_embedding_text`)
- `test_indexing_search.py` — FAISS 저장/로드 라운드트립, 랭킹 결정성(동점 tie-break), 산출물
  지문·모델 불일치 거부(본문/제목 수정 포함) (스텁 임베더로 모델 다운로드 없이 실행)
- `test_evaluation.py` — recall@k/MRR 계산, 공고 중복 제거 순위, 평가셋 무결성(제목/본문 어휘
  분류가 코퍼스와 일치)
- `test_entrypoints.py` — 실제 스크립트 실행: 빈 입력·빈 공고·DB 직접 학습(`--db`)·잘못된 `--k`·빈 평가셋·
  경량 임포트 시 모델 스택 미로딩(라이브러리/CLI 분리 경계)·
  산출물 불일치의 깔끔한 error 표기 (트레이스백 없음)

## 설계 결정 및 한계

- **청킹은 문자 기준, 토큰은 검증**: 토크나이저는 stage 2의 책임이므로 stage 1은 문자/문장 단위로
  동작하되, stage 2 출력의 `truncated_count`가 청킹 파라미터의 가드레일 역할을 합니다.
- **모델 한계**: KURE-v1도 극단 약어와 일부 법률↔일상어 간극이 남아 있습니다 (미스 원인 분석 참조).
  `LAWCAST_SEMANTIC_MODEL`으로 다른 모델을 교체해 A/B할 수 있습니다.
- **측정은 회귀 검출용**: 300건/24쌍 규모이며 절대 품질 척도가 아닙니다. 제목/본문 세트의 일부
  질의("조세특례제한법 개정" 등)는 12건 코퍼스 설계 시점이라, 300건에서는 동일 법안 형제안이
  실질 정답인데 단일 정답 기준 미스로 계산됩니다.
- **중복 libomp 충돌 회피 (macOS)**: torch+faiss가 한 프로세스에 로드되면 검색 시 세그폴트하는
  환경 이슈로, 우회와 그 근거는 `lawcast_semantic/omp_env.py`가 단일 소유합니다. 두 엔진이
  공존하는 엔트리포인트(04/05 스크립트, 사이드카)가 시작 시 호출합니다.

## 확장 아이디어

1. 질의 확장(약어 사전)·경량 교차 인코더 리랭킹 — 극단 약어 미스와 recall@3 하락 공략
2. `notice_archives_fts` + 벡터 검색 하이브리드 (단순 BM25 RRF는 위 실험 기록의 회귀 근거 참고)
3. `aiSummary`(Ollama 요약)를 청킹 대상에 포함해 검색 품질 향상
4. 백엔드 `SemanticSearchService` 통합 (FAISS 파일 또는 sqlite-vec) 및 `notice_archives`
   변경 이벤트 기반 증분 인덱싱 (전체 재구축은 MPS 기준 ~40분이므로 증분화 가치 큼)
6. 미검증 후보(KoE5, BGE-m3-ko)와의 추가 A/B
