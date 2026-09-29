# LawCast Semantic Search (사이드 프로젝트)

LawCast의 법률안(입법예고) 데이터를 대상으로 **의미(시맨틱) 검색** 파이프라인을 구축한 독립 사이드 프로젝트입니다.
`backend/`, `frontend/` 와는 분리된 디렉토리이며 LawCast 코드베이스를 수정하지 않습니다.

4개 단계는 각각 독립적으로 실행 가능한 스크립트이며, 앞 단계의 산출물(artifact)을 다음 단계가 읽는 방식입니다.

```
(0) 샘플 추출            (1) 전처리 + 청킹       (2) 토크나이징 + 임베딩
backend/lawcast.db ───► sample_notices.jsonl ──► chunks.jsonl ──► embeddings.npz
                                                                        │
                     (4) 질의 처리 + 유사도        (3) FAISS 인덱싱        ▼
                     query ────────────────────► faiss.index ◄──── 73 x 768 벡터
                                                id_map.json
```

## LawCast 프로젝트 구조 분석 (참고)

| 영역 | 위치 | 내용 |
| ---- | ---- | ---- |
| 데이터 저장 | `notice_archives` (SQLite, TypeORM) | 법률안 2만여 건: `noticeNum`, `subject`, `committee`, `proposalReason`, `aiSummary` 등 |
| 수집 | `backend/src/modules/crawling/` | 국회 의안시스템(NSM) 브라우저 크롤링 → 아카이브 동기화 |
| 검색 | `notice_archives_fts` (SQLite FTS5) | `subject`/`committee`/`proposalReason` 대상 키워드 전문검색 (트리거 동기화) |
| API | `backend/src/controllers/api.controller.ts` | `/api/notices/recent`, `/notices/archive`, `/notices/search` 등 |
| 요약 | `backend/src/modules/ollama/` | Ollama 기반 AI 요약 (`aiSummary`) |

현재 검색은 **FTS5 키워드 매칭**뿐이라 "세입자 보호" 같은 질의가 "임차인"이 들어간 법률안을 찾지 못합니다.
이 사이드 프로젝트는 그 위에 올릴 의미 검색 레이어의 프로토타입입니다.

핵심 텍스트 소스는 `proposalReason`(제안이유 및 주요내용)이며, `제안이유` / `주요내용` 헤더와
`가. 나. 다.` 항목이 개행으로 구분된 구조화된 한국어 법률 텍스트입니다.

## 임베딩 모델 선택: `jhgan/ko-sbert-sts`

한국어 특화 임베딩 모델 후보를 검토한 결과 **`jhgan/ko-sbert-sts`** 를 선택했습니다.

| 후모델 | 차원 | 크기 | 특징 | 탈락 이유 |
| ------ | ---- | ---- | ---- | --------- |
| **jhgan/ko-sbert-sts** | 768 | ~420MB | 한국어 Sentence-BERT, KorSTS(한국어 의미적 유사도) 파인튜닝, sentence-transformers 네이티브 지원 | **채택** |
| KoSimCSE (BM-K/KoSimCSE-roberta-multitask) | 768 | ~420MB | KorNLI+KorSTS 멀티태스크 | sentence-transformers 비표준 로딩(커스텀 코드 필요), CPU 프로토타입에 불필요한 복잡도 |
| upskyy/bge-m3-korean, dragonkue/bge-m3-ko | 1024 | ~2.2GB | BGE-M3 기반, 최신·고성능 | 다운로드/추론 비용이 프로토타입 대비 과도 |
| KURE (nlpai-lab) | 768 | ~1GB | 2025년 한국어 검색 특화 SOTA급 | 검색 특화지만 무거우며, 본 파이프라인의 유사도 기반 demo 목적에는 ko-sbert-sts로 충분 |

**채택 근거**

1. **한국어 특화 학습**: KorSTS(한국어 의미적 유사도) 데이터로 파인튜닝되어 본 파이프라인의 핵심인
   "질의-문서 유사도 계산"과 목적이 정확히 일치합니다.
2. **실제 다운로드·사용 검증**: HuggingFace에서 정상 다운로드되어 모든 단계에서 실제로 사용했습니다
   (`sentence-transformers 6.x` 네이티브 포맷, 커스텀 코드 없음).
3. **적당한 크기**: BERT 계열(klue/bert-base) 768-dim, ~420MB — CPU(macOS arm64)에서 수십 개 청크 임베딩이 수 초 내 완료.
4. **한계 명시**: `max_seq_length=128` 토큰이라 긴 문서에는 부적합 — 그래서 청크 크기를 200자로
   캘리브레이션해 모든 청크가 모델 윈도우에 들어가게 했습니다(아래 참조).

환경변수 `LAWCAST_SEMANTIC_MODEL`로 다른 모델(예: `upskyy/bge-m3-korean`)로 손쉽게 교체 가능합니다.

## 디렉토리 구조

```
semantic-search/
├── config.py                     # 모델명/경로/청킹 파라미터 (중앙 설정)
├── lawcast_semantic/             # 파이프라인 라이브러리 (각 단계별 모듈)
│   ├── preprocess.py             # 1a. 텍스트 정규화 + 섹션(제안이유/주요내용) 감지
│   ├── chunking.py               # 1b. 섹션 인식 청킹 (문장 경계 + 오버랩)
│   ├── embedding.py              # 2.  KoreanEmbedder: 토크나이징 + 임베딩 추출
│   ├── indexing.py               # 3.  VectorIndex: FAISS 생성/저장/로드/검색
│   └── search.py                 # 4.  SemanticSearcher: 질의 처리 + 유사도 랭킹
├── scripts/                      # 독립 실행 CLI (각 단계)
│   ├── extract_sample_data.py    # 0.  LawCast DB에서 샘플 추출
│   ├── 01_preprocess_chunk.py    # 1.  전처리 + 청킹
│   ├── 02_extract_embeddings.py  # 2.  토크나이징 + 임베딩
│   ├── 03_build_index.py         # 3.  FAISS 인덱싱 + 저장
│   └── 04_search.py              # 4.  질의 + 유사도 검색
├── data/
│   └── sample_notices.jsonl      # 샘플 데이터 (backend/lawcast.db에서 추출, 12건)
├── artifacts/                    # 단계별 산출물 (gitignore 대상)
│   ├── chunks.jsonl              #   73 청크 (stage 1 출력)
│   ├── embeddings.npz            #   73 x 768 float32 (stage 2 출력)
│   ├── faiss.index               #   IndexFlatIP (stage 3 출력)
│   └── id_map.json               #   행 -> chunk_id 매핑 (stage 3 출력)
├── tests/                        # pytest 29종 (모델 다운로드 없이 동작)
├── requirements.txt
└── .venv/                        # 로컬 가상환경 (직접 생성)
```

## 설치

```bash
cd semantic-search
python -m venv .venv          # Python 3.13 (macOS arm64) 기준
.venv/bin/pip install -r requirements.txt
```

모델(`jhgan/ko-sbert-sts`, ~420MB)은 **최초 실행 시 HuggingFace에서 자동 다운로드**됩니다.

## 파이프라인 실행 (순서대로)

각 스크립트는 어느 디렉토리에서든 실행 가능하며(경로가 `config.py` 기준 절대 해석),
이전 단계의 artifact를 기본 입력으로 읽습니다.

```bash
# 0. LawCast SQLite DB에서 샘플 법률안 추출 (읽기 전용, 기본 소스: ../backend/lawcast.db)
.venv/bin/python scripts/extract_sample_data.py

# 1. 전처리 + 청킹  -> artifacts/chunks.jsonl
.venv/bin/python scripts/01_preprocess_chunk.py

# 2. 토크나이징 + 임베딩 추출 -> artifacts/embeddings.npz
.venv/bin/python scripts/02_extract_embeddings.py

# 3. FAISS 인덱싱 + 저장 -> artifacts/faiss.index + id_map.json
.venv/bin/python scripts/03_build_index.py

# 4. 질의 처리 + 유사도 계산
.venv/bin/python scripts/04_search.py --query "국가 연구시설과 장비의 공동 활용" --k 3
.venv/bin/python scripts/04_search.py --query "..." --query "..." --json  # 다중 질의/JSON 출력
```

## 단계별 상세

### Stage 1 — 전처리 및 청킹

- **전처리** (`preprocess.py`): NFC 정규화, HTML 태그/제로폭 문자 제거, 가로 공백 정리.
  **개행은 보존**합니다 — `proposalReason`의 줄바꿈은 문단 구조이며 의미를 가집니다.
- **섹션 감지**: `제안이유`, `제안이유 및 주요내용`, `주요내용` 등 독립 헤더 줄 기준으로 섹션 분할.
- **청킹** (`chunking.py`):
  1. 섹션별 문단을 문장 경계(`.!?`, 개행) 기준 유닛으로 분할 (유닛당 최대 200자)
  2. 유닛을 200자 이하 청크로 그리디 패킹 (개행으로 결합해 문단 구조 유지)
  3. 청크 경계에 이전 청크의 꼬리(~50자)를 **오버랩**해 문맥 손실 방지
  4. `proposal_reason`가 빈 공고는 제목 기반 단일 청크로 폴백 (모든 공고가 검색 가능)
- 청크 스키마: `chunk_id`, `notice_num`, `subject`, `committee`, `section`, `chunk_index`,
  `text`, `char_count`, `sentence_count`

### Stage 2 — 토크나이징 + 임베딩 추출

- `KoreanEmbedder`가 `jhgan/ko-sbert-sts` 로드 (sentence-transformers).
- **토크나이징 단계**: 각 청크를 모델 토크나이저로 변환해 토큰 수/절단 여부를 리포트.
  최대 시퀀스 길이(128 토큰)를 초과해 잘리는 청크가 **0개**임을 출력으로 검증합니다
  (`truncated_count: 0`, 최대 113 토큰).
- **임베딩 추출**: 768-dim float32, L2 정규화(코사인 유사도용).
  청킹 파라미터(200자)는 한국어 법률 텍스트의 토크나이즈 비율(~1.9자/토큰)과
  128 토큰 윈도우를 맞춰 캘리브레이션된 값입니다.

### Stage 3 — FAISS 인덱싱 및 저장

- `faiss.IndexFlatIP` + L2 정규화 벡터 = **코사인 유사도** (정밀 탐색).
- 현재 규모(수만 건 예상)에서는 정밀 탐색이 충분하며, 대규모화 시 IVF/PQ로 교체 가능.
- `faiss.index`(바이너리)와 `id_map.json`(행 → chunk_id)을 함께 저장해 청크 메타데이터와 연결.

### Stage 4 — 질의 처리 + 유사도 계산

- 질의도 문서와 동일한 정규화(`normalize_text`)를 거쳐 동일 모델로 임베딩 (도메인 일관성).
- FAISS top-k 탐색 후 코사인 유사도 점수와 청크 메타데이터(공고 번호, 제목, 섹션, 원문 발췌)를 랭킹 출력.

## 데이터 원천 (`backend/lawcast.db`)

샘플 추출의 기본 소스는 **`backend/lawcast.db`** (백엔드 개발 DB)입니다. 두 DB를 실제로
비교 검증한 결과:

| 항목 | `lawcast_prod.db` (루트) | `backend/lawcast.db` |
| ---- | ------------------------ | -------------------- |
| `notice_archives` 행 수 | 20,177 | 20,895 (상위집합: 신규 718건) |
| 스키마 | 구버전 (스크린샷 캡처 상태 컬럼 없음) | 최신 (`screenshot_capture_status/error` 등 포함) |
| 공유 20,177건 중 내용 불일치 | — | 29건 (소스 갱신 반영) |
| 샘플 대상 12건 본문 동일성 | — | 12/12 동일 |

`lawcast_prod.db`는 루트의 프로덕션 스냅샷이며 `--db ../lawcast_prod.db`로 여전히 사용할
수 있습니다. 샘플 선택은 길이 구간별 최신순이므로 데이터가 갱신되면 샘플 대상 공고가
바뀔 수 있습니다.

## 실행 결과 예시

```
query: 국가 연구시설과 장비의 공동 활용
  1. score=0.5192 notice=2221463 [제안이유] 국가연구시설·장비의 관리 및 활용 등에 관한 법률안 (송기헌의원 등 10인)
     그러나 표준지침에는 이를 지키지 아니한 보유기관을 제재할 수단이 없어 연구시설ㆍ장비를 도입하고도 ...
  2. score=0.4774 notice=2221463 [제안이유] 국가연구시설·장비의 관리 및 활용 등에 관한 법률안 (송기헌의원 등 10인)
  3. score=0.4461 notice=2221463 [주요내용] 국가연구시설·장비의 관리 및 활용 등에 관한 법률안 (송기헌의원 등 10인)
```

질의어 "공동 활용"이 원문에 직접 없어도 관련 청크가 최상위로 검색됩니다 — 키워드 검색(FTS5)과
차별화되는 의미 검색의 동작을 보여줍니다. 참고: 임베딩 대상은 본문(`proposalReason`) 청크이며
공고 제목은 메타데이터로만 사용되므로, 제목에만 등장하는 어휘 질의는 본문 서술 기반 검색보다
성능이 낮을 수 있습니다.

## 린트 / 포맷

ruff로 코드 품질을 관리합니다 ([ruff.toml](ruff.toml): E/F/W/I/UP 규칙, 100자, single quote).

```bash
.venv/bin/ruff check --fix .        # 린트 (자동 수정)
.venv/bin/ruff format .             # 포맷
.venv/bin/ruff check . && .venv/bin/ruff format --check .   # 통과 검증
```

## 테스트

```bash
.venv/bin/python -m pytest tests/ -q    # 29 passed
```

- `test_preprocess.py` — 정규화(개행 보존, NFC, HTML 제거), 섹션 감지
- `test_chunking.py` — 청크 크기 상한, 오버랩, 섹션 전파, chunk_id 유일성, 중복 notice_num 거부, 폴백
- `test_indexing_search.py` — FAISS 저장/로드 라운드트립, 랭킹 결정성(동점 tie-break), 산출물
  지문·모델 불일치 거부 (스텁 임베더로 모델 다운로드 없이 실행)
- `test_entrypoints.py` — 실제 스크립트 서브프로세스 실행: 빈 입력·빈 공고·잘못된 `--k` 처리

## 설계 결정 및 한계

- **청킹은 문자 기준, 토큰은 검증**: 토크나이저는 stage 2의 책임이므로 stage 1은 문자/문장 단위로
  동작하되, stage 2 출력의 `truncated_count`가 청킹 파라미터의 가드레일 역할을 합니다.
- **모델 한계**: ko-sbert-sts는 128 토큰 윈도우의 2021년 모델. 최신 KURE나 BGE-m3-ko로
  `LAWCAST_SEMANTIC_MODEL`만 바꿔 교체 가능하며, 그 경우 청크 크기를 키워도 됩니다.
- **하이브리드 검색 미구현**: FTS5 키워드 검색 + 벡터 검색의 하이브리드(리랭킹)는 확장 과제.
- **증분 인덱싱 미구현**: 전체 재구축 방식. LawCast 운영 연동 시 `notice_archives` 변경 이벤트와
  연동한 증분 업데이트가 필요.

## 확장 아이디어

1. LawCast DB 전체(2만+ 건) 인제스트 및 배치 임베딩 (GPU/MPS 가속)
2. `notice_archives_fts` + FAISS 하이브리드 검색 (BM25 + 코사인 리랭킹)
3. `aiSummary`(Ollama 요약)까지 청킹 대상에 포함해 검색 품질 향상
4. 백엔드에 `SemanticSearchService`로 통합 (FAISS 파일 또는 sqlite-vec 활용)
5. KURE-v1 / bge-m3-ko로 모델 업그레이드 및 KorSTS 벤치마크로 품질 비교
