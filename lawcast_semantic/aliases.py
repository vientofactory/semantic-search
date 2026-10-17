"""Stage 4a: query-side citizen alias expansion.

Citizens refer to bills by syllable-initial acronyms and short forms that never
appear in `proposalReason` or in `subject` (`중처법`, `산안법`, `전상법`, ...).
The embedding therefore has no surface form to match and retrieval returns
either nothing or a near-namesake — measured: `중처법` MISS, `산안법` MISS,
`전상법` rank 154,
`AI 기본법` rank 8. This module owns one curated alias -> official-name mapping
and rewrites those tokens in the query before it is embedded. **The index is not
touched**: expansion moves the query vector onto the official name the corpus
actually stores, so it needs no re-embedding and no artifact change.

Rules for entries:

- The canonical value must be a bill name that exists in
  `notice_archives.subject` (prefix match), because expansion works by aligning
  the query to that surface form. A wrong or absent name is worse than no entry.
- Every entry is measured: it must either fix a failure or be neutral. An
  entry that turns a rank-1 alias into a non-rank-1 result is removed —
  `스토킹처벌법` was dropped for exactly that (1 -> 2), while short forms that
  were already rank 1 are kept only where they measured neutral.
- Alias matching is bounded: an alias only fires when it stands alone, not
  embedded inside a longer word (`중처법상` matches, `중처법률` does not).
  Common trailing particles ('은/는/이/가/을/를/의/에/도/만/상/로/과/와/에서/으로')
  are matched and re-emitted, so `중처법은 개정되나요` expands too.

Not covered here: near-namesake confusions where the query already IS the
official name (`지방의회법` vs `지방자치법`, rank 7 while the query needs no
rewriting) — those are ranking problems, not alias problems.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Alias -> official bill name fragment, exactly as the corpus stores it.
# Curated and measured: every entry below either fixed a retrieval failure or
# measured neutral (see agent_memories/20-semantic-query-quality-eval/ §9).
ALIASES: dict[str, str] = {
    # Serious industrial accidents
    '중처법': '중대재해 처벌 등에 관한 법률',
    '중대재해처벌법': '중대재해 처벌 등에 관한 법률',
    # Workplace safety
    '산안법': '산업안전보건법',
    '산재법': '산업재해보상보험법',
    # Labor / consumer / fair trade
    '근기법': '근로기준법',
    '노조법': '노동조합 및 노동관계조정법',
    '전상법': '전자상거래 등에서의 소비자보호에 관한 법률',
    '하도급법': '하도급거래 공정화에 관한 법률',
    '가맹사업법': '가맹사업거래의 공정화에 관한 법률',
    '대부업법': '대부업 등의 등록 및 금융이용자 보호에 관한 법률',
    '공정거래법': '독점규제 및 공정거래에 관한 법률',
    # Tax / finance
    '상증법': '상속세 및 증여세법',
    '자본시장법': '자본시장과 금융투자업에 관한 법률',
    # Housing / lease deposits
    '주임법': '주택임대차보호법',
    '전세사기특별법': '전세사기피해자 지원 및 주거안정에 관한 특별법',
    # Data / platforms
    '개보법': '개인정보 보호법',
    '정통망법': '정보통신망 이용촉진 및 정보보호 등에 관한 법률',
    '플랫폼법': '온라인 플랫폼 중개거래의 공정화에 관한 법률',
    'AI 기본법': '인공지능 발전과 신뢰 기반 조성 등에 관한 기본법',
    '인공지능 기본법': '인공지능 발전과 신뢰 기반 조성 등에 관한 기본법',
    # Criminal / family
    # NOTE: '스토킹처벌법' is deliberately absent - its raw form already ranks
    # the bill 1st and expanding it measured 1 -> 2, so the entry was removed.
    # Re-measured on BGE-m3-ko (full corpus, 2026-10-17): raw 13 -> expanded
    # 17, still a regression, so the exclusion stands on the new baseline too.
    '청탁금지법': '부정청탁 및 금품등 수수의 금지에 관한 법률',
    '양육비이행법': '양육비 이행확보 및 지원에 관한 법률',
    # BGE-gap abbreviations (added 2026-10-17): BGE-m3-ko still ranks these
    # 3-4 digits on the full corpus, so they are alias problems, not ranking
    # problems. Each entry measured raw -> expanded on the BGE full index:
    # `예보법` 3128 -> 48, `KIC` 2447 -> 14, `ODA`/`개발원조` 4694 -> 62,
    # `보이스피싱` 147 -> 4, `요양원` 902 -> 100.
    # Deliberately NOT added: `고준위방폐물` (74 -> 52, marginal, dict kept
    # tight) and `통화녹음` (never fires: the colloquial query continues with
    # `해서`, which is not a trailing particle - needs query rewriting, not
    # an alias entry).
    '예보법': '예금자보호법',
    'KIC': '한국투자공사법',
    'ODA': '국제개발협력기본법',
    '개발원조': '국제개발협력',
    '보이스피싱': '전기통신금융사기',
    '요양원': '노인장기요양',
}

# Trailing particles that may follow an alias without making it a different
# word ("중처법은", "산안법상"). Longest first so '으로' wins over '으'/'로'.
_PARTICLES = (
    '에서',
    '으로',
    '보다',
    '은',
    '는',
    '이',
    '가',
    '을',
    '를',
    '의',
    '에',
    '와',
    '과',
    '도',
    '만',
    '상',
    '로',
)

# Korean has no `\b`, so boundaries are explicit: an alias only matches when it
# is not glued to a longer Hangul/latin word on either side.
_BOUNDARY_LEFT = r'(?<![0-9A-Za-z가-힣])'
_BOUNDARY_RIGHT = r'(?![0-9A-Za-z가-힣])'

# Word lookup for the callback. Casefolded so latin aliases ('AI 기본법') match
# however the user capitalises them, and whitespace-stripped so a multi-word
# alias matches whether or not the user typed the space ('AI기본법').
_LOOKUP = {
    re.sub(r'\s+', '', alias).casefold(): (alias, official) for alias, official in ALIASES.items()
}


def _alias_pattern_body(alias: str) -> str:
    """Regex body for one alias; whitespace inside it matches zero-or-more.

    Lets `AI기본법` and `AI 기본법` share one entry.
    """
    return r'\s*'.join(re.escape(part) for part in alias.split())


# Longest alias first: alternation is leftmost-first, so a multi-word key would
# otherwise be partly consumed by a shorter key it contains.
_ORDERED_ALIASES = tuple(sorted(ALIASES, key=len, reverse=True))

_ALIAS_RE = re.compile(
    _BOUNDARY_LEFT
    + '(?P<alias>'
    + '|'.join(_alias_pattern_body(alias) for alias in _ORDERED_ALIASES)
    + ')'
    + '(?P<particle>'
    + '|'.join(re.escape(particle) for particle in _PARTICLES)
    + ')?'
    + _BOUNDARY_RIGHT,
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QueryExpansion:
    """Query text after alias expansion plus the aliases that fired."""

    text: str
    matched: tuple[str, ...]


def expand_query(query: str | None) -> QueryExpansion:
    """Rewrite citizen aliases in `query` to their official bill names.

    Only known aliases standing alone (optionally with a trailing particle) are
    touched, so ordinary queries pass through byte-identical with an empty
    `matched` tuple. A query with no text returns an empty expansion.
    """
    if not query:
        return QueryExpansion('', ())
    matched: list[str] = []

    def replace(match: re.Match) -> str:
        key = re.sub(r'\s+', '', match.group('alias')).casefold()
        alias, official = _LOOKUP[key]
        if alias not in matched:
            matched.append(alias)
        return official + (match.group('particle') or '')

    return QueryExpansion(_ALIAS_RE.sub(replace, query), tuple(matched))
