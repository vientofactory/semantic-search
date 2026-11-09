"""Paired relevance replay with the production encoder and fixed candidate windows."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from itertools import product
from math import comb, sqrt
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lawcast_semantic.omp_env import use_single_threaded_omp  # noqa: E402

use_single_threaded_omp()

from lawcast_semantic import config  # noqa: E402
from lawcast_semantic.aliases import expand_query  # noqa: E402
from lawcast_semantic.embedding import KoreanEmbedder  # noqa: E402
from lawcast_semantic.hybrid import filter_by_lexical_evidence  # noqa: E402
from lawcast_semantic.preprocess import normalize_text  # noqa: E402
from lawcast_semantic.relevance import DEFAULT_CUTOFFS  # noqa: E402
from lawcast_semantic.search import SemanticSearcher  # noqa: E402


def read(name):
    return [
        json.loads(line)
        for line in (config.DATA_DIR / f'{name}.jsonl').read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]


def normalized(query):
    return normalize_text(expand_query(query).text)


def cases(split):
    old = read('unrelated_queries')
    if split == 'calibration':
        return [
            dict(query=row['query'], group=row['category'], negative=row['expect'] == 'empty')
            for row in old
            if row['expect'] != 'boundary'
        ] + [
            dict(query=row['query'], group=row['kind'], targets=[row['notice_num']])
            for row in read('eval_queries')
        ]
    calibration = {normalized(row['query']) for row in old + read('eval_queries')}
    # Leave out older positives targeting the same notices used in calibration.
    calibrated_ids = {
        row['notice_num']
        for row in read('eval_queries') + read('eval_holdout_queries')
        if normalized(row['query']) in calibration
    }
    rows = [dict(row, negative=True) for row in read('eval_relevance_holdout')]
    rows.extend(
        dict(query=row['query'], group=row['kind'], targets=[row['notice_num']])
        for row in read('eval_holdout_queries')
        if row['notice_num'] not in calibrated_ids and normalized(row['query']) not in calibration
    )
    rows.extend(
        dict(query=row['query'], group=row['kind'], targets=row['relevant'])
        for row in read('eval_search_priorities')
        if row['kind'] in {'recent_content', 'law_name'}
        and normalized(row['query']) not in calibration
    )
    return rows


def notice_lists(clear, weak, k):
    strong = list(dict.fromkeys(hit.notice_num for hit in clear))
    strong_ids = set(strong)
    weak_ids = list(
        dict.fromkeys(hit.notice_num for hit in weak if hit.notice_num not in strong_ids)
    )
    return {'strong': strong[:k], 'weak': weak_ids[:k]}


def rate_ci(count, n):
    if n == 0:
        return None
    z = 1.959964
    p = count / n
    mid = (p + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0, mid - radius), 4), round(min(1, mid + radius), 4)]


def paired(rows, baseline, metric, treatment='calibrated'):
    wins, losses = 0, 0
    for row in rows:
        before = metric(row, row[baseline])
        after = metric(row, row[treatment])
        wins += after and not before
        losses += before and not after
    n = wins + losses
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(wins, losses) + 1)) / (2**n))
    return {'improved': wins, 'regressed': losses, 'exact_p': p}


def summarize(rows):
    negative = [row for row in rows if row.get('negative')]
    positive = [row for row in rows if not row.get('negative')]
    labeled = [row for row in positive if row.get('targets')]

    def rejects(_row, output):
        return not (output['strong'] or output['weak'])

    def strong_rejects(_row, output):
        return not output['strong']

    def hit(row, output):
        return bool(set(row['targets']).intersection(output['strong'] + output['weak']))

    def strong_hit(row, output):
        return bool(set(row['targets']).intersection(output['strong']))

    summary = {
        'negative': len(negative),
        'positive': len(positive),
        'positive_labeled': len(labeled),
        'target_present_in_candidates': sum(row['target_in_candidates'] for row in labeled),
    }
    for method in ('baseline', 'legacy_lexical', 'cosine_only', 'cosine_background', 'calibrated'):
        empty = sum(rejects(row, row[method]) for row in negative)
        strong_empty = sum(strong_rejects(row, row[method]) for row in negative)
        summary[method] = {
            'negative_empty': empty,
            'negative_empty_wilson95': rate_ci(empty, len(negative)),
            'negative_no_strong': strong_empty,
            'negative_no_strong_wilson95': rate_ci(strong_empty, len(negative)),
            'positive_any_served': sum(not rejects(row, row[method]) for row in positive),
            'positive_strong_served': sum(bool(row[method]['strong']) for row in positive),
            'positive_strong_hit': sum(strong_hit(row, row[method]) for row in labeled),
            'positive_either_tier_hit': sum(hit(row, row[method]) for row in labeled),
        }
    summary['paired_vs_baseline'] = {
        'negative_empty': paired(negative, 'baseline', rejects),
        'negative_no_strong': paired(negative, 'baseline', strong_rejects),
        'positive_strong_hit': paired(labeled, 'baseline', strong_hit),
        'positive_either_tier_hit': paired(labeled, 'baseline', hit),
    }
    summary['paired_vs_legacy_lexical'] = {
        'negative_empty': paired(negative, 'legacy_lexical', rejects),
        'negative_no_strong': paired(negative, 'legacy_lexical', strong_rejects),
        'positive_strong_hit': paired(labeled, 'legacy_lexical', strong_hit),
        'positive_either_tier_hit': paired(labeled, 'legacy_lexical', hit),
    }
    return summary


# Registered before scoring the new intent probes. The minimum-only policy
# keeps the current Strong boundary; the 0.49/0.51 pair is the prior ablation.
COSINE_POLICIES = {
    'baseline': (0.25, 0.45),
    'minimum_only': (0.45, 0.45),
    'equal_049': (0.49, 0.49),
    'prior_cosine_only': (0.49, 0.51),
}


def cosine_window(hits, minimum, strong, notice_k, *, widening):
    """Apply score floors and optionally replay backend-style prefix widening."""
    k = notice_k * 3
    while True:
        limit = min(k, len(hits)) if widening else len(hits)
        prefix = hits[:limit]
        clear = [hit for hit in prefix if hit.score >= strong]
        weak = [hit for hit in prefix if minimum <= hit.score < strong]
        lists = notice_lists(clear, weak, notice_k)
        if (
            not widening
            or len(lists['strong']) >= notice_k
            or len(clear) + len(weak) < k
            or k >= len(hits)
        ):
            return lists
        k = min(k * 2, len(hits))


def cosine_cases():
    rows = cases('holdout')
    new = read('eval_cosine_intent_negatives')
    seen = {normalized(row['query']) for row in rows}
    for row in new:
        key = normalized(row['query'])
        if not key or key in seen or not row['out_of_scope_intent'].strip():
            raise ValueError(f'invalid or duplicate intent probe: {row["query"]!r}')
        seen.add(key)
        rows.append(dict(row, negative=True))
    return rows


def blocked(_row, output):
    return not (output['strong'] or output['weak'])


def no_strong(_row, output):
    return not output['strong']


def target_hit(row, output):
    return bool(set(row['targets']).intersection(output['strong']))


def cluster_test(rows, policy, metric):
    """Paired sign-flip test over topic groups, rather than query variants."""
    deltas = {}
    for row in rows:
        group = row['group']
        deltas[group] = deltas.get(group, 0) + (
            int(metric(row, row[policy])) - int(metric(row, row['baseline']))
        )
    active = [delta for delta in deltas.values() if delta]
    observed = abs(sum(active))
    exceed = sum(
        abs(sum(sign * delta for sign, delta in zip(signs, active, strict=True))) >= observed
        for signs in product((-1, 1), repeat=len(active))
    )
    return {'topic_deltas': deltas, 'p': exceed / (2 ** len(active))}


def cosine_summary(rows, window):
    positive = [row for row in rows if row.get('targets')]
    hard = [row for row in rows if row.get('anchor_notice')]
    other = [row for row in rows if row.get('negative') and not row.get('anchor_notice')]
    result = {}
    for group, subset, metrics in (
        ('hard_negatives', hard, {'blocked': blocked, 'no_strong': no_strong}),
        ('original_negatives', other, {'blocked': blocked, 'no_strong': no_strong}),
        ('all_negatives', hard + other, {'blocked': blocked, 'no_strong': no_strong}),
        ('positives', positive, {'hit_at_5': target_hit}),
    ):
        window_rows = [dict(row, **row[window]) for row in subset]
        result[group] = {
            'n': len(subset),
            'policies': {
                policy: {
                    name: {
                        'count': (count := sum(metric(row, row[policy]) for row in window_rows)),
                        'wilson95': rate_ci(count, len(subset)),
                        'paired': paired(window_rows, 'baseline', metric, treatment=policy),
                        **(
                            {'topic_paired': cluster_test(window_rows, policy, metric)}
                            if group == 'hard_negatives'
                            else {}
                        ),
                    }
                    for name, metric in metrics.items()
                }
                for policy in COSINE_POLICIES
            },
        }
    return result


def evaluate_cosine(rows, vectors, searcher, chunk_k, notice_k):
    records = {}
    for chunk in searcher.chunks_by_id.values():
        records.setdefault(int(chunk['notice_num']), chunk['subject'])
    for row, vector in zip(rows, vectors, strict=True):
        hits = searcher._rank(vector, chunk_k) if normalized(row['query']) else []
        row['top1'] = (
            dict(notice=hits[0].notice_num, subject=hits[0].subject, score=hits[0].score)
            if hits
            else None
        )
        row['target_in_candidates'] = any(hit.notice_num in row.get('targets', ()) for hit in hits)
        if 'anchor_notice' in row:
            row['anchor_subject'] = records[row['anchor_notice']]
            row['anchor_in_candidates'] = any(
                hit.notice_num == row['anchor_notice'] for hit in hits
            )
        for name, widening in (('fixed_window', False), ('backend_prefix_replay', True)):
            row[name] = {
                policy: cosine_window(hits, *floors, notice_k, widening=widening)
                for policy, floors in COSINE_POLICIES.items()
            }
            assert row[name]['minimum_only']['strong'] == row[name]['baseline']['strong']
    return {
        'policies': COSINE_POLICIES,
        'summary': {
            window: cosine_summary(rows, window)
            for window in ('fixed_window', 'backend_prefix_replay')
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--split', choices=('calibration', 'holdout', 'cosine-intent'), default='holdout'
    )
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--chunk-k', type=int, default=100)
    parser.add_argument('--notice-k', type=int, default=5)
    args = parser.parse_args()
    if min(args.chunk_k, args.notice_k) < 1:
        parser.error('window sizes must be positive')
    cosine = args.split == 'cosine-intent'
    if cosine and args.notice_k != 5:
        parser.error('cosine-intent reports Hit@5 and requires --notice-k 5')
    rows = cosine_cases() if cosine else cases(args.split)
    embedder = KoreanEmbedder(device='cpu')
    searcher = SemanticSearcher.load(embedder)
    if not cosine and searcher.relevance_profile is None:
        parser.error('calibrated Korean BGE FlatIP artifacts are required')
    texts = [normalized(row['query']) for row in rows]
    vectors = embedder.embed_texts(texts, batch_size=8)
    if cosine:
        report = {
            'split': args.split,
            'model': embedder.model_name,
            'device': embedder.device,
            'indexed_chunks': len(searcher.chunk_ids),
            'index_updated_at': searcher.index_updated_at,
            'chunk_k': args.chunk_k,
            'notice_k': args.notice_k,
            **evaluate_cosine(rows, vectors, searcher, args.chunk_k, args.notice_k),
            'rows': rows,
        }
        args.out.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8'
        )
        for window, groups in report['summary'].items():
            print(window)
            for policy in COSINE_POLICIES:
                hard = groups['hard_negatives']['policies'][policy]
                other = groups['original_negatives']['policies'][policy]
                positive = groups['positives']['policies'][policy]
                print(
                    f'{policy}: hard blocked={hard["blocked"]["count"]}/'
                    f'{groups["hard_negatives"]["n"]}, '
                    f'hard no Strong={hard["no_strong"]["count"]}/'
                    f'{groups["hard_negatives"]["n"]}, '
                    f'original blocked={other["blocked"]["count"]}/'
                    f'{groups["original_negatives"]["n"]}, '
                    f'Strong Hit@5={positive["hit_at_5"]["count"]}/{groups["positives"]["n"]}'
                )
        return
    c = DEFAULT_CUTOFFS
    for row, text, vector in zip(rows, texts, vectors, strict=True):
        query = row['query']
        hits = searcher._rank(vector, args.chunk_k) if text else []
        clear = [hit for hit in hits if hit.score >= config.CLEAR_SIMILARITY]
        weak = [hit for hit in hits if config.MIN_SIMILARITY <= hit.score < config.CLEAR_SIMILARITY]
        row['target_in_candidates'] = any(hit.notice_num in row.get('targets', ()) for hit in hits)
        row['top1_cosine'] = hits[0].score if hits else None
        row['baseline'] = notice_lists(clear, weak, args.notice_k)
        if hits and hits[0].score < 0.6:
            # The legacy gate ORs evidence per notice but retains the best chunk's rank.
            def legacy(tier):
                kept = filter_by_lexical_evidence(query, tier, searcher.chunks_by_id)
                ids = {hit.notice_num for hit in kept}
                return [hit for hit in tier if hit.notice_num in ids]

            clear, weak = legacy(clear), legacy(weak)
        row['legacy_lexical'] = notice_lists(clear, weak, args.notice_k)
        started = perf_counter()
        result = searcher.tier_candidates(query, vector, hits)
        row['policy_ms'] = round((perf_counter() - started) * 1000, 3)
        row['calibrated'] = notice_lists(result.results, result.weak_results, args.notice_k)

        evidence = searcher.relevance_profile.for_query(query, vector)
        for variant in ('cosine_only', 'cosine_background'):
            clear, weak = [], []
            for hit in hits:
                s = hit.score
                if s < config.MIN_SIMILARITY:
                    continue
                z = evidence.measure(s, {}).z
                if variant == 'cosine_only':
                    if s >= max(config.CLEAR_SIMILARITY, c.strong_cosine):
                        clear.append(hit)
                    elif s >= c.weak_cosine:
                        weak.append(hit)
                else:
                    if z is None:
                        continue
                    if s >= max(config.CLEAR_SIMILARITY, c.strong_cosine) and z >= c.strong_z:
                        clear.append(hit)
                    elif s >= c.weak_cosine and z >= c.weak_z:
                        weak.append(hit)
            row[variant] = notice_lists(clear, weak, args.notice_k)

    report = {
        'split': args.split,
        'model': embedder.model_name,
        'device': embedder.device,
        'indexed_chunks': len(searcher.chunk_ids),
        'index_updated_at': searcher.index_updated_at,
        'chunk_k': args.chunk_k,
        'notice_k': args.notice_k,
        'thresholds': asdict(c),
        'summary': summarize(rows),
        'rows': rows,
    }
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
