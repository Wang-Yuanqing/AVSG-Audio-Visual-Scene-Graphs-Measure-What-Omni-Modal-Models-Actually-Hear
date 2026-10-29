"""Recompute the all-retained edge audit from anonymous answer-level evidence."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def summarize(rows):
    counts = Counter()
    family = {}
    model = {}
    weight_all = weight_correct = 0.0
    raters = Counter()
    seen = set()
    for row in rows:
        assert row['task_id'] not in seen
        seen.add(row['task_id'])
        answers = row['judgments']
        assert 2 <= len(answers) <= 3
        assert len({j['rater'] for j in answers}) == len(answers)
        verdicts = [j['verdict'] for j in answers]
        assert set(verdicts) <= {'correct', 'incorrect', 'uncertain'}
        accepted = all(v == 'correct' for v in verdicts)
        counts[f"edges_with_{len(answers)}_raters"] += 1
        counts['judgments'] += len(answers)
        counts['exact_unanimity'] += len(set(verdicts)) == 1
        for j in answers:
            raters[j['rater']] += 1
            counts['verdict_' + j['verdict']] += 1
        if row['origin'] == 'gold':
            counts['gold_edges'] += 1
            counts['gold_unanimous_correct'] += accepted
            cell = family.setdefault(row['family'], {'audited': 0, 'unanimous_correct': 0})
            weight = 1 / row['clip_inclusion_probability']
            weight_all += weight
            weight_correct += weight * accepted
        else:
            assert row['origin'] == 'prediction'
            counts['unmatched_edges'] += 1
            counts['unmatched_unanimous_correct'] += accepted
            cell = model.setdefault(row['model'], {'audited': 0, 'unanimous_correct': 0})
        cell['audited'] += 1
        cell['unanimous_correct'] += accepted
    return {'clips': len({r['filename'] for r in rows}),
            'source_videos': len({r['filename'].removesuffix('.mp4').rsplit('_', 2)[0] for r in rows}),
            'edges': len(rows), 'raters': len(raters), 'counts': dict(counts),
            'answers_per_anonymous_rater': dict(sorted(raters.items())),
            'gold_by_family': family, 'unmatched_by_model': model,
            'gold_unweighted_unanimous_correct': counts['gold_unanimous_correct'] / counts['gold_edges'],
            'gold_design_weighted_unanimous_correct': weight_correct / weight_all,
            'weighting': 'Ratio of inverse-clip-inclusion-probability weighted unanimous-correct gold edges to all audited gold edges.',
            'scope': 'Sampled reference correctness and unmatched-relation validity. Not full-graph human F1, omitted-edge recall, or a corrected corpus model score.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    source = args.root / 'data/audit/edges.jsonl'
    rows = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines() if line.strip()]
    report = summarize(rows)
    assert (report['clips'], report['edges'], report['raters'], report['counts']['judgments']) == (24, 135, 8, 330)
    assert (report['counts']['gold_unanimous_correct'], report['counts']['gold_edges']) == (106, 114)
    assert round(report['gold_design_weighted_unanimous_correct'] * 100, 1) == 91.0
    report['input_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    output = args.output or args.root / 'results/analysis/audit.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
