"""Verify frozen audit sampling and reproduce full-corpus unmatched counts."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import compute_results as C

def unmatched_indices(metric, gold_record, prediction):
    """Original full-graph alignment; return original prediction edge indices."""
    gold, pred = map(metric.record_to_graph, (gold_record, prediction))
    score = metric.score_pair(gold, pred)
    _, gb, ge = metric._graph_parts(gold)
    _, pb, pe = metric._graph_parts(pred)
    node_ids = {n["id"] for n in pred["nodes"]}
    original_indices = [i for i, e in enumerate(pred["edges"])
                        if e.get("source") in node_ids and e.get("target") in node_ids]
    assert len(original_indices) == len(pe)
    original = metric._count_matched

    def count(edges, mapping, gold_edges):
        available = Counter(metric._ekey(rel, src, dst) for _, src, rel, dst in gold_edges)
        total, counts = 0, Counter()
        for index, (fam, src, rel, dst) in enumerate(edges):
            gs, gt = mapping.get(src), mapping.get(dst)
            if gs is None or gt is None:
                continue
            key = metric._ekey(rel, gs, gt)
            if available[key]:
                available[key] -= 1
                total += 1
                counts[fam] += 1
                counts[("prediction_index", index)] += 1
        return total, counts

    metric._count_matched = count
    try:
        total, detail = metric._best_alignment(pb, gb, pe, ge)
    finally:
        metric._count_matched = original
    assert total == score["edge"]["tp"]
    for fam in ("v2v", "v2a", "a2a"):
        assert detail[fam] == score[fam]["tp"]
    matched = {key[1] for key in detail if isinstance(key, tuple)}
    return [original_indices[i] for i in range(len(pe)) if i not in matched]



def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = p.parse_args()
    root = args.root
    metric, metric_hash = C.module_from_file(root/'evaluation/avsg_metric.py', 'audit_sampling_metric')
    gold = C.read_records(root/'data/gold/1k_result.jsonl')
    protocol = json.loads((root/'data/audit/protocol.json').read_text(encoding='utf-8'))
    clips = json.loads((root/'data/audit/clips.json').read_text(encoding='utf-8'))
    edges = [json.loads(l) for l in (root/'data/audit/edges.jsonl').read_text(encoding='utf-8').splitlines()]
    tasks = defaultdict(list)
    for row in edges:
        tasks[row['clip_id']].append(row)
    sampling = protocol['sampling']
    models = sampling['models']
    preds = {model:C.read_records(root/f'results/predictions/main/{model}_raw.jsonl') for model in models}
    pools = {model:{fn:unmatched_indices(metric, gold[fn], pred) for fn,pred in rows.items()} for model,rows in preds.items()}
    totals = {m:sum(len(pool) for pool in rows.values()) for m,rows in pools.items()}
    assert totals == {'gemini-3.1-pro':2233, 'qwen3-omni-30b-think':1990}
    strata = {label:sorted(fn for fn,r in gold.items() if
        any(str(e['source']).startswith('a') and str(e['target']).startswith('a') for e in r['ground_truth_edges']) == present)
        for label,present in [('a2a_present',True),('a2a_absent',False)]}
    assert {k:len(v) for k,v in strata.items()} == sampling['stratum_sizes']
    rng = random.Random(sampling['seed'])
    selected = []
    for label in ('a2a_present','a2a_absent'):
        selected.extend((fn,label) for fn in rng.sample(strata[label],sampling['per_stratum']))
    assignments = [models[0]]*12+[models[1]]*12
    random.Random(sampling['seed']+1).shuffle(assignments)
    for i,((fn,label),model) in enumerate(zip(selected,assignments),1):
        cid = f'c{i:03d}'
        frozen = next(c for c in clips if c['clip_id']==cid)
        assert (frozen['filename'],frozen['stratum'],frozen['assigned_model']) == (fn,label,model)
        pending = [('gold',j) for j in range(len(gold[fn]['ground_truth_edges']))]
        pool = pools[model][fn]
        assert frozen['eligible_pool_size'] == len(pool)
        if pool:
            pending.append(('prediction',rng.choice(pool)))
        rng.shuffle(pending)
        frozen_tasks = tasks[cid]
        assert pending == [(r['origin'],r['gold_edge_index'] if r['origin']=='gold' else r['prediction_edge_index']) for r in frozen_tasks]
        for row in frozen_tasks:
            pred = gold[fn] if row['origin']=='gold' else preds[model][fn]
            graph = metric.record_to_graph(pred)
            index = row['gold_edge_index'] if row['origin']=='gold' else row['prediction_edge_index']
            e = graph['edges'][index]
            nodes = {n['id']:n for n in graph['nodes']}
            assert row['source'] == nodes[e['source']] and row['target'] == nodes[e['target']]
            assert row['relation'] == e['edge']
    assert metric._fallback_clips == 0
    report = {'corpus_unmatched_edges':totals,'sampled_unmatched_edges':dict(Counter(r['model'] for r in edges if r['origin']=='prediction')),
              'sampling_seed':sampling['seed'],'clip_sampling_model_assignment_edge_sampling_and_shuffle_match':True,
              'all_135_displayed_edges_match_released_graphs':True,'metric_sha256':metric_hash,
              'scope':'Sampled unmatched-edge acceptance describes the small audit sample; it does not correct corpus F1.'}
    output = root/'results/analysis/audit_sampling.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
