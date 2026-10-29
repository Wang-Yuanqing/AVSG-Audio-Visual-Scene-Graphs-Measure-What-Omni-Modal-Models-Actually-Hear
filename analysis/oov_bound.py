"""Deterministic bound for relabeling OOV labels at fixed topology and counts."""
import argparse
from collections import Counter
import json
from pathlib import Path
import compute_results as C


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    root = args.root
    metric, metric_hash = C.module_from_file(root/'evaluation/avsg_metric.py', 'bound_metric')
    vocab, _ = C.module_from_file(root/'evaluation/vocab.py', 'bound_vocab')
    gold_path = root/'data/gold/1k_result.jsonl'
    gold = C.read_records(gold_path)
    classes = vocab.VISUAL_CLASSES | vocab.AUDIO_CLASSES
    models = {}
    for model in ('qwen3-omni-30b-think', 'gemini-3.1-pro', 'video-salmonn2plus-7b'):
        path = root/f'results/predictions/main/{model}_raw.jsonl'
        predictions = C.read_records(path)
        assert set(predictions) == set(gold)
        totals = Counter()
        per_clip = []
        for filename in sorted(gold):
            g = metric.record_to_graph(gold[filename])
            pred = metric.record_to_graph(predictions[filename])
            score = metric.score_pair(g, pred)
            nodes = {n['id']: n for n in pred['nodes']}
            oov_ids = {n['id'] for n in pred['nodes'] if n['class'] not in classes}
            _, _, edges = metric._graph_parts(pred)
            affected = sum(rel not in vocab.ALL_EDGES or src in oov_ids or dst in oov_ids
                           for _, src, rel, dst in edges)
            incident = sum(src in oov_ids or dst in oov_ids for _,src,rel,dst in edges)
            oov_edges = sum(rel not in vocab.ALL_EDGES for _,src,rel,dst in edges)
            node_gain = min(len(oov_ids), score['node']['n_gold'] - score['node']['tp'])
            edge_gain = min(affected, score['edge']['n_gold'] - score['edge']['tp'])
            counts = {'node_tp':score['node']['tp'], 'edge_tp':score['edge']['tp'],
                      'n_pred':score['all']['n_pred'], 'n_gold':score['all']['n_gold'],
                      'oov_nodes':len(oov_ids), 'oov_relations_scored':oov_edges,
                      'oov_node_incident_edges':incident, 'affected_edges':affected,
                      'node_gain_upper':node_gain, 'edge_gain_upper':edge_gain,
                      'nodes_stored':len(pred['nodes']), 'edges_stored':len(pred['edges']),
                      'duplicate_id_excess':len(pred['nodes'])-len(nodes),
                      'dangling_edges':len(pred['edges'])-len(edges),
                      'bad_prefix_nodes':sum(not str(n['id']).startswith(('v','a')) for n in pred['nodes'])}
            totals.update(counts)
            per_clip.append({'filename':filename, **counts})
        if model == 'qwen3-omni-30b-think':
            assert totals['duplicate_id_excess'] == totals['dangling_edges'] == totals['bad_prefix_nodes'] == 0
        den = totals['n_pred'] + totals['n_gold']
        observed = 2*(totals['node_tp']+totals['edge_tp'])/den
        gain = 2*(totals['node_gain_upper']+totals['edge_gain_upper'])/den
        models[model] = {'prediction_sha256':C.digest(path.read_bytes()), 'counts':dict(totals),
                         'ALL_micro_F1':observed, 'per_clip':per_clip}
        if model == 'qwen3-omni-30b-think':
            models[model].update(OOV_only_F1_gain_upper=gain, OOV_only_ALL_upper=observed+gain)
    assert metric._fallback_clips == 0
    q = models['qwen3-omni-30b-think']
    assert (q['counts']['node_gain_upper'],q['counts']['edge_gain_upper']) == (129,169)
    assert abs(q['OOV_only_ALL_upper'] - 0.6389338907760648) < 1e-12
    report = {'metric_sha256':metric_hash, 'gold_sha256':C.digest(gold_path.read_bytes()),
              'scope':'Only currently OOV node/relation labels may change; IDs, modalities, topology, multiplicities and in-vocabulary labels stay fixed. This is an optimistic bound, not an achieved correction.',
              'formula':'Per clip, node gain <= min(OOV node count, gold nodes - node TP); edge gain <= min(edges incident to OOV nodes or carrying OOV relations, gold edges - edge TP). Sum gains before converting to pooled F1.',
              'alignment_argument':'Restrict any corrected alignment to originally in-vocabulary nodes. Its unaffected edge matches cannot exceed the original exact optimum. All additional matches must come from affected edges, even when the optimal alignment changes.',
              'models':models}
    output = args.output or root/'results/analysis/oov_bound.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({model:{k:v for k,v in r.items() if k not in ('per_clip','counts')} for model,r in models.items()}, indent=2))


if __name__ == '__main__':
    main()
