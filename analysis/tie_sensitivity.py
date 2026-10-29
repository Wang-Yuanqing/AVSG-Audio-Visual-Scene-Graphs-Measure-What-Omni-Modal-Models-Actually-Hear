"""Repeat the paper's single deterministic gold-ID renaming sensitivity check."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import compute_results as C

COMPONENTS = ("node", "edge", "all", "v2v", "v2a", "a2a")
FAMILIES = ("v2v", "v2a", "a2a")

def reverse_duplicate_ids(graph):
    """Reverse lexicographically sorted IDs within each modality/class group.

    Retain node/edge list order, classes, attributes and all incidence relations;
    rename endpoints simultaneously. Predictions are not renamed.
    """
    groups = defaultdict(list)
    for node in graph["nodes"]:
        modality = "v" if str(node["id"]).startswith("v") else "a"
        groups[(modality, node.get("class", ""))].append(node["id"])
    renaming = {}
    for ids in groups.values():
        ordered = sorted(ids)
        renaming.update(zip(ordered, reversed(ordered)))
    return {
        "nodes": [{**node, "id": renaming[node["id"]]} for node in graph["nodes"]],
        "edges": [{**edge, "source": renaming.get(edge["source"], edge["source"]),
                   "target": renaming.get(edge["target"], edge["target"])}
                  for edge in graph["edges"]],
    }


def score_row(metric, gold, renamed_gold, pred):
    assert set(pred) == set(gold), "prediction cohort differs from the 1,024 gold clips"
    totals = [{component: Counter() for component in COMPONENTS} for _ in range(2)]
    changed = []
    fallback = [[], []]
    for filename in sorted(gold):
        scores = []
        for variant, refs in enumerate((gold, renamed_gold)):
            before = metric._fallback_clips
            score = metric.score_pair(refs[filename], pred[filename])
            scores.append(score)
            if metric._fallback_clips != before:
                fallback[variant].append(filename)
            for component in COMPONENTS:
                for count in ("tp", "n_pred", "n_gold"):
                    totals[variant][component][count] += score[component][count]
        for component in ("node", "edge", "all"):
            assert scores[0][component] == scores[1][component], (filename, component)
        if any(scores[0][family]["tp"] != scores[1][family]["tp"] for family in FAMILIES):
            changed.append({"filename": filename,
                            "original_tp": {c: scores[0][c]["tp"] for c in FAMILIES},
                            "renamed_tp": {c: scores[1][c]["tp"] for c in FAMILIES}})
    result = {}
    for component in COMPONENTS:
        for variant in (0, 1):
            count = totals[variant][component]
            denominator = count["n_pred"] + count["n_gold"]
            count["f1"] = 2 * count["tp"] / denominator if denominator else 0.0
        result[component] = {"original": dict(totals[0][component]),
                             "renamed": dict(totals[1][component]),
                             "delta_f1": totals[1][component]["f1"] - totals[0][component]["f1"]}
    return {"scores": result, "changed_family_clips": changed,
            "fallback_original": fallback[0], "fallback_renamed": fallback[1]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    root = args.root
    metric, metric_hash = C.module_from_file(root/'evaluation/avsg_metric.py', 'tie_metric')
    gold_path = root/'data/gold/1k_result.jsonl'
    gold = {fn:metric.record_to_graph(r) for fn,r in C.read_records(gold_path).items()}
    renamed = {fn:reverse_duplicate_ids(g) for fn,g in gold.items()}
    systems = {}
    for path in sorted((root/'results/predictions/main').glob('*.jsonl')):
        pred = {fn:metric.record_to_graph(r) for fn,r in C.read_records(path).items()}
        systems[path.stem] = {'prediction_sha256': C.digest(path.read_bytes()),
                              **score_row(metric, gold, renamed, pred)}
        print('CHECKED', path.stem, flush=True)
    assert len(systems) == 24
    changes = [len(r['changed_family_clips']) for r in systems.values() if r['changed_family_clips']]
    maximum = max((abs(r['scores'][f]['delta_f1']), model, f)
                  for model,r in systems.items() for f in FAMILIES)
    report = {'metric_sha256':metric_hash, 'gold_sha256':C.digest(gold_path.read_bytes()),
              'scope':'One deterministic duplicate-class gold-ID reversal preserving graph content. This is not an exhaustive sensitivity bound and does not replace archived family scores.',
              'summary':{'configurations':24, 'configurations_with_family_count_changes':len(changes),
                         'changed_clips_per_affected_configuration_range':[min(changes),max(changes)],
                         'max_abs_family_micro_f1_change':{'absolute_delta_f1':maximum[0], 'model_setting':maximum[1], 'family':maximum[2]},
                         'all_clip_node_edge_all_counts_invariant':True},
              'systems':systems}
    assert len(changes) == 18
    assert abs(maximum[0]-0.0047562425683709275) < 1e-10
    output = args.output or root/'results/analysis/tie_sensitivity.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report['summary'],indent=2))


if __name__ == '__main__':
    main()
