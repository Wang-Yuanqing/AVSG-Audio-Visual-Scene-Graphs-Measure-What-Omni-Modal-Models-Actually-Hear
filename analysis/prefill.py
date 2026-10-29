#!/usr/bin/env python3
"""Audit author-confirmed original prefills and raw-model A2A recall.

Uses stored outputs only. Instruments successful-match accounting while keeping
the original metric's alignment enumeration, strict-greater ties and fallback.
No vocabulary normalization, attribute scoring or subset-specific realignment.
"""
import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path, PurePosixPath
import types
import zipfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def records(data):
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    result = {row["filename"]: row for row in rows}
    assert len(rows) == len(result), "Duplicate filenames"
    return result


def detailed_match(metric, pred_edges, mapping, gold_edges):
    """Return the original counts, plus consumed gold indices in file order."""
    available = defaultdict(deque)
    for index, (_family, src, rel, dst) in enumerate(gold_edges):
        available[metric._ekey(rel, src, dst)].append(index)
    total, counts = 0, Counter()
    for family, src, rel, dst in pred_edges:
        gs, gt = mapping.get(src), mapping.get(dst)
        if gs is None or gt is None:
            continue
        key = metric._ekey(rel, gs, gt)
        if available[key]:
            index = available[key].popleft()
            total += 1
            counts[family] += 1
            counts[("gold_index", index)] += 1
    return total, counts


def aligned_gold_indices(metric, gold_record, prediction):
    gold, pred = map(metric.record_to_graph, (gold_record, prediction))
    before = metric._fallback_clips
    score = metric.score_pair(gold, pred)
    fallback = metric._fallback_clips - before
    _, gold_by, gold_edges = metric._graph_parts(gold)
    # Gold has no dangling edges: metric indices are the original file indices.
    assert len(gold_edges) == len(gold["edges"])
    assert all((src, rel, dst) == (edge["source"], edge["edge"], edge["target"])
               for (_family, src, rel, dst), edge in zip(gold_edges, gold["edges"]))
    _, pred_by, pred_edges = metric._graph_parts(pred)
    original = metric._count_matched
    metric._count_matched = lambda pe, mapping, ge: detailed_match(
        metric, pe, mapping, ge)
    try:
        tp, details = metric._best_alignment(
            pred_by, gold_by, pred_edges, gold_edges)
    finally:
        metric._count_matched = original
    assert metric._fallback_clips - before == 2 * fallback
    assert tp == score["edge"]["tp"]
    for family in ("v2v", "v2a", "a2a"):
        assert details[family] == score[family]["tp"]
    indices = {key[1] for key in details
               if isinstance(key, tuple) and key[0] == "gold_index"}
    assert len(indices) == tp
    return score, gold_edges, indices, fallback


def fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    json_path = args.output_json or root / "results/analysis/prefill.json"
    metric_path = root / "evaluation/avsg_metric.py"
    metric_bytes = metric_path.read_bytes()
    metric = types.ModuleType("actual_prefill_metric")
    exec(compile(metric_bytes, str(metric_path), "exec"), metric.__dict__)
    gold_path = root / "data/gold/1k_result.jsonl"
    gold_bytes = gold_path.read_bytes()
    gold = records(gold_bytes)
    prefill_path = root / "data/prefill_original/prefill_1023.jsonl"
    provenance_path = root / "data/prefill_original/provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    prefills = records(prefill_path.read_bytes())
    missing = {row["filename"]: row["reason"] for row in provenance["excluded"]}
    sources = {key: prefill_path.relative_to(root).as_posix() for key in prefills}
    batch_counts = {str(row["batch"]): row["source_record_count"] for row in provenance["source_batches"]}
    hashes = {p.relative_to(root).as_posix(): digest(p.read_bytes())
              for p in (metric_path, gold_path, prefill_path, provenance_path, Path(__file__).resolve())}
    assert len(gold) == 1024 and len(prefills) == 1023
    assert set(prefills) | set(missing) == set(gold)
    assert set(prefills).isdisjoint(missing)
    keys = sorted(prefills)
    totals, retained_a2a, absent_a2a = Counter(), {}, {}
    per_clip, fallback_records = {}, []
    for filename in keys:
        score, edges, matched, fallback = aligned_gold_indices(
            metric, gold[filename], prefills[filename])
        totals["unchanged_graphs"] += score["all"]["f1"] == 1
        totals["gold_edges"] += len(edges)
        totals["retained_edges"] += len(matched)
        if fallback:
            fallback_records.append({"source": "actual_prefill", "filename": filename})
        retained_a2a[filename] = {i for i in matched if edges[i][0] == "a2a"}
        absent_a2a[filename] = {i for i, edge in enumerate(edges)
                                if edge[0] == "a2a" and i not in matched}
        for label, indices in (("retained", retained_a2a[filename]),
                               ("absent", absent_a2a[filename])):
            totals[label + "_a2a"] += len(indices)
            for i in indices:
                totals[label + "_" + edges[i][2]] += 1
        per_clip[filename] = {
            "prefill_source": sources[filename],
            "all_f1": score["all"]["f1"],
            "gold_edge_count": len(edges),
            "matched_gold_edge_indices": sorted(matched),
            "retained_a2a_gold_indices": sorted(retained_a2a[filename]),
            "absent_a2a_gold_indices": sorted(absent_a2a[filename]),
            "raw_model_a2a_matches": {},
        }
    totals["changed_graphs"] = len(keys) - totals["unchanged_graphs"]
    totals["absent_edges"] = totals["gold_edges"] - totals["retained_edges"]
    assert [totals[k] for k in ("unchanged_graphs", "changed_graphs", "gold_edges",
                                "retained_edges", "absent_edges", "retained_a2a",
                                "absent_a2a")] == [65, 958, 4429, 2774, 1655, 267, 186]
    exclusions = []
    for filename, reason in missing.items():
        graph = metric.record_to_graph(gold[filename])
        _, _, edges = metric._graph_parts(graph)
        row = {"filename": filename, "reason": reason,
               "gold_nodes": len(graph["nodes"]), "gold_edges": len(edges),
               "gold_a2a": sum(edge[0] == "a2a" for edge in edges)}
        assert (row["gold_nodes"], row["gold_edges"], row["gold_a2a"]) == (5, 4, 0)
        exclusions.append(row)
    models, model_inputs = [], []
    names = sorted((root / "results/predictions/main").glob("*_raw.jsonl"))
    assert len(names) == 12
    for path in names:
        model = path.name.removesuffix("_raw.jsonl")
        data = path.read_bytes()
        source = path.relative_to(root).as_posix()
        prediction = records(data)
        assert set(prediction) == set(gold), model
        model_inputs.append({"model": model, "source": source,
                             "sha256": digest(data),
                             "lf_normalized_sha256": digest(data.replace(b"\r\n", b"\n"))})
        counts = Counter()
        for filename in keys:
            score, edges, matched, fallback = aligned_gold_indices(
                metric, gold[filename], prediction[filename])
            if fallback:
                fallback_records.append({"source": model, "filename": filename})
            model_indices = {i for i in matched if edges[i][0] == "a2a"}
            assert len(model_indices) == score["a2a"]["tp"]
            kept = model_indices & retained_a2a[filename]
            added = model_indices & absent_a2a[filename]
            assert len(kept) + len(added) == len(model_indices)
            counts["retained_tp"] += len(kept)
            counts["absent_tp"] += len(added)
            for label, indices in (("retained", kept), ("absent", added)):
                for i in indices:
                    counts[label + "_" + edges[i][2] + "_tp"] += 1
            if retained_a2a[filename] or absent_a2a[filename]:
                per_clip[filename]["raw_model_a2a_matches"][model] = {
                    "retained_gold_indices": sorted(kept),
                    "absent_gold_indices": sorted(added),
                }
        row = {"model": model, "clips": len(keys), **dict(counts),
               "retained_gold": totals["retained_a2a"],
               "absent_gold": totals["absent_a2a"],
               "retained_recall": fraction(counts["retained_tp"], totals["retained_a2a"]),
               "absent_recall": fraction(counts["absent_tp"], totals["absent_a2a"])}
        for relation in ("in_dialogue_with", "responds_to"):
            row["absent_" + relation + "_gold"] = totals["absent_" + relation]
        models.append(row)
        print(f"{model}: absent A2A {row['absent_tp']}/{row['absent_gold']} "
              f"= {row['absent_recall']:.6f}", flush=True)
    report = {
        "provenance": "Author-confirmed original Gemini 3.1 Pro annotation-input batches; parsed graph-only exports, not mixed recovered versions.",
        "scope": "Raw class-level prefill-to-gold differences; attributes ignored; includes vocabulary changes. Not independent annotation accuracy or proof of removal of generator bias.",
        "matching": "Original full-graph metric alignment, original strict-greater tie-breaking and fallback. Matching consumes duplicate gold edges in file order. Model recall uses the same fixed absent-edge indices; no class-triplet multiset or subset realignment.",
        "gold_edge_index_integrity": "Asserted no dangling gold edges and exact edge-order identity with original gold records for every comparison.",
        "n_gold": len(gold), "n_comparable": len(keys),
        "batch_record_counts": batch_counts, "excluded": exclusions,
        "counts": dict(totals),
        "changed_graph_fraction": totals["changed_graphs"] / len(keys),
        "absent_edge_fraction": totals["absent_edges"] / totals["gold_edges"],
        "raw_models": models, "sha256": hashes,
        "model_inputs": model_inputs, "fallback_records": fallback_records,
        "original_scorer_equality_checks": len(keys) * (1 + len(models)),
        "per_clip": per_clip,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"counts": report["counts"], "output": str(json_path),
                      "fallback_records": fallback_records}, indent=2))


if __name__ == "__main__":
    main()
