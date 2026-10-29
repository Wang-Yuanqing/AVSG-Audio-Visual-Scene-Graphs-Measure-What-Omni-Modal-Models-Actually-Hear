#!/usr/bin/env python3
"""Paper statistics from released predictions; never runs a model.

Requires Python 3.10+ and NumPy. Preserves the archived metric, including
alignment ordering/ties, OOV predictions, multiplicities and dangling-edge
handling. Source-video bootstrap weights are shared across all comparisons
within each cohort. See manifest.json for input hashes and exact protocol.
"""
import argparse
import csv
import gzip
import hashlib
import itertools
import json
import os
from pathlib import Path, PurePosixPath
import sys
import types
import zipfile
from collections import Counter

for env_name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(env_name, "1")
import numpy as np

METRICS = ["node", "v2v", "v2a", "a2a", "edge", "all",
           "in_dialogue_with", "responds_to", "other_a2a", "audio_node"]
RELATIONS = {"in_dialogue_with", "responds_to"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def module_from_file(path, module_name):
    name = path.name
    source = path.read_bytes()
    module = types.ModuleType(module_name)
    exec(compile(source, name, "exec"), module.__dict__)
    return module, digest(source)


def read_records(path):
    name = str(path)
    data = path.read_bytes()
    records = [json.loads(line) for line in data.splitlines()
               if line.strip()]
    out = {record["filename"]: record for record in records}
    assert len(out) == len(records), (name, "duplicate filename")
    return out


def source_video(filename):
    video, start, end = filename.removesuffix(".mp4").rsplit("_", 2)
    assert float(end) >= float(start), filename
    return video


def detailed_match(metric, pred_edges, mapping, gold_edges):
    gold = Counter(metric._ekey(rel, src, dst)
                   for _, src, rel, dst in gold_edges)
    total, counts = 0, Counter()
    for family, src, rel, dst in pred_edges:
        gs, gt = mapping.get(src), mapping.get(dst)
        if gs is None or gt is None:
            continue
        key = metric._ekey(rel, gs, gt)
        if gold[key] > 0:
            gold[key] -= 1
            total += 1
            counts[family] += 1
            if family == "a2a":
                counts[(family, rel)] += 1
    return total, counts


def clip_counts(metric, gold_record, prediction):
    gold = metric.record_to_graph(gold_record)
    pred = metric.record_to_graph(prediction)
    fallback_before = metric._fallback_clips
    original = metric.score_pair(gold, pred)
    original_fallback = metric._fallback_clips - fallback_before
    _, gold_by, gold_edges = metric._graph_parts(gold)
    _, pred_by, pred_edges = metric._graph_parts(pred)
    original_counter = metric._count_matched
    # Instrument only the successful-match accounting. The original alignment
    # enumeration, cap and strict-greater tie-breaking remain unchanged.
    metric._count_matched = lambda pe, m, ge: detailed_match(metric, pe, m, ge)
    try:
        edge_tp, detailed = metric._best_alignment(
            pred_by, gold_by, pred_edges, gold_edges)
    finally:
        metric._count_matched = original_counter
    assert metric._fallback_clips - fallback_before == 2 * original_fallback
    assert edge_tp == original["edge"]["tp"]
    for family in ("v2v", "v2a", "a2a"):
        assert detailed[family] == original[family]["tp"]
    counts = []
    for component in METRICS[:6]:
        score = original[component]
        counts.append([score["tp"], score["n_pred"], score["n_gold"]])
    for relation in ("in_dialogue_with", "responds_to", "other_a2a"):
        accepts = (lambda r: r not in RELATIONS) if relation == "other_a2a" else (
            lambda r, expected=relation: r == expected)
        tp = sum(value for key, value in detailed.items()
                 if isinstance(key, tuple) and key[0] == "a2a" and accepts(key[1]))
        npred = sum(fam == "a2a" and accepts(rel)
                    for fam, _, rel, _ in pred_edges)
        ngold = sum(fam == "a2a" and accepts(rel)
                    for fam, _, rel, _ in gold_edges)
        counts.append([tp, npred, ngold])
    assert np.array(counts[6:9]).sum(axis=0).tolist() == counts[3]
    # Exactly the audio-class multiset recall used in the submitted scorer.
    gc = Counter(n["class"] for n in gold["nodes"] if str(n["id"]).startswith("a"))
    pc = Counter(n["class"] for n in pred["nodes"] if str(n["id"]).startswith("a"))
    counts.append([sum((gc & pc).values()), sum(pc.values()), sum(gc.values())])
    return np.asarray(counts, dtype=np.int64), original_fallback


def scores(counts):
    tp, predicted, gold = np.moveaxis(np.asarray(counts, dtype=float), -1, 0)
    precision = np.divide(tp, predicted, out=np.full_like(tp, np.nan), where=predicted > 0)
    recall = np.divide(tp, gold, out=np.full_like(tp, np.nan), where=gold > 0)
    f1 = np.divide(2 * tp, predicted + gold, out=np.zeros_like(tp), where=predicted + gold > 0)
    return np.stack((precision, recall, f1), axis=-1)


def interval(samples):
    finite = samples[np.isfinite(samples)]
    if not len(finite):
        return None, None, 0
    lo, hi = np.quantile(finite, [0.025, 0.975], method="linear")
    return float(lo), float(hi), len(finite)


def scalar(value):
    return float(value) if np.isfinite(value) else None


def write_csv(path, records):
    if not records:
        return
    with path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse-counts", type=Path,
                        help="Optional verified count cache; every input hash must match")
    args = parser.parse_args()
    root = args.root.resolve()
    args.output = args.output or root / "results/analysis/bootstrap"
    args.output.mkdir(parents=True, exist_ok=True)
    metric, metric_hash = module_from_file(root / "evaluation/avsg_metric.py", "analysis_metric")
    vocab, _ = module_from_file(root / "evaluation/vocab.py", "analysis_vocab")
    gold_path = root / "data/gold/1k_result.jsonl"
    gold = read_records(gold_path)
    split = json.loads((root / "data/splits/split_v1.json").read_text(encoding="utf-8"))
    keys = sorted(gold)
    key_index = {key: i for i, key in enumerate(keys)}
    test_keys = sorted(split["test_clips"])
    assert len(keys) == 1024 and len(test_keys) == len(set(test_keys)) == 197
    assert set(test_keys) <= set(gold)
    assert {source_video(fn) for fn in test_keys} == set(split["test_videos"])
    all_classes = vocab.VISUAL_CLASSES | vocab.AUDIO_CLASSES
    cached_counts, cached_hashes, cached_fallbacks = {}, {}, set()
    parent_manifest_hash = None
    if args.reuse_counts:
        parent_bytes = (args.reuse_counts / "manifest.json").read_bytes()
        parent = json.loads(parent_bytes)
        assert parent["metric_sha256"] == metric_hash
        assert parent["gold_sha256"] == digest(gold_path.read_bytes())
        with gzip.open(args.reuse_counts / "per_clip_counts.json.gz", "rt", encoding="utf-8") as stream:
            cache = json.load(stream)
        assert cache["filenames"] == keys and cache["metrics"] == METRICS
        cached_counts = cache["systems"]
        cached_hashes = {entry["system"]: entry["sha256"] for entry in parent["inputs"]}
        cached_fallbacks = {(entry["system"], entry["filename"])
                            for entry in parent["alignment_fallback_prediction_records"]}
        parent_manifest_hash = digest(parent_bytes)
    entries, systems, audits, fallback_records = [], {}, [], []
    verified_live_fallbacks = 0
    prediction_root = root / "results/predictions"
    paths = sorted(path for folder in ("main", "noaudio", "finetuned", "finetuned_noaudio")
                   for path in (prediction_root / folder).glob("*.jsonl"))
    for path in paths:
        name = path.relative_to(root).as_posix()
        model, setting = path.stem.rsplit("_", 1)
        directory = path.parent.name
        if directory == "main":
            condition = setting
        elif directory == "noaudio":
            condition = "noaudio"
        elif directory == "finetuned":
            condition, model = "ft_raw", model.removesuffix("-ft")
        elif directory == "finetuned_noaudio":
            condition, model = "ft_noaudio", model.removesuffix("-ft")
        else:
            raise AssertionError(name)
        system_key = model + "/" + condition
        assert system_key not in systems
        prediction_bytes = path.read_bytes()
        prediction_hash = digest(prediction_bytes)
        records = read_records(path)
        expected = set(test_keys) if condition.startswith("ft_") else set(keys)
        assert set(records) == expected, (name, "coverage", len(records))
        array = np.zeros((len(keys), len(METRICS), 3), dtype=np.int64)
        reusable = np.asarray(cached_counts[system_key], dtype=np.int64) if (
            cached_hashes.get(system_key) == prediction_hash) else None
        if reusable is not None:
            assert reusable.shape == array.shape
        audit = Counter()
        for filename, record in records.items():
            graph = metric.record_to_graph(record)
            if record.get("valid") is False:
                audit["invalid"] += 1
                assert not graph["nodes"] and not graph["edges"], (name, filename, "nonempty invalid")
            if "valid" not in record:
                audit["missing_valid_flag"] += 1
            ids = [node["id"] for node in graph["nodes"]]
            audit["duplicate_node_ids"] += len(ids) - len(set(ids))
            audit["bad_id_prefix"] += sum(not str(nid).startswith(("v", "a")) for nid in ids)
            audit["node_count"] += len(ids)
            audit["oov_node_count"] += sum(node["class"] not in all_classes for node in graph["nodes"])
            audit["edge_count_stored"] += len(graph["edges"])
            audit["oov_edge_count"] += sum(edge.get("edge", edge.get("relation", "")) not in vocab.ALL_EDGES
                                           for edge in graph["edges"])
            audit["dangling_edges"] += sum(edge.get("source") not in ids or edge.get("target") not in ids
                                            for edge in graph["edges"])
            if reusable is None:
                counts_here, fallback = clip_counts(metric, gold[filename], record)
                verified_live_fallbacks += fallback
            else:
                counts_here = reusable[key_index[filename]]
                fallback = int((system_key, filename) in cached_fallbacks)
            array[key_index[filename]] = counts_here
            audit["alignment_fallback_clips"] += fallback
            if fallback:
                fallback_records.append({"system": system_key, "filename": filename,
                                         "nodes": len(ids), "edges": len(graph["edges"])})
        metadata = {"system": system_key, "model": model, "condition": condition,
                    "file": name,
                    "sha256": prediction_hash,
                    "n_clips": len(records)}
        entries.append(metadata)
        systems[system_key] = (array, metadata)
        audits.append({**metadata, **{field: audit[field] for field in
                       ["invalid", "missing_valid_flag", "duplicate_node_ids", "bad_id_prefix",
                        "node_count", "oov_node_count", "edge_count_stored", "oov_edge_count", "dangling_edges",
                        "alignment_fallback_clips"]}})
        print("REUSED" if reusable is not None else "SCORED", system_key, len(records), flush=True)
    assert len(systems) == 38, len(systems)
    assert metric._fallback_clips == 2 * verified_live_fallbacks
    estimates, contrasts, bootstraps = [], [], {}
    cohort_metadata = {}
    for cohort, filenames in (("full", keys), ("test", test_keys)):
        videos = sorted({source_video(fn) for fn in filenames})
        assert len(videos) == (455 if cohort == "full" else 91)
        vindex = {video: i for i, video in enumerate(videos)}
        indices = [key_index[fn] for fn in filenames]
        group_ids = [vindex[source_video(fn)] for fn in filenames]
        rng = np.random.default_rng(args.seed)
        weights = rng.multinomial(len(videos), np.full(len(videos), 1 / len(videos)),
                                  size=args.bootstrap).astype(np.float64)
        cohort_metadata[cohort] = {"clips": len(filenames), "source_videos": len(videos),
                                   "sorted_source_video_ids": videos,
                                   "weights_sha256": digest(weights.tobytes())}
        for system_key, (counts, metadata) in systems.items():
            if cohort == "full" and metadata["condition"].startswith("ft_"):
                continue
            selected = counts[indices]
            grouped = np.zeros((len(videos), len(METRICS), 3), dtype=np.float64)
            np.add.at(grouped, group_ids, selected)
            pooled = selected.sum(axis=0)
            point = scores(pooled)
            boot_counts = (weights @ grouped.reshape(len(videos), -1)).reshape(
                args.bootstrap, len(METRICS), 3)
            boot_scores = scores(boot_counts)
            bootstraps[(cohort, system_key)] = (point, boot_scores)
            for mi, component in enumerate(METRICS):
                row = {"cohort": cohort, "model": metadata["model"], "condition": metadata["condition"],
                       "metric": component, "clips": len(filenames), "source_videos": len(videos),
                       "tp": int(pooled[mi, 0]), "predicted": int(pooled[mi, 1]), "gold": int(pooled[mi, 2])}
                for si, stat in enumerate(("precision", "recall", "f1")):
                    low, high, nfinite = interval(boot_scores[:, mi, si])
                    row.update({stat: scalar(point[mi, si]), stat + "_low": low,
                                stat + "_high": high, stat + "_finite_draws": nfinite})
                estimates.append(row)
        print("BOOTSTRAPPED", cohort, len(videos), args.bootstrap, flush=True)

    def contrast(cohort, label, terms, components=("all", "a2a", "audio_node", "in_dialogue_with", "responds_to")):
        for component in components:
            mi = METRICS.index(component)
            si = 1 if component == "audio_node" else 2
            point, draws = 0.0, np.zeros(args.bootstrap)
            for coefficient, key in terms:
                estimates_here, bootstrap_here = bootstraps[(cohort, key)]
                point += coefficient * estimates_here[mi, si]
                draws += coefficient * bootstrap_here[:, mi, si]
            low, high, nfinite = interval(draws)
            contrasts.append({"cohort": cohort, "comparison": label, "metric": component,
                              "statistic": "recall" if si == 1 else "f1", "estimate": float(point),
                              "ci_low": low, "ci_high": high, "finite_draws": nfinite,
                              "terms": json.dumps(terms)})

    for cohort in ("full", "test"):
        for key, (_, metadata) in systems.items():
            model, condition = metadata["model"], metadata["condition"]
            if condition == "noaudio":
                contrast(cohort, model + ": audio removal gap (AV - noaudio)",
                         [(1, model + "/raw"), (-1, key)])
        for frontier in ("gemini-2.5-pro", "gemini-3.1-pro", "gemini-3.5-flash"):
            contrast(cohort, frontier + " - qwen3-omni-30b-think (raw)",
                     [(1, frontier + "/raw"), (-1, "qwen3-omni-30b-think/raw")],
                     components=("all", "a2a", "in_dialogue_with", "responds_to"))
    for model in ("qwen2.5-omni-7b", "qwen3-omni-30b-instruct"):
        b, d, t, td = [model + "/" + condition for condition in ("raw", "noaudio", "ft_raw", "ft_noaudio")]
        contrast("test", model + ": tuned audio removal gap", [(1, t), (-1, td)])
        contrast("test", model + ": LoRA gain with audio", [(1, t), (-1, b)])
        contrast("test", model + ": LoRA gain without audio", [(1, td), (-1, d)])
        contrast("test", model + ": change in audio removal gap (DID)", [(1, t), (-1, td), (-1, b), (1, d)])

    write_csv(args.output / "estimates.csv", estimates)
    write_csv(args.output / "paired_contrasts.csv", contrasts)
    write_csv(args.output / "input_audit.csv", audits)
    payload = {"metrics": METRICS, "count_order": ["tp", "predicted", "gold"], "filenames": keys,
               "systems": {key: array.tolist() for key, (array, _) in systems.items()}}
    with gzip.open(args.output / "per_clip_counts.json.gz", "wt", encoding="utf-8") as stream:
        json.dump(payload, stream, separators=(",", ":"))
    manifest = {
                "metric_sha256": metric_hash, "script_sha256": digest(Path(__file__).read_bytes()),
                "gold_sha256": digest(gold_path.read_bytes()),
                "python": sys.version, "numpy": np.__version__, "bootstrap_draws": args.bootstrap,
                "seed": args.seed, "rng": "numpy.default_rng / PCG64",
                "resampling": "Uniform source-video cluster bootstrap via multinomial multiplicities; paired shared weights within each cohort",
                "interval": "Pointwise 95% percentile, numpy.quantile method=linear; no multiplicity correction",
                "undefined_rates": "Precision/recall undefined at zero denominators; conditional intervals use finite draws and expose their counts. Micro-F1 empty/empty=0, matching archived dataset scorer.",
                "alignment": "Archived full-graph alignment/tie-breaking; per-relation counts instrument original matcher; all clip edge and family TPs asserted identical",
                "invalid": "All explicit valid=False records asserted stored empty; no OOV filtering or repairs applied",
                "alignment_fallback_prediction_records": fallback_records,
                "fallback_counter_including_instrumented_recheck": metric._fallback_clips,
                "reused_verified_count_parent_manifest_sha256": parent_manifest_hash,
                "cohorts": cohort_metadata, "inputs": entries,
                "notes": "Intervals quantify source-video sampling variability conditional on these references, outputs and selected pool; not annotation, selection, decoding or API uncertainty."}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    # Simple, non-rounded structured report; Markdown rendering is done separately.
    (args.output / "results.json").write_text(json.dumps({"estimates": estimates, "contrasts": contrasts}, indent=2) + "\n", encoding="utf-8")
    print("COMPLETE", args.output, "estimates", len(estimates), "contrasts", len(contrasts), flush=True)


if __name__ == "__main__":
    main()
