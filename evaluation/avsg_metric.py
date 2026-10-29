"""AVSG graph accuracy metric — class-typed graph alignment.

Compares two AVSG graphs by node CLASS and edge (class, relation, class),
ignoring attributes and arbitrary node ids.

  * Nodes: scored alignment-free as the multiset of (modality, class).
  * Edges: scored under the node alignment m (a class-respecting partial
    injection pred->gold) that MAXIMIZES the number of matched edges. This is a
    Smatch-style match (Cai & Knight, 2013) specialized by node type; the node
    side is SPICE-style tuple matching (Anderson et al., 2016). Class-typing
    makes the optimal alignment EXACT and cheap: only duplicate classes (e.g.
    two `man` nodes) have any alignment freedom, so we enumerate the small
    product of per-duplicate-class bijections and take the max.

Symmetric relations (next_to, playing_with, in_dialogue_with) are matched
order-agnostically. The same function serves benchmark scoring (pred vs gold)
and IAA (annotator A vs B).

Reports node-F1, edge-F1, per-family edge-F1 (v2v/v2a/a2a), and a combined
node+edge F1, as both macro (mean per-clip) and micro (pooled).
"""

import itertools
import json
import sys
from collections import Counter

SYMMETRIC = {"next_to", "playing_with", "in_dialogue_with"}
COMPONENTS = ["node", "edge", "v2v", "v2a", "a2a", "all"]

_ALIGN_CAP = 200_000      # max joint alignments to enumerate before greedy fallback
_fallback_clips = 0       # clips that hit the cap (expected 0 on real data)


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #
def record_to_graph(rec):
    return {"nodes": rec.get("ground_truth_nodes", rec.get("nodes", [])),
            "edges": rec.get("ground_truth_edges", rec.get("edges", []))}


def _ntype(node_id):
    return "v" if str(node_id).startswith("v") else "a"


def _graph_parts(graph):
    """-> (id2tc: id->(type,class),  by: (type,class)->[ids],  edges:[(family,src,rel,tgt)])."""
    id2tc, by = {}, {}
    for n in graph["nodes"]:
        nid = n["id"]
        tc = (_ntype(nid), n.get("class", ""))
        id2tc[nid] = tc
        by.setdefault(tc, []).append(nid)
    for tc in by:
        by[tc].sort()                                    # deterministic
    edges = []
    for e in graph["edges"]:
        s, t = e.get("source"), e.get("target")
        r = e.get("edge", e.get("relation", ""))
        if s in id2tc and t in id2tc:                    # drop dangling edges
            edges.append((_ntype(s) + "2" + _ntype(t), s, r, t))
    return id2tc, by, edges


# --------------------------------------------------------------------------- #
# Edge matching under a node alignment
# --------------------------------------------------------------------------- #
def _ekey(r, a, b):
    return (r, frozenset((a, b))) if r in SYMMETRIC else (r, a, b)


def _count_matched(pred_edges, m, gold_edges):
    """Matched pred edges under alignment m (each gold edge consumed at most once)."""
    gold = Counter(_ekey(r, gs, gt) for _f, gs, r, gt in gold_edges)
    tp, per = 0, Counter()
    for fam, s, r, t in pred_edges:
        gs, gt = m.get(s), m.get(t)
        if gs is None or gt is None:
            continue
        k = _ekey(r, gs, gt)
        if gold[k] > 0:
            gold[k] -= 1
            tp += 1
            per[fam] += 1
    return tp, per


def _class_injections(pred_ids, gold_ids):
    """All partial injections pairing min(|pred|,|gold|) same-class nodes."""
    p, g = len(pred_ids), len(gold_ids)
    if p == 0 or g == 0:
        return [()]
    if p <= g:
        return [tuple(zip(pred_ids, sel)) for sel in itertools.permutations(gold_ids, p)]
    return [tuple(zip(sel, gold_ids)) for sel in itertools.permutations(pred_ids, g)]


def _best_alignment(pred_by, gold_by, pred_edges, gold_edges):
    """Max matched edges over class-respecting node alignments (exact; greedy if huge)."""
    global _fallback_clips
    classes = sorted(pred_by)
    options = [_class_injections(pred_by[tc], gold_by.get(tc, [])) for tc in classes]
    total = 1
    for o in options:
        total *= len(o)

    if total > _ALIGN_CAP:                               # pathological #duplicates
        _fallback_clips += 1
        m = {p: g for o in options for (p, g) in o[0]}
        return _count_matched(pred_edges, m, gold_edges)

    best_tp, best_per = 0, Counter()
    for combo in itertools.product(*options):
        m = {p: g for pairs in combo for (p, g) in pairs}
        tp, per = _count_matched(pred_edges, m, gold_edges)
        if tp > best_tp:
            best_tp, best_per = tp, per
    return best_tp, best_per


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _prf(tp, n_pred, n_gold):
    p = tp / n_pred if n_pred else (1.0 if n_gold == 0 else 0.0)
    r = tp / n_gold if n_gold else (1.0 if n_pred == 0 else 0.0)
    f1 = 2 * p * r / (p + r) if (p + r) else (1.0 if n_gold == 0 and n_pred == 0 else 0.0)
    return {"p": p, "r": r, "f1": f1, "tp": tp, "n_gold": n_gold, "n_pred": n_pred}


def score_pair(gold, pred):
    """Score one graph pair. Returns {component: {p,r,f1,tp,n_gold,n_pred}}."""
    g_id, g_by, g_ed = _graph_parts(gold)
    p_id, p_by, p_ed = _graph_parts(pred)

    # nodes — alignment-free multiset of (modality, class)
    tp_node = sum((Counter(g_id.values()) & Counter(p_id.values())).values())
    # edges — best class-typed alignment
    tp_edge, per_fam = _best_alignment(p_by, g_by, p_ed, g_ed)

    g_fam = Counter(f for f, *_ in g_ed)
    p_fam = Counter(f for f, *_ in p_ed)
    out = {
        "node": _prf(tp_node, len(p_id), len(g_id)),
        "edge": _prf(tp_edge, len(p_ed), len(g_ed)),
    }
    for fam in ("v2v", "v2a", "a2a"):
        out[fam] = _prf(per_fam.get(fam, 0), p_fam.get(fam, 0), g_fam.get(fam, 0))
    out["all"] = _prf(tp_node + tp_edge, len(p_id) + len(p_ed), len(g_id) + len(g_ed))
    return out


# --------------------------------------------------------------------------- #
# Dataset / IAA aggregation
# --------------------------------------------------------------------------- #
def _load(path):
    by_name = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                by_name[rec["filename"]] = record_to_graph(rec)
    return by_name


def score_dataset(gold_path, pred_path):
    global _fallback_clips
    _fallback_clips = 0
    gold, pred = _load(gold_path), _load(pred_path)
    keys = sorted(set(gold) & set(pred))
    if not keys:
        raise SystemExit("No overlapping filenames between the two files.")

    macro = {c: [] for c in COMPONENTS}        # only clips where the component is present
    micro = {c: Counter() for c in COMPONENTS}
    for k in keys:
        s = score_pair(gold[k], pred[k])
        for c in COMPONENTS:
            sc = s[c]
            if sc["n_gold"] or sc["n_pred"]:    # empty-gold & empty-pred score 1.0 -> exclude
                macro[c].append(sc["f1"])
            micro[c]["tp"] += sc["tp"]
            micro[c]["n_gold"] += sc["n_gold"]
            micro[c]["n_pred"] += sc["n_pred"]

    def micro_f1(c):
        tp, ng, npd = micro[c]["tp"], micro[c]["n_gold"], micro[c]["n_pred"]
        p = tp / npd if npd else 0.0
        r = tp / ng if ng else 0.0
        return (2 * p * r / (p + r)) if (p + r) else 0.0

    print(f"Scored {len(keys)} clips ({gold_path} as gold, {pred_path} as pred)\n")
    print(f"{'component':8s}  {'macro-F1':>9s}  {'micro-F1':>9s}  {'n_macro':>7s}")
    for c in COMPONENTS:
        vals = macro[c]
        mac = f"{sum(vals) / len(vals):9.3f}" if vals else f"{'-':>9s}"
        print(f"{c:8s}  {mac}  {micro_f1(c):9.3f}  {len(vals):>7d}")
    if _fallback_clips:
        print(f"\n[warn] {_fallback_clips} clip(s) exceeded the alignment cap (greedy fallback).")


def iaa_pairwise(paths):
    sets = [_load(p) for p in paths]
    common = set(sets[0])
    for s in sets[1:]:
        common &= set(s)
    common = sorted(common)
    if not common:
        raise SystemExit("No clips annotated by all annotators.")
    n = len(paths)
    agg = {c: [] for c in COMPONENTS}
    for i in range(n):
        for j in range(i + 1, n):
            for k in common:
                s = score_pair(sets[i][k], sets[j][k])
                for c in COMPONENTS:
                    sc = s[c]
                    if sc["n_gold"] or sc["n_pred"]:    # exclude empty/empty (would score 1.0)
                        agg[c].append(sc["f1"])
    print(f"IAA over {len(common)} shared clips, {n} annotators, "
          f"{n * (n - 1) // 2} pair(s), mean pairwise macro-F1:\n")
    for c in COMPONENTS:
        vals = agg[c]
        val = f"{sum(vals) / len(vals):7.3f}" if vals else f"{'-':>7s}"
        print(f"{c:8s}  {val}  (n={len(vals)})")


# --------------------------------------------------------------------------- #
# Self-test / CLI
# --------------------------------------------------------------------------- #
def _g(nodes, edges):
    return {"nodes": [{"id": i, "class": c} for i, c in nodes],
            "edges": [{"source": s, "edge": r, "target": t} for s, r, t in edges]}


def _selftest():
    # one man does both vs the prediction splitting it into two men
    gold = _g([("v1", "man"), ("v2", "cup"), ("a1", "male_speech")],
              [("v1", "holding", "v2"), ("v1", "speaking", "a1")])
    pred = _g([("v1", "man"), ("v2", "man"), ("v3", "cup"), ("a1", "male_speech")],
              [("v1", "holding", "v3"), ("v2", "speaking", "a1")])
    s = score_pair(gold, pred)
    print("duplicate-identity test (gold: 1 man holds+speaks; pred: 2 men split):")
    for c in ("node", "edge", "all"):
        print(f"  {c:5s} P={s[c]['p']:.3f} R={s[c]['r']:.3f} F1={s[c]['f1']:.3f}")
    assert abs(s["edge"]["f1"] - 0.5) < 1e-9, "alignment should penalize the split -> edge F1 0.5"
    assert abs(s["node"]["f1"] - 6 / 7) < 1e-9
    # identity sanity
    assert score_pair(gold, gold)["all"]["f1"] == 1.0
    print("OK: edge-F1=0.5 (coreference penalized), self=1.0")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        _selftest()
    elif args[0] == "--iaa":
        iaa_pairwise(args[1:])
    elif len(args) == 2:
        score_dataset(args[0], args[1])
    else:
        print("usage:\n  python avsg_metric.py gold.jsonl pred.jsonl\n"
              "  python avsg_metric.py --iaa A.jsonl B.jsonl [C.jsonl ...]")
