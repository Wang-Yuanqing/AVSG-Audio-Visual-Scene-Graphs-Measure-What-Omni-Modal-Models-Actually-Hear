"""Score the binary validation.

Reads every data/validation/judgments_*.json plus catch_key.json and reports:
  * acceptance rate on REAL clips, per set (paper table) and per validator, Wilson 95% CI
  * inter-validator agreement on the agree/disagree verdict: raw % agreement + Gwet's
    AC1 (skew-robust; Cohen's kappa is NOT used because it collapses under skew)
  * per-item consensus (both raters agree the clip is good)
  * error-type breakdown of disagreements
  * catch-trial rejection rate (specificity -- should be high)

It first prints a completion table (who finished their assigned set), then scores.

Usage:
  python score_validation.py                       # all validators
  python score_validation.py --complete-only       # only those who finished their set
  python score_validation.py alice bob carol dave  # only the named validators (pick the 4)
"""

import glob
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8")   # avoid cp1252 crashes on Windows
except Exception:
    pass

VAL_DIR = os.environ.get("AVSG_VAL_DIR", os.path.join(os.path.dirname(__file__), "..", "data", "validation"))


def set_index_of(vid):
    m = re.match(r"s(\d+)_", str(vid))
    return int(m.group(1)) if m else None


def set_sizes():
    sizes = {}
    for p in glob.glob(os.path.join(VAL_DIR, "validation_set_*.jsonl")):
        m = re.search(r"validation_set_(\d+)\.jsonl$", p.replace("\\", "/"))
        if m:
            sizes[int(m.group(1))] = sum(1 for l in open(p, encoding="utf-8") if l.strip())
    return sizes


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (max(0.0, c - h), min(1.0, c + h))


def gwet_ac1(pairs):
    """Gwet's AC1 for two raters -- a chance-corrected agreement coefficient that,
    unlike Cohen's kappa, does NOT collapse when labels are heavily skewed (here
    almost everything is 'agree'). Returns (ac1, observed_agreement)."""
    n = len(pairs)
    if n == 0:
        return None, None
    po = sum(1 for a, b in pairs if a == b) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    cats = set(ca) | set(cb)
    q = len(cats)
    if q < 2:                      # both raters used a single label -> perfect
        return 1.0, po
    pe = sum(((ca[c] / n + cb[c] / n) / 2) * (1 - (ca[c] / n + cb[c] / n) / 2)
             for c in cats) / (q - 1)
    ac1 = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return ac1, po


def main():
    key_path = os.path.join(VAL_DIR, "catch_key.json")
    key = json.load(open(key_path, encoding="utf-8")) if os.path.exists(key_path) else {}
    is_catch = {vid: meta.get("is_catch", False) for vid, meta in key.items()}

    files = sorted(glob.glob(os.path.join(VAL_DIR, "judgments_*.json")))
    if not files:
        raise SystemExit(f"No judgments_*.json in {VAL_DIR}.")

    judgments = {}      # validator -> {vid: verdict}
    raw = {}            # validator -> {vid: full entry}
    for fp in files:
        name = os.path.basename(fp)[len("judgments_"):-len(".json")]
        data = json.load(open(fp, encoding="utf-8"))
        raw[name] = data
        judgments[name] = {vid: e["verdict"] for vid, e in data.items()}

    # ---- completion report + optional validator selection ----
    sizes = set_sizes()

    def vset(verds):
        for vid in verds:
            si = set_index_of(vid)
            if si:
                return si
        return None

    info = {name: (vset(verds), len(verds), sizes.get(vset(verds)),
                   sizes.get(vset(verds)) is not None and len(verds) >= sizes.get(vset(verds)))
            for name, verds in judgments.items()}

    print("=" * 64)
    print("VALIDATOR COMPLETION")
    print("=" * 64)
    for name in sorted(judgments):
        si, done, total, comp = info[name]
        print(f"  {name:16s} set {si}  recorded {done}/{total if total is not None else '?'}"
              f"   {'COMPLETE' if comp else 'partial'}")

    sel = [a for a in sys.argv[1:] if not a.startswith("-")]
    if sel:
        chosen = [n for n in judgments if n in sel]
        print(f"\n  scoring only: {', '.join(chosen) or '(no match)'}")
    elif "--complete-only" in sys.argv:
        chosen = [n for n in judgments if info[n][3]]
        print(f"\n  scoring only COMPLETE validators: {', '.join(chosen) or '(none)'}")
    else:
        chosen = list(judgments)
    judgments = {n: judgments[n] for n in chosen}
    if not judgments:
        raise SystemExit("No validators selected.")
    print()

    errors = Counter()      # only over selected validators
    for name in judgments:
        for vid, e in raw[name].items():
            if e["verdict"] == "disagree" and not is_catch.get(vid, False):
                errors[e.get("error_type", "other")] += 1

    def real(vid):
        return not is_catch.get(vid, False)

    # ---- acceptance rate on real clips ----
    print("=" * 64)
    print("ACCEPTANCE RATE on real clips (agree = annotation is acceptable)")
    print("=" * 64)

    # ---- per-set table (paper-ready) ----
    by_set = {}     # set_idx -> [agree, decided, skip]
    for name, verds in judgments.items():
        si = info[name][0]
        agg = by_set.setdefault(si, [0, 0, 0])
        for vid, v in verds.items():
            if not real(vid):
                continue
            if v == "agree":
                agg[0] += 1; agg[1] += 1
            elif v == "disagree":
                agg[1] += 1
            elif v == "skip":
                agg[2] += 1

    print(f"  {'set':<6}{'clips':>7}{'judged':>8}{'accept%':>10}   95% CI")
    tot_a = tot_d = tot_clips = 0
    for si in sorted(by_set):
        a, d, _sk = by_set[si]
        clips = sizes.get(si, 0)
        tot_a += a; tot_d += d; tot_clips += clips
        lo, hi = wilson(a, d)
        acc = f"{a / d:.1%}" if d else "-"
        print(f"  {si:<6}{clips:>7}{d:>8}{acc:>10}   [{lo:.1%}, {hi:.1%}]")
    if tot_d:
        lo, hi = wilson(tot_a, tot_d)
        print(f"  {'ALL':<6}{tot_clips:>7}{tot_d:>8}{tot_a / tot_d:>9.1%}   [{lo:.1%}, {hi:.1%}]")

    # ---- per-validator detail ----
    print("\n  per validator:")
    pooled_skip = 0
    for name, verds in judgments.items():
        decided = [v for vid, v in verds.items() if real(vid) and v in ("agree", "disagree")]
        nskip = sum(1 for vid, v in verds.items() if real(vid) and v == "skip")
        k = sum(1 for v in decided if v == "agree")
        n = len(decided)
        pooled_skip += nskip
        if n:
            lo, hi = wilson(k, n)
            print(f"    {name:16s} {k:3d}/{n:<3d} = {k / n:5.1%}   95% CI [{lo:.1%}, {hi:.1%}]"
                  f"   (+{nskip} skipped)")
        else:
            print(f"    {name:16s} (no decided clips; {nskip} skipped)")
    print(f"  (acceptance = agree / (agree+disagree); {pooled_skip} skipped clips excluded)")

    # ---- agreement on the verdict ----
    print("\n" + "=" * 64)
    print("INTER-VALIDATOR AGREEMENT on the agree/disagree verdict")
    print("=" * 64)
    names = list(judgments)
    vid_raters = defaultdict(list)
    for name, verds in judgments.items():
        for vid, v in verds.items():
            if real(vid) and v in ("agree", "disagree"):   # skips excluded from agreement
                vid_raters[vid].append((name, v))
    shared = {vid: rs for vid, rs in vid_raters.items() if len(rs) >= 2}

    if not shared:
        print("  No real clips judged by >=2 validators (no overlap).")
    else:
        # pairwise: raw % agreement + Gwet's AC1 (skew-robust; kappa not used)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                pairs = [(judgments[a][vid], judgments[b][vid])
                         for vid in judgments[a]
                         if vid in judgments[b] and real(vid)
                         and judgments[a][vid] in ("agree", "disagree")
                         and judgments[b][vid] in ("agree", "disagree")]
                if pairs:
                    ac1, po = gwet_ac1(pairs)
                    print(f"  {a} vs {b}: n={len(pairs)}  agreement={po:.1%}  Gwet AC1={ac1:.3f}")

        unanimous_good = sum(1 for vid, rs in shared.items()
                             if all(v == "agree" for _, v in rs))
        print(f"  Consensus 'good' (all raters agree): "
              f"{unanimous_good}/{len(shared)} = {unanimous_good / len(shared):.1%}")

    print("\n  AC1 guide (Landis-Koch scale): .41-.60 moderate, .61-.80 substantial, "
          ".81-1 almost perfect  (AC1 stays meaningful under skew; kappa does not)")

    # ---- error breakdown ----
    if sum(errors.values()):
        print("\n" + "=" * 64)
        print("DISAGREEMENT ERROR TYPES (real clips)")
        print("=" * 64)
        tot = sum(errors.values())
        for et, c in errors.most_common():
            print(f"  {et:28s} {c:3d}  ({c / tot:.0%})")

    # ---- catch trials ----
    if any(is_catch.values()):
        print("\n" + "=" * 64)
        print("CATCH TRIALS — rejection rate (should be HIGH = validators discriminate)")
        print("=" * 64)
        ck = cn = 0
        for name, verds in judgments.items():
            cv = [v for vid, v in verds.items() if is_catch.get(vid, False)]
            rej = sum(1 for v in cv if v == "disagree")
            ck += rej; cn += len(cv)
            if cv:
                print(f"  {name:16s} rejected {rej}/{len(cv)} = {rej / len(cv):.1%}")
        if cn:
            print(f"  {'POOLED':16s} rejected {ck}/{cn} = {ck / cn:.1%}")


if __name__ == "__main__":
    main()
