"""Robust parser for model output -> AVSG graph.

Models emit messy text: markdown fences, single-key wrappers, "relation" vs
"edge" keys, stray prose. parse() recovers {nodes, edges} or marks it invalid.
Out-of-vocab classes are KEPT (they should hurt precision, not be hidden) but
counted so we can report an out-of-vocab rate per model.
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from vocab import VISUAL_CLASSES, AUDIO_CLASSES  # noqa: E402


def _repair_json(s, _depth=0):
    """Best-effort structural repair (string-aware), used only as a last resort so it
    never touches already-parseable output. Handles the model degeneration modes seen:
    DROPS mismatched closing brackets (stray ']'), APPENDS any missing ones (dropped
    final '}'), and CLOSES an unterminated string that swallowed a trailing run of
    structural chars (e.g. gemini-3.1-pro emits `"a1}]}` for `"a1"}]}`)."""
    if _depth > 2:
        return None
    out, stack = [], []
    in_str = esc = False
    str_start = None                                  # index in s of the open quote
    for i, ch in enumerate(s):
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True; str_start = i; out.append(ch)
        elif ch in "{[":
            stack.append(ch); out.append(ch)
        elif ch == "}":
            if stack and stack[-1] == "{":
                stack.pop(); out.append(ch)           # else: drop mismatched
        elif ch == "]":
            if stack and stack[-1] == "[":
                stack.pop(); out.append(ch)           # else: drop mismatched
        else:
            out.append(ch)

    if in_str:                                        # unterminated trailing string
        if str_start is None:
            return None
        content = s[str_start + 1:]                   # chars after the open quote
        value = content.rstrip("}] \t\r\n")           # strip the absorbed structure
        run = content[len(value):]
        if not run.strip():                           # nothing structural to recover
            return None
        return _repair_json(s[:str_start + 1] + value + '"' + run, _depth + 1)

    base = "".join(out).rstrip()
    base += "".join("}" if c == "{" else "]" for c in reversed(stack))
    if not base:
        return None
    for attempt in (base, re.sub(r",\s*([}\]])", r"\1", base)):
        try:
            return json.loads(attempt)
        except Exception:
            continue
    return None


def _is_graph_dict(d):
    return isinstance(d, dict) and ("nodes" in d or "edges" in d)


def _merge_graphs(graphs):
    """Concat several {nodes, edges} objects into one graph. Node ids dedup on first
    occurrence (the objects describe disjoint modality subgraphs; a repeated id would
    otherwise double-count its class in the metric's multiset); edges dedup exact."""
    nodes, edges, seen_ids, seen_edges = [], [], set(), set()
    for g in graphs:
        for n in g.get("nodes") or []:
            nid = n.get("id") if isinstance(n, dict) else None
            if isinstance(nid, str):
                if nid in seen_ids:
                    continue
                seen_ids.add(nid)
            nodes.append(n)
        for e in g.get("edges") or []:
            key = json.dumps(e, sort_keys=True) if isinstance(e, dict) else repr(e)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            edges.append(e)
    return {"nodes": nodes, "edges": edges}


def _extract_json_blob(text):
    if not isinstance(text, str):
        return None
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1]
    text = text.strip()
    start = text.find("{")
    if start == -1:
        return None
    # Parse the FIRST complete JSON object starting at '{'; raw_decode stops at the
    # end of that object and ignores any trailing garbage. Some models (e.g.
    # gemini-3.5-flash) emit a valid graph then degenerate into repeated "}]} }]}…",
    # which a naive first-'{' to last-'}' slice would wrongly include.
    try:
        dec = json.JSONDecoder()
        first, idx = dec.raw_decode(text, start)
        # Some models (video-SALMONN 2+) emit TWO complete graph objects back to back —
        # `{visual nodes/edges}, {audio nodes/edges}` — and dropping the tail loses every
        # audio node. Keep decoding complete objects and merge those that are themselves
        # graph dicts; anything undecodable (the degeneration tails above) still stops
        # the scan, so single-object outputs parse exactly as before.
        if _is_graph_dict(first):
            graphs = [first]
            while True:
                nxt = text.find("{", idx)
                if nxt == -1:
                    break
                try:
                    obj, idx = dec.raw_decode(text, nxt)
                except Exception:
                    break
                obj = _unwrap(obj)
                if not _is_graph_dict(obj):
                    break
                graphs.append(obj)
            if len(graphs) > 1:
                return _merge_graphs(graphs)
        return first
    except Exception:
        pass
    # fallback: outermost { ... } with trailing-comma cleanup
    j = text.rfind("}")
    if j > start:
        blob = text[start:j + 1]
        for attempt in (blob, re.sub(r",\s*([}\]])", r"\1", blob)):
            try:
                return json.loads(attempt)
            except Exception:
                continue
    return _repair_json(text[start:])     # balance unclosed brackets (truncated final brace)


def _salvage_prefix(text):
    """Very last resort, tried only after every whole-output recovery (including
    the bare-key retry) failed: the output died on a syntax event partway through
    (prose degeneration, a corrupt token, a mid-key truncation). Walk '}' cut
    points backwards and repair the longest salvageable prefix. Accept ONLY a
    graph with non-empty nodes, so this can never manufacture an empty-but-
    "valid" record from junk — it recovers what the model actually said."""
    if not isinstance(text, str):
        return None
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    start = text.find("{")
    if start == -1:
        return None
    end = len(text)
    for _ in range(200):
        j = text.rfind("}", start + 1, end)
        if j == -1:
            break
        end = j
        obj = _repair_json(text[start:j + 1])
        if obj is not None:
            obj = _unwrap(obj)
            if isinstance(obj, dict) and obj.get("nodes"):
                return obj
    return None


def _unwrap(data):
    """{'clip.mp4': {nodes, edges}} -> {nodes, edges}."""
    if isinstance(data, dict) and "nodes" not in data and "edges" not in data and len(data) == 1:
        inner = list(data.values())[0]
        if isinstance(inner, dict) and ("nodes" in inner or "edges" in inner):
            return inner
    return data


def _coerce_node(n):
    if not isinstance(n, dict):
        return None
    nid = n.get("id") or n.get("node_id")
    cls = n.get("class") or n.get("label") or n.get("type")
    if not isinstance(nid, str) or not isinstance(cls, str):
        return None
    return {"id": nid, "class": cls.strip(), "attribute": n.get("attribute", "")}


def _coerce_edge(e):
    if not isinstance(e, dict):
        return None
    src = e.get("source") or e.get("src") or e.get("from")
    tgt = e.get("target") or e.get("tgt") or e.get("to")
    rel = e.get("edge") or e.get("relation") or e.get("rel") or e.get("name")
    if not (isinstance(src, str) and isinstance(tgt, str) and isinstance(rel, str)):
        return None
    return {"source": src, "edge": rel.strip(), "target": tgt}


_BARE_KEY = re.compile(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)')


def _quote_bare_keys(text):
    """Quote JSON5-style unquoted object keys (`target:` -> `"target":`), string-aware
    so key-like runs inside attribute values are left alone. Last-resort only — callers
    must try the verbatim text first, so anything parseable today parses identically."""
    out, in_str, esc = [], False, False
    i = 0
    while i < len(text):
        ch = text[i]
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            esc = False
            out.append(ch)
            i += 1
            continue
        m = _BARE_KEY.match(text, i)
        if m:
            out.append(f'{m.group(1)}"{m.group(2)}"{m.group(3)}')
            i = m.end()
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def parse(text):
    """Return {nodes, edges, valid, oov_nodes}."""
    data = _extract_json_blob(text)
    if data is None and isinstance(text, str) and text:
        data = _extract_json_blob(_quote_bare_keys(text))
        if data is None:
            data = _salvage_prefix(text)
        if data is None:
            data = _salvage_prefix(_quote_bare_keys(text))
    if data is None:
        return {"nodes": [], "edges": [], "valid": False, "oov_nodes": 0}
    data = _unwrap(data)
    if not isinstance(data, dict):
        return {"nodes": [], "edges": [], "valid": False, "oov_nodes": 0}

    nodes = [c for c in (_coerce_node(n) for n in data.get("nodes", [])) if c]
    edges = [c for c in (_coerce_edge(e) for e in data.get("edges", [])) if c]

    oov = sum(1 for n in nodes
              if (n["id"].startswith("v") and n["class"] not in VISUAL_CLASSES)
              or (n["id"].startswith("a") and n["class"] not in AUDIO_CLASSES))
    return {"nodes": nodes, "edges": edges, "valid": True, "oov_nodes": oov}
