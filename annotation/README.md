# Annotation tools

These are portable adaptations of the graph-editing and blinded relation-audit interfaces. They operate on media you obtain separately; video files are not included. Install optional dependencies from the repository root:

```sh
python -m pip install -r requirements-annotation.txt
```

## Graph editor

```sh
python annotation/graph_editor.py --input annotation/example.jsonl --video-dir clips --output-dir outputs/annotations
```

Open `http://127.0.0.1:7861`, select annotator ID 1, and click Start. The example uses a complete released gold graph. Supply your own parsed prefill JSONL to begin from model output; records use `filename`, `nodes`, `edges` and optional `caption`. Gold-form records using `ground_truth_nodes`/`ground_truth_edges` also work. Nodes require `id`, `class` and `attribute`; edges require `source`, `edge`, and `target`.

The editor shows video, graph and JSON with controls for adding/removing nodes and relations. It saves one canonical gold-schema JSON file per clip under the selected output directory. Revisiting a clip loads that annotator's saved version. Read [the rubric](../data/rubric.md) while annotating; the editor does not replace rubric review or validate all vocabulary constraints.

For multiple annotators, pass `--annotators N --clips-per-annotator K`. Sorted input clips are allocated in fixed, non-overlapping slices. Numeric IDs are assignment labels, not authentication. Use a trusted local environment or an access-controlled deployment. `--host`, `--port` and opt-in `--share` control serving; the default is local only. The output directory is blocked from Gradio file serving.

## Blinded relation-audit interface

Prepare a **fresh demonstration study**, separate from the paper's saved evidence:

```sh
python -m annotation.create_audit_example --output outputs/audit_demo_qa
python -m annotation.audit.app --study-dir outputs/audit_demo_qa --video-dir clips --preview
```

Open `http://127.0.0.1:7862`. Volunteers enter a name, consent to the displayed study instructions and claim a task. A browser-held token permits resuming; no assigned password is needed. The interface requires watching the clip and then offers **Correct / Incorrect / Cannot tell**, with no default judgment, plus a media-failure button. It displays the edge and its one-sentence rule; it hides graph provenance and other participants' answers.

The demo has one clip and two rater slots. It is a UI example, not the original sampling procedure. A new research study needs a frozen `study.json` containing clips/tasks, anonymous rater slots, assignments and per-rater orders. Do not run new judgments against the paper's released audit data. That final study involved **eight volunteers, 135 unique edges and 330 judgments, with two or three judgments per edge**; the initial six-slot deployment design is not its final sample size. The released evidence and its aggregation are documented separately under `data/audit/` and `analysis/`.

Runtime files include `participants.json` (private names, enrollment token hashes and a browser-storage secret), per-rater judgments and revision histories. Keep these local, preserve them for safe resumption, and never add them to a public repository. The public repo contains no original participant enrollment file.
