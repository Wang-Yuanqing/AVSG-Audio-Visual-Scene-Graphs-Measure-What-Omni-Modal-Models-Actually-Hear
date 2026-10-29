"""Run the AVSG relation audit: python -m annotation.audit.app --study-dir PATH."""
import argparse
import hashlib
import html
import os
from pathlib import Path
import random
import sys
import time

import gradio as gr
from .storage import AnswerStore
from .enrollment import ParticipantStore
from .rules import rule_for

ROOT = Path(__file__).resolve().parents[1]
GUIDE = """
Use headphones and watch each complete clip at least once. Judge independently; do not discuss answers.

Read the relationship and the **one-sentence rule next to it**, then choose:
- **Correct:** the clip supports the two endpoints, relationship and direction, following the shown rule.
- **Incorrect:** the relationship is wrong, absent, points the wrong way, or breaks the shown rule.
- **Cannot tell:** the video/audio is too ambiguous to decide.

Clicking one of the three buttons **saves your answer and advances**. You can go back to revise it.
There is no expected acceptance rate. Some category names intentionally cover similar objects
(for example a banjo is labeled guitar, and an ATV is labeled motorcycle).
You only judge the displayed relationship, not a complete graph. The progress panel shows your assigned task count.
"""
CSS = """
.gradio-container { max-width: 1240px !important; margin: 0 auto !important; }
.audit-intro { color: #526173; }
.relation-card {padding:20px;border:1px solid #d9e3ef;border-radius:14px;background:#f7fafc;color:#172c43;}
.relation-card, .relation-card * {color:#111111 !important;}
.relation-flow {display:flex;gap:14px;align-items:center;justify-content:space-between;}
.endpoint {flex:1;min-width:0;padding:14px;background:white;border-radius:10px;border-top:4px solid #2f7199;}
.endpoint.audio {border-color:#ad623c;}
.endpoint small {text-transform:uppercase;letter-spacing:.09em;font-size:11px;color:#637487;}
.endpoint strong {display:block;font-size:22px;margin:5px 0;overflow-wrap:anywhere;}
.endpoint p {font-size:14px;line-height:1.45;margin:6px 0;overflow-wrap:anywhere;}
.relation {text-align:center;max-width:160px;overflow-wrap:anywhere;font-weight:700;}
.relation span {display:block;font-size:30px;color:#3b607d;}
.edge-rule {margin-top:16px;padding:12px 14px;border-radius:8px;background:#eaf0f5;font-size:16px;line-height:1.55;}
@media(max-width:700px){.relation-flow{flex-direction:column;align-items:stretch;}.relation{align-self:center}.endpoint strong{font-size:18px}}
"""


def task_card(task):
    def endpoint(node):
        modality = "Visual" if str(node.get("id", "")).startswith("v") else "Audio"
        attr = html.escape(str(node.get("attribute") or "No further description"))
        label = html.escape(str(node.get("class", "")))
        return f'<div class="endpoint {modality.lower()}"><small>{modality} endpoint</small><strong>{label}</strong><p>{attr}</p></div>'
    rel = task["relation"]
    arrow = "↔" if rel in {"next_to", "playing_with", "in_dialogue_with"} else "→"
    return (f'<div class="relation-card"><div class="relation-flow">{endpoint(task["source"])}'
            f'<div class="relation">{html.escape(rel)}<span>{arrow}</span></div>'
            f'{endpoint(task["target"])}</div><div class="edge-rule"><b>Rule for this relationship:</b> '
            f'{html.escape(task.get("rule") or rule_for(task))}</div></div>')


class AuditController:
    def __init__(self, study_dir, video_dirs):
        self.store = AnswerStore(study_dir)
        self.study = self.store.study
        self.clips = {c["clip_id"]: c for c in self.study["clips"]}
        self.video_dirs = [Path(p).resolve() for p in video_dirs]
        self.videos = {}
        for cid, clip in self.clips.items():
            filename = clip["filename"]
            if Path(filename).name != filename:
                raise ValueError("Clip filenames must be basenames.")
            found = next((p / filename for p in self.video_dirs if (p / filename).is_file()), None)
            if found is None:
                raise FileNotFoundError(f"Missing clip: {filename}. Add its directory with --video-dir.")
            self.videos[cid] = str(found)
        self.queues = {}
        for rater in self.study["annotators"]:
            queue = []
            for cid in self.study["orders"][rater]:
                tasks = list(self.clips[cid]["tasks"])
                seed = hashlib.sha256(f'{self.study["seed"]}:{rater}:{cid}'.encode()).hexdigest()
                random.Random(seed).shuffle(tasks)
                queue.extend((cid, t) for t in tasks)
            assert {t["task_id"] for _, t in queue} == self.store.assigned[rater]
            self.queues[rater] = queue

    def begin(self, rater, confirmed):
        if rater not in self.queues:
            raise ValueError("Use Start / resume to claim your task.")
        if not confirmed:
            raise ValueError("Read the instructions and check the box before starting.")
        answers = self.store.read(rater)["answers"]
        idx = next((i for i, (_, t) in enumerate(self.queues[rater]) if t["task_id"] not in answers), 0)
        return {"rater": rater, "index": idx, "watched": [], "loaded_at": time.monotonic()}

    def view(self, session, prior_clip=None, message=""):
        rater = session["rater"]
        cid, task = self.queues[rater][session["index"]]
        answers = self.store.read(rater)["answers"]
        answer = answers.get(task["task_id"], {})
        session = dict(session, revision=answer.get("revision", 0), loaded_at=time.monotonic())
        clip_ids = self.study["orders"][rater]
        local_tasks = [t for c, t in self.queues[rater] if c == cid]
        local_index = next(i for i, t in enumerate(local_tasks) if t["task_id"] == task["task_id"])
        completed = len(answers)
        total = len(self.store.assigned[rater])
        progress = (f'**{completed}/{total} relations saved**  \n'
                    f'Clip **{clip_ids.index(cid) + 1}/{len(clip_ids)}** · '
                    f'Relation **{local_index + 1}/{len(local_tasks)}** in this clip')
        if completed == total:
            progress += "  \n**All assigned relations are saved. Thank you! You can review answers below.**"
        labels = {"correct": "Correct", "incorrect": "Incorrect", "uncertain": "Cannot tell"}
        saved = f'Saved: **{labels[answer["verdict"]]}** · revision {answer["revision"]}' if answer else "Not yet saved"
        return (session, gr.update(visible=True), progress,
                self.videos[cid] if cid != prior_clip else gr.skip(), task_card(task),
                cid in session["watched"],
                message or saved, gr.update(interactive=session["index"] > 0))

    def save(self, session, verdict, note, watched, reason="none"):
        if not session:
            raise ValueError("Start the audit first.")
        rater = session["rater"]
        cid, task = self.queues[rater][session["index"]]
        self.store.save(rater, task["task_id"], verdict, note or "", reason,
                        expected_revision=session["revision"],
                        elapsed_seconds=round(max(0, time.monotonic() - session["loaded_at"]), 2),
                        media_watched=watched)
        new = dict(session, watched=list(session["watched"]))
        if watched and cid not in new["watched"]:
            new["watched"].append(cid)
        new["index"] = min(session["index"] + 1, len(self.queues[rater]) - 1)
        return self.view(new, prior_clip=cid, message="Answer saved.")


class VolunteerAccess:
    """Resolve browser tokens before accessing a rater's name and answers."""

    def __init__(self, controller):
        self.controller = controller
        self.participants = ParticipantStore(controller.store.root)

    def begin(self, confirmed, token, cache, name=""):
        if not confirmed:
            raise ValueError("Read the instructions and check the box before starting.")
        # A queued second click can arrive before the first token reaches the
        # browser. Keep the claim in Gradio's server-side session as well.
        rater, token = self.participants.claim(token or cache.get("token"), name=name)
        cache["token"] = token
        return (token, *self.controller.view(self.controller.begin(rater, True)),
                self.participants.get_name(token))

    def identity(self, state, token):
        if not state or self.participants.lookup(token) != state.get("rater"):
            raise ValueError("Session expired. Use Start / resume again.")


def build_app(controller, preview=False):
    access = VolunteerAccess(controller)
    browser_key = "avsg-edge-" + hashlib.sha256(access.participants.browser_secret.encode()).hexdigest()[:24]
    with gr.Blocks(title="AVSG · Relation audit", css=CSS,
                   theme=gr.themes.Soft(primary_hue="blue")) as app:
        gr.Markdown("# Audio–visual relation audit")
        gr.Markdown("Watch each assigned clip and judge each relationship independently.", elem_classes="audit-intro")
        if controller.study.get("recruitment_note"):
            gr.Markdown(controller.study["recruitment_note"])
        if preview:
            gr.Markdown("**LOCAL PREVIEW — saved clicks belong to a separate test copy and are excluded from the study.**")
        gr.Markdown("**No account or password needed.** Enter your name when you are ready to start. "
                    "Your task is remembered in this browser; use the same browser and link to resume. "
                    "Please participate once and avoid private/incognito mode.")
        with gr.Accordion("Instructions — read before starting", open=True):
            gr.Markdown(GUIDE)
        volunteer_name = gr.Textbox(label="Your name", placeholder="Enter your name",
                                    lines=1, max_lines=1,
                                    info="Required when you first start; saved with your participation record.")
        ready = gr.Checkbox(label="I have read the instructions and will judge independently.", value=False)
        start = gr.Button("Start / resume my audit", variant="primary")
        browser_token = gr.BrowserState("", storage_key=browser_key,
                                        secret=access.participants.browser_secret)
        claim_cache = gr.State({})
        session = gr.State({})
        with gr.Group(visible=False) as workspace:
            progress = gr.Markdown()
            with gr.Row():
                with gr.Column(scale=5):
                    video = gr.Video(label="Watch with audio", interactive=False, height=340,
                                     autoplay=False, show_download_button=False, show_share_button=False)
                    watched = gr.Checkbox(label="I have watched and listened to this complete clip.", value=False)
                    gr.Markdown("Replay as needed. If media cannot play, use the button below; it will not count as an edge error.")
                    media_problem = gr.Button("Cannot play video / audio")
                with gr.Column(scale=6):
                    card = gr.HTML()
                    gr.Markdown("**Choose one — your answer is saved immediately.**")
                    with gr.Row():
                        correct = gr.Button("Correct", size="lg")
                        incorrect = gr.Button("Incorrect", size="lg")
                        unsure = gr.Button("Cannot tell", size="lg")
                    with gr.Row():
                        previous = gr.Button("Previous relation")
                        resume = gr.Button("Go to next unanswered")
                    status = gr.Markdown()
        outputs = [session, workspace, progress, video, card, watched, status, previous]

        def identity(state, token):
            try:
                access.identity(state, token)
            except ValueError as e:
                raise gr.Error(str(e)) from e

        def restore_name(token):
            if not token:
                return ""
            try:
                return access.participants.get_name(token)
            except ValueError as e:
                raise gr.Error(str(e)) from e

        def begin(confirmed, token, cache, name):
            try:
                return access.begin(confirmed, token, cache, name)
            except ValueError as e:
                raise gr.Error(str(e)) from e

        def handler(verdict):
            def commit(state, seen, token):
                identity(state, token)
                try:
                    return controller.save(state, verdict, "", seen)
                except ValueError as e:
                    raise gr.Error(str(e)) from e
            return commit

        def back(state, token):
            identity(state, token)
            cid = controller.queues[state["rater"]][state["index"]][0]
            return controller.view(dict(state, index=max(0, state["index"] - 1)), prior_clip=cid,
                                   message="Showing saved values. Any unsaved selections from the previous screen were not recorded.")

        def unanswered(state, token):
            identity(state, token)
            fresh = controller.begin(state["rater"], True)
            fresh["watched"] = state["watched"]
            cid = controller.queues[state["rater"]][state["index"]][0]
            return controller.view(fresh, prior_clip=cid)

        def media_issue(state, token):
            identity(state, token)
            try:
                return controller.save(state, "uncertain", "Media would not play.", False, "media_issue")
            except ValueError as e:
                raise gr.Error(str(e)) from e

        app.load(restore_name, [browser_token], [volunteer_name], api_name=False)
        start.click(begin, [ready, browser_token, claim_cache, volunteer_name],
                    [browser_token, *outputs, volunteer_name],
                    concurrency_id="enrollment", concurrency_limit=1, trigger_mode="once", api_name=False)
        for button, verdict in ((correct, "correct"), (incorrect, "incorrect"), (unsure, "uncertain")):
            button.click(handler(verdict), [session, watched, browser_token], outputs, api_name=False)
        previous.click(back, [session, browser_token], outputs, api_name=False)
        resume.click(unanswered, [session, browser_token], outputs, api_name=False)
        media_problem.click(media_issue, [session, browser_token], outputs, api_name=False)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, action="append", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--preview", action="store_true", help="Show a preview banner; requires a separate QA study directory.")
    args = parser.parse_args()
    if args.preview and "qa" not in args.study_dir.name.lower():
        parser.error("--preview requires a separate QA study directory, not the real study.")
    # The full-path interpreter does not activate Conda's media-tool PATH.
    for subdir in ("Library/bin", "Scripts", "bin"):
        folder = Path(sys.prefix) / subdir
        if folder.is_dir():
            os.environ["PATH"] = str(folder) + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    controller = AuditController(args.study_dir, args.video_dir)
    print("Volunteer access: share one link; tasks are claimed on Start, with no account or password.", flush=True)
    print(f"Frozen study: {len(controller.clips)} clips; assignments "
          f"{ {r: len(q) for r, q in controller.queues.items()} }", flush=True)
    build_app(controller, preview=args.preview).queue().launch(
        server_name=args.host, server_port=args.port, share=args.share,
        allowed_paths=list(controller.videos.values()), blocked_paths=[str(args.study_dir.resolve())],
        show_api=False, show_error=False)


if __name__ == "__main__":
    main()
