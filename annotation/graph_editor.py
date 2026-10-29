# AVSG Distributed Annotation Tool
# Portable version of the graph editor used for human refinement.
# Pass parsed gold or prediction JSONL; outputs use the canonical gold schema.

import os
import argparse
from pathlib import Path
import json
import gradio as gr
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =================================================================
# ⚙️ CONFIGURATION
# =================================================================
ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__ or "Review and edit AVSG graphs against local clips.")
parser.add_argument("--input", type=Path, default=ROOT / "annotation/example.jsonl")
parser.add_argument("--video-dir", type=Path, default=ROOT / "clips")
parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/annotations")
parser.add_argument("--annotators", type=int, default=1)
parser.add_argument("--clips-per-annotator", type=int, default=1024)
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=7861)
parser.add_argument("--share", action="store_true")
args = parser.parse_args() if __name__ == "__main__" else parser.parse_args([])
if args.annotators < 1 or args.clips_per_annotator < 1:
    parser.error("Assignment counts must be positive.")
BASE_DIR = str(args.output_dir.resolve())
JSONL_RESULT_PATH = str(args.input.resolve())
VIDEO_DIR = str(args.video_dir.resolve())
NUM_ANNOTATORS = ACTIVE_ANNOTATORS = args.annotators
CLIPS_PER_ANNOTATOR = args.clips_per_annotator
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")


# =================================================================
# 📊 SCENE GRAPH VISUALIZATION
# =================================================================
def draw_scene_graph(json_input):
    if not json_input:
        return None
    try:
        if isinstance(json_input, str):
            if "```json" in json_input:
                json_input = json_input.split("```json")[1].split("```")[0]
            data = json.loads(json_input)
            if len(data.keys()) == 1 and "_" in list(data.keys())[0]:
                data = list(data.values())[0]
        else:
            data = json_input

        # Also unwrap if top-level is a single segment key with nodes inside
        if isinstance(data, dict) and "nodes" not in data and len(data) == 1:
            inner = list(data.values())[0]
            if isinstance(inner, dict) and ("nodes" in inner or "edges" in inner):
                data = inner

        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        if not nodes:
            return None

        # Close ALL existing figures first to prevent leaks / stale renders
        plt.close('all')

        G = nx.DiGraph()
        color_v_node, color_a_node = "#87CEEB", "#FF4500"
        color_v2v_edge, color_v2a_edge, color_a2a_edge = "#1E90FF", "#FF6347", "#9B59B6"

        node_ids = set()
        for node in nodes:
            G.add_node(node["id"], label=node.get("class", node["id"]))
            node_ids.add(node["id"])

        # Only add edges whose source AND target exist as nodes
        for edge in edges:
            if edge["source"] in node_ids and edge["target"] in node_ids:
                G.add_edge(edge["source"], edge["target"], label=edge.get("edge", ""))

        if len(G.nodes()) == 0:
            return None

        node_colors = [color_a_node if str(n).startswith("a") else color_v_node for n in G.nodes()]
        edge_colors = [
            color_a2a_edge if str(u).startswith("a") and str(v).startswith("a")
            else color_v2a_edge if str(v).startswith("a")
            else color_v2v_edge
            for u, v in G.edges()
        ]

        fig, ax = plt.subplots(figsize=(12, 8))

        best_pos, best_score = None, float('inf')
        layout_attempts = [
            nx.spring_layout(G, k=2.5, iterations=100, seed=42),
            nx.spring_layout(G, k=3.0, iterations=100, seed=123),
        ]
        try:
            if len(G.nodes()) > 1:
                layout_attempts.append(nx.kamada_kawai_layout(G))
        except:
            pass
        try:
            visual_nodes = [n for n in G.nodes() if str(n).startswith("v")]
            audio_nodes = [n for n in G.nodes() if str(n).startswith("a")]
            if visual_nodes and audio_nodes:
                layout_attempts.append(nx.shell_layout(G, nlist=[visual_nodes, audio_nodes]))
        except:
            pass

        for pos in layout_attempts:
            if not pos:
                continue
            min_dist = float('inf')
            nodes_list = list(pos.keys())
            for i in range(len(nodes_list)):
                for j in range(i + 1, len(nodes_list)):
                    n1, n2 = nodes_list[i], nodes_list[j]
                    dist = ((pos[n1][0] - pos[n2][0])**2 + (pos[n1][1] - pos[n2][1])**2)**0.5
                    min_dist = min(min_dist, dist)
            if min_dist > best_score or best_pos is None:
                best_score = min_dist
                best_pos = pos

        pos = best_pos if best_pos else nx.spring_layout(G, k=2.5, iterations=100)

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3500, alpha=0.9, ax=ax)
        for (u, v), color in zip(G.edges(), edge_colors):
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=[color],
                                   width=2.5, arrowsize=20, arrowstyle='-|>',
                                   connectionstyle='arc3,rad=0.2', alpha=0.7,
                                   min_source_margin=25, min_target_margin=25, ax=ax)

        labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold', ax=ax)

        edge_labels = nx.get_edge_attributes(G, 'label')
        for edge, label in edge_labels.items():
            x = (pos[edge[0]][0] + pos[edge[1]][0]) / 2
            y = (pos[edge[0]][1] + pos[edge[1]][1]) / 2
            ax.text(x, y, label, fontsize=9, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                              edgecolor='lightgray', alpha=0.95))

        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        ax.legend(handles=[
            Patch(facecolor=color_v_node, edgecolor='gray', label='Visual node'),
            Patch(facecolor=color_a_node, edgecolor='gray', label='Audio node'),
            Line2D([0], [0], color=color_v2v_edge, linewidth=2, label='Visual → Visual'),
            Line2D([0], [0], color=color_v2a_edge, linewidth=2, label='Visual → Audio'),
            Line2D([0], [0], color=color_a2a_edge, linewidth=2, label='Audio ↔ Audio'),
        ], loc='upper left', fontsize=9, framealpha=0.9)

        ax.axis('off')
        fig.tight_layout()
        return fig
    except Exception as e:
        print(f"Error drawing graph: {e}")
        plt.close('all')
        return None


# =================================================================
# 💾 DATA HELPERS
# =================================================================
def key_to_video_filename(raw_key):
    """The key is already the video filename (e.g. '-K_V8L2ePqI_147.614_150.216.mp4').
       Ensure it ends with .mp4."""
    clean = raw_key
    if clean.startswith("req_"):
        clean = clean[4:]
    if not clean.endswith(".mp4"):
        clean += ".mp4"
    return clean


def key_to_save_name(raw_key):
    """Derive a clean save-name for the JSON output file (strip .mp4 and any req_ prefix).
       e.g. '-K_V8L2ePqI_147.614_150.216.mp4' -> '-K_V8L2ePqI_147.614_150.216'"""
    clean = raw_key
    if clean.startswith("req_"):
        clean = clean[4:]
    if clean.endswith(".mp4"):
        clean = clean[:-4]
    return clean


def extract_caption(entry):
    return entry.get("caption", "")


def extract_graph_json(entry):
    return {"nodes": entry.get("ground_truth_nodes", entry.get("nodes", [])),
            "edges": entry.get("ground_truth_edges", entry.get("edges", []))}


def load_all_entries():
    entries = []
    if not os.path.isfile(JSONL_RESULT_PATH):
        raise FileNotFoundError(f"Input JSONL not found: {JSONL_RESULT_PATH}")
    with open(JSONL_RESULT_PATH, encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            entry = json.loads(line)
            filename = entry["filename"]
            if not filename or "/" in filename or "\\" in filename or not filename.endswith(".mp4"):
                raise ValueError("Each filename must be an MP4 basename.")
            entry["key"] = filename
            entries.append(entry)
    if len({entry["key"] for entry in entries}) != len(entries):
        raise ValueError("Duplicate clip filenames in input.")
    return sorted(entries, key=lambda entry: entry["key"])


ALL_ENTRIES = load_all_entries()
TOTAL_CLIPS = len(ALL_ENTRIES)
print(f"Loaded {TOTAL_CLIPS} clips from {JSONL_RESULT_PATH}")
print(f"With {NUM_ANNOTATORS} annotators x {CLIPS_PER_ANNOTATOR} clips = {NUM_ANNOTATORS * CLIPS_PER_ANNOTATOR} slots")
if TOTAL_CLIPS < NUM_ANNOTATORS * CLIPS_PER_ANNOTATOR:
    print(f"WARNING: Only {TOTAL_CLIPS} clips available — some annotators will get fewer than {CLIPS_PER_ANNOTATOR}")


def get_assigned_entries(annotator_id):
    """Annotator i (1-indexed) gets a deterministic non-overlapping slice of the sorted input."""
    start = (annotator_id - 1) * CLIPS_PER_ANNOTATOR
    end = start + CLIPS_PER_ANNOTATOR
    return ALL_ENTRIES[start:end]


def get_output_dir(annotator_id):
    d = os.path.join(BASE_DIR, str(annotator_id))
    os.makedirs(d, exist_ok=True)
    return d


def node_display(node):
    return f"{node.get('class', node['id'])} ({node.get('attribute', '')})"


def renumber_nodes(data):
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    v_nodes = [n for n in nodes if n["id"].startswith("v")]
    a_nodes = [n for n in nodes if n["id"].startswith("a")]
    id_map = {}
    for i, n in enumerate(v_nodes, 1):
        id_map[n["id"]] = f"v{i}"
        n["id"] = f"v{i}"
    for i, n in enumerate(a_nodes, 1):
        id_map[n["id"]] = f"a{i}"
        n["id"] = f"a{i}"
    for e in edges:
        e["source"] = id_map.get(e["source"], e["source"])
        e["target"] = id_map.get(e["target"], e["target"])
    data["nodes"] = v_nodes + a_nodes
    return data


# =================================================================
# GRADIO UI
# =================================================================
with gr.Blocks(title="AVSG Distributed Annotation", theme=gr.themes.Default(primary_hue="orange")) as demo:
    gr.Markdown("# 🎬 AVSG Distributed Annotation Tool")
    gr.Markdown(f"**{TOTAL_CLIPS}** total clips loaded  ·  **{ACTIVE_ANNOTATORS}** annotator slots open  ·  **{CLIPS_PER_ANNOTATOR}** clips each")

    # -- Annotator Login --
    with gr.Row():
        with gr.Column(scale=2):
            annotator_input = gr.Number(label=f"Annotator ID (1–{ACTIVE_ANNOTATORS})", value=1,
                                        precision=0, minimum=1, maximum=ACTIVE_ANNOTATORS)
        with gr.Column(scale=1):
            login_btn = gr.Button("🚀 Start / Resume", variant="primary", size="lg")

    login_status = gr.Markdown("*Enter your assigned annotator ID and click Start.*")

    # -- Hidden states --
    annotator_id_state = gr.State(None)
    assigned_clips_state = gr.State([])     # list of raw keys
    current_idx_state = gr.State(0)
    caption_state = gr.State("")

    # -- Main workspace (hidden until login) --
    with gr.Group(visible=False) as workspace:
        with gr.Row():
            progress_md = gr.Markdown("**Clip 1 / 50**")
            saved_count_md = gr.Markdown("**Saved: 0 / 50**")

        with gr.Row():
            # -- LEFT COLUMN --
            with gr.Column(scale=1):
                clip_picker = gr.Dropdown(label="Select Clip ID", choices=[], value=None)
                video_output = gr.Video(label="Clip Source", height=320)
                caption_display = gr.Textbox(label="📄 Caption", interactive=False, lines=3)
                with gr.Row():
                    prev_btn = gr.Button("⬅️ Previous", variant="secondary")
                    next_btn = gr.Button("Next ➡️", variant="secondary")

                gr.Markdown("### 📝 JSON Annotation")
                code_editor = gr.Code(language="json", interactive=True, lines=20)
                save_btn = gr.Button("💾 Commit & Save", variant="primary")
                status_msg = gr.Markdown("*Ready.*")

            # -- RIGHT COLUMN --
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Live AVSG Visualization")
                viz_plot = gr.Plot()

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Node Management")
                        with gr.Row():
                            new_node_id = gr.Textbox(label="Node ID", placeholder="e.g. v3 (blank=auto)")
                            new_node_class = gr.Textbox(label="Node Class", placeholder="e.g. man, music")
                            new_node_attribute = gr.Textbox(label="Attribute", placeholder="e.g. woman in red")
                        node_av = gr.Radio(["Visual (v)", "Audio (a)"], label="Type", value="Visual (v)")
                        add_node_btn = gr.Button("➕ Add Node")
                        node_del_drop = gr.Dropdown(label="Delete Node (by Class & Attribute)")
                        del_node_btn = gr.Button("🗑️ Delete Node", variant="stop")

                    with gr.Column():
                        gr.Markdown("#### Edge Management")
                        with gr.Row():
                            src_drop = gr.Dropdown(label="Source")
                            tgt_drop = gr.Dropdown(label="Target")
                        edge_rel = gr.Textbox(label="Edge", placeholder="e.g. holding, responds_to")
                        add_edge_btn = gr.Button("➕ Add Edge")
                        edge_del_drop = gr.Dropdown(label="Delete Edge")
                        del_edge_btn = gr.Button("🗑️ Delete Edge", variant="stop")

    # =================================================================
    # INTERNAL HELPERS
    # =================================================================
    def sync_ui_from_json(json_str):
        try:
            data = json.loads(json_str)
            # Handle wrapped format just in case
            if len(data.keys()) == 1 and "nodes" not in data:
                inner = list(data.values())[0]
                if isinstance(inner, dict):
                    data = inner

            nodes = data.get("nodes", [])
            edges = data.get("edges", [])
            id_to_display = {n["id"]: node_display(n) for n in nodes}
            node_displays = [node_display(n) for n in nodes]
            edge_display_labels = []
            for e in edges:
                s_lbl = id_to_display.get(e['source'], e['source'])
                t_lbl = id_to_display.get(e['target'], e['target'])
                edge_display_labels.append(f"{s_lbl} -> {t_lbl} ({e.get('edge', '')})")
            plot = draw_scene_graph(data)
            return (plot,
                    gr.update(choices=node_displays),
                    gr.update(choices=node_displays),
                    gr.update(choices=node_displays),
                    gr.update(choices=edge_display_labels))
        except:
            return (None, gr.update(choices=[]), gr.update(choices=[]),
                    gr.update(choices=[]), gr.update(choices=[]))

    def get_id_from_display(display, nodes):
        for n in nodes:
            if node_display(n) == display:
                return n["id"]
        return display

    def count_saved(annotator_id, clips):
        out_dir = get_output_dir(annotator_id)
        return sum(1 for key in clips
                   if os.path.exists(os.path.join(out_dir, f"{key_to_save_name(key)}.json")))

    def find_entry_by_key(key):
        for e in ALL_ENTRIES:
            if e.get("key") == key:
                return e
        return None

    def load_clip(annotator_id, clips, idx):
        """Load clip by index. Prefer saved annotation if it exists."""
        if not clips or idx < 0 or idx >= len(clips):
            return None, "", "", "", None, gr.update(), gr.update(), gr.update(), gr.update()

        raw_key = clips[idx]
        save_name = key_to_save_name(raw_key)
        video_file = key_to_video_filename(raw_key)
        out_dir = get_output_dir(annotator_id)
        saved_path = os.path.join(out_dir, f"{save_name}.json")

        entry = find_entry_by_key(raw_key)
        v_path = os.path.join(VIDEO_DIR, video_file) if entry else None
        caption = extract_caption(entry) if entry else ""

        # If previously saved, load that
        if os.path.exists(saved_path):
            with open(saved_path, "r", encoding="utf-8") as f:
                saved = json.loads(f.read())
            graph_data = {
                "nodes": saved.get("ground_truth_nodes", []),
                "edges": saved.get("ground_truth_edges", [])
            }
            clean_text = json.dumps(graph_data, indent=4)
        else:
            # Load from Gemini response
            if entry:
                graph_data = extract_graph_json(entry)
                clean_text = json.dumps(graph_data, indent=4)
            else:
                clean_text = '{"nodes": [], "edges": []}'

        plot, n_up, s_up, t_up, e_up = sync_ui_from_json(clean_text)
        return v_path, clean_text, caption, caption, plot, n_up, s_up, t_up, e_up

    # =================================================================
    # CALLBACKS
    # =================================================================
    def handle_login(aid):
        aid = int(aid)
        if aid < 1 or aid > ACTIVE_ANNOTATORS:
            return (gr.update(visible=False), None, [], 0, "",
                    f"❌ Invalid ID. Enter 1–{ACTIVE_ANNOTATORS}.", gr.update(), gr.update(),
                    None, None, "", "", None,
                    gr.update(), gr.update(), gr.update(), gr.update())

        entries = get_assigned_entries(aid)
        if not entries:
            return (gr.update(visible=False), None, [], 0, "",
                    f"❌ No clips assigned to annotator {aid} (only {TOTAL_CLIPS} clips loaded).",
                    gr.update(), gr.update(),
                    None, None, "", "", None,
                    gr.update(), gr.update(), gr.update(), gr.update())

        clips = [e["key"] for e in entries]
        saved = count_saved(aid, clips)
        status = (f"✅ **Annotator {aid}** logged in — "
                  f"**{len(clips)}** clips assigned, "
                  f"**{saved}/{len(clips)}** already saved.")

        v_path, clean_text, caption, _, plot, n_up, s_up, t_up, e_up = load_clip(aid, clips, 0)

        return (
            gr.update(visible=True),       # workspace
            aid,                            # annotator_id_state
            clips,                          # assigned_clips_state
            0,                              # current_idx_state
            caption,                        # caption_state
            status,                         # login_status
            f"**Clip 1 / {len(clips)}**",   # progress_md
            f"**Saved: {saved} / {len(clips)}**",  # saved_count_md
            v_path,                         # video_output
            clean_text,                     # code_editor
            gr.update(choices=[key_to_save_name(c) for c in clips],
                      value=key_to_save_name(clips[0])),  # clip_picker
            caption,                        # caption_display
            plot,                           # viz_plot
            n_up, s_up, t_up, e_up          # dropdowns
        )

    login_btn.click(
        handle_login,
        inputs=[annotator_input],
        outputs=[workspace, annotator_id_state, assigned_clips_state,
                 current_idx_state, caption_state, login_status,
                 progress_md, saved_count_md,
                 video_output, code_editor, clip_picker, caption_display,
                 viz_plot, node_del_drop, src_drop, tgt_drop, edge_del_drop]
    )

    # -- Navigation --
    def navigate(aid, clips, idx, direction):
        if not clips:
            return (idx, gr.update(), gr.update(), gr.update(),
                    None, None, "", "", None,
                    gr.update(), gr.update(), gr.update(), gr.update(), "")
        new_idx = max(0, min(idx + direction, len(clips) - 1))
        v_path, clean_text, caption, _, plot, n_up, s_up, t_up, e_up = load_clip(aid, clips, new_idx)
        saved = count_saved(aid, clips)
        return (
            new_idx, caption,
            f"**Clip {new_idx + 1} / {len(clips)}**",
            f"**Saved: {saved} / {len(clips)}**",
            v_path, clean_text, key_to_save_name(clips[new_idx]), caption,
            plot, n_up, s_up, t_up, e_up, "*Ready.*"
        )

    nav_outputs = [current_idx_state, caption_state, progress_md, saved_count_md,
                   video_output, code_editor, clip_picker, caption_display,
                   viz_plot, node_del_drop, src_drop, tgt_drop, edge_del_drop, status_msg]

    prev_btn.click(
        lambda aid, clips, idx: navigate(aid, clips, idx, -1),
        inputs=[annotator_id_state, assigned_clips_state, current_idx_state],
        outputs=nav_outputs
    )
    next_btn.click(
        lambda aid, clips, idx: navigate(aid, clips, idx, +1),
        inputs=[annotator_id_state, assigned_clips_state, current_idx_state],
        outputs=nav_outputs
    )

    # -- Dropdown clip selection --
    def handle_clip_pick(selected_name, aid, clips):
        """When user selects a clip from the dropdown, jump to it."""
        if not selected_name or not clips:
            return (0, gr.update(), gr.update(), gr.update(),
                    None, None, "", None,
                    gr.update(), gr.update(), gr.update(), gr.update(), "")
        # Find index by matching save_name
        target_idx = 0
        for i, key in enumerate(clips):
            if key_to_save_name(key) == selected_name:
                target_idx = i
                break
        v_path, clean_text, caption, _, plot, n_up, s_up, t_up, e_up = load_clip(aid, clips, target_idx)
        saved = count_saved(aid, clips)
        return (
            target_idx, caption,
            f"**Clip {target_idx + 1} / {len(clips)}**",
            f"**Saved: {saved} / {len(clips)}**",
            v_path, clean_text, caption,
            plot, n_up, s_up, t_up, e_up, "*Ready.*"
        )

    clip_pick_outputs = [current_idx_state, caption_state, progress_md, saved_count_md,
                         video_output, code_editor, caption_display,
                         viz_plot, node_del_drop, src_drop, tgt_drop, edge_del_drop, status_msg]

    clip_picker.change(handle_clip_pick,
                       inputs=[clip_picker, annotator_id_state, assigned_clips_state],
                       outputs=clip_pick_outputs)

    # -- Live viz update on JSON edit --
    code_editor.change(sync_ui_from_json, inputs=[code_editor],
                       outputs=[viz_plot, node_del_drop, src_drop, tgt_drop, edge_del_drop])

    # -- Node editing --
    def add_node_logic(custom_id, node_class, node_attribute, av_choice, current_json):
        data = json.loads(current_json)
        prefix = "v" if "Visual" in av_choice else "a"
        if custom_id and custom_id.strip():
            new_id = custom_id.strip()
        else:
            existing_ids = [
                int(n["id"][1:]) for n in data["nodes"]
                if n["id"].startswith(prefix) and n["id"][1:].isdigit()
            ]
            new_id = f"{prefix}{max(existing_ids) + 1 if existing_ids else 1}"
        data["nodes"].append({
            "id": new_id,
            "class": node_class.strip() if node_class else "",
            "attribute": node_attribute.strip() if node_attribute else ""
        })
        data = renumber_nodes(data)
        return json.dumps(data, indent=4)

    add_node_btn.click(add_node_logic,
                       [new_node_id, new_node_class, new_node_attribute, node_av, code_editor],
                       [code_editor])

    def del_node_logic(display, current_json):
        data = json.loads(current_json)
        target_id = get_id_from_display(display, data["nodes"])
        data["nodes"] = [n for n in data["nodes"] if n["id"] != target_id]
        data["edges"] = [e for e in data["edges"]
                         if e["source"] != target_id and e["target"] != target_id]
        data = renumber_nodes(data)
        return json.dumps(data, indent=4)

    del_node_btn.click(del_node_logic, [node_del_drop, code_editor], [code_editor])

    # -- Edge editing --
    def add_edge_logic(src_display, tgt_display, rel, current_json):
        data = json.loads(current_json)
        sid = get_id_from_display(src_display, data["nodes"])
        tid = get_id_from_display(tgt_display, data["nodes"])
        data["edges"].append({"source": sid, "edge": rel, "target": tid})
        return json.dumps(data, indent=4)

    add_edge_btn.click(add_edge_logic, [src_drop, tgt_drop, edge_rel, code_editor], [code_editor])

    def del_edge_logic(edge_display_str, current_json):
        data = json.loads(current_json)
        nodes = data.get("nodes", [])
        id_to_display = {n["id"]: node_display(n) for n in nodes}
        new_edges = []
        for e in data["edges"]:
            s_lbl = id_to_display.get(e['source'], e['source'])
            t_lbl = id_to_display.get(e['target'], e['target'])
            check_str = f"{s_lbl} -> {t_lbl} ({e.get('edge', '')})"
            if check_str != edge_display_str:
                new_edges.append(e)
        data["edges"] = new_edges
        return json.dumps(data, indent=4)

    del_edge_btn.click(del_edge_logic, [edge_del_drop, code_editor], [code_editor])

    # -- Save --
    def save_output(aid, clips, idx, content, caption):
        if aid is None:
            return "❌ Not logged in.", gr.update()
        try:
            graph = json.loads(content)
        except Exception:
            return "❌ Invalid JSON — fix errors before saving.", gr.update()

        # Handle wrapped format
        if len(graph.keys()) == 1 and "nodes" not in graph:
            inner = list(graph.values())[0]
            if isinstance(inner, dict):
                graph = inner

        raw_key = clips[idx]
        save_name = key_to_save_name(raw_key)
        video_file = key_to_video_filename(raw_key)

        output = {
            "filename": video_file,
            "caption": caption,
            "ground_truth_nodes": graph.get("nodes", []),
            "ground_truth_edges": graph.get("edges", [])
        }

        out_dir = get_output_dir(aid)
        filepath = os.path.join(out_dir, f"{save_name}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(json.dumps(output, ensure_ascii=False))

        saved = count_saved(aid, clips)
        return (f"✨ Saved `{save_name}.json` → `{out_dir}`",
                f"**Saved: {saved} / {len(clips)}**")

    save_btn.click(save_output,
                   inputs=[annotator_id_state, assigned_clips_state,
                           current_idx_state, code_editor, caption_state],
                   outputs=[status_msg, saved_count_md])


if __name__ == "__main__":
    demo.launch(server_name=args.host, server_port=args.port, share=args.share,
                allowed_paths=[VIDEO_DIR], blocked_paths=[BASE_DIR], show_api=False)