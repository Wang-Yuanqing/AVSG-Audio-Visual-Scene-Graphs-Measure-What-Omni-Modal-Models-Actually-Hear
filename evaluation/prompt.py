"""Turn the v2.29 annotation rubric into a model prompt.

Principles (see avsg_annotation_rubric_v2_29.md):
  * vocab tables    -> kept verbatim (closed set; exact labels required)
  * substitutions   -> full table kept (fairness: don't penalize un-stated mappings)
  * prose rules     -> condensed imperative bullets (decision-relevant only)
  * Common Patterns -> few-shot examples in the EXACT output JSON
  * added           -> output schema, "JSON only", id conventions, "never invent"
  * dropped         -> changelog, vocab-summary, eval code, tooling notes
  * NOT added       -> dataset statistics (would be benchmark-gaming)

Verified against annotations/1k_result.jsonl: all 74 visual + 36 audio classes are
used, 0 out-of-vocab, mean 5.17 nodes / clip, every node has an attribute.

The SYSTEM prompt is identical for both settings. Only the USER turn differs: the
caption is appended for the caption-assisted setting, omitted for raw audio+video.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vocab import ALL_CLASSES, ALL_EDGES  # noqa: E402

_VISUAL_GROUPS = [
    ("People", ["man", "woman", "boy", "girl", "baby", "crowd", "band", "team"]),
    ("Animals", ["dog", "cat", "horse", "bird", "cow"]),
    ("Objects", ["phone", "camera", "microphone", "guitar", "ball", "board", "book",
                 "cup", "bottle", "bowl", "food", "bag", "box", "paper", "drum", "gun",
                 "rod", "rope", "bat", "toy", "plate", "spoon", "knife", "brush", "flag",
                 "piano", "card", "can", "scissors", "dumbbell", "shoe", "flute", "pen",
                 "trumpet", "violin", "fish", "sword", "axe", "pan", "hammer", "accordion"]),
    ("Vehicles", ["car", "motorcycle", "bicycle", "boat", "bus"]),
    ("Furniture", ["chair", "bed", "desk", "door", "window"]),
    ("Scene", ["screen", "sign", "water", "grass", "tree", "field", "court", "fire", "building"]),
    ("Body", ["hand"]),
]
_AUDIO_GROUPS = [
    ("Voice", ["male_speech", "female_speech", "laughter", "crying", "male_singing",
               "female_singing", "child_singing", "panting", "coughing"]),
    ("Music (instrument visible)", ["guitar_sound", "piano_sound", "drum_sound",
                                    "violin_sound", "trumpet_sound", "flute_sound", "accordion_sound"]),
    ("Background (always isolated)", ["background_music", "background_sound", "sound_effect"]),
    ("Crowd", ["crowd_sound", "applause"]),
    ("Animal", ["bark", "meow", "chirp"]),
    ("Mechanical", ["engine", "footstep", "phone_ring", "whistle", "gunshot", "splash",
                    "door_sound", "crashing_sound"]),
    ("Environment", ["wind", "rain"]),
    ("Contact", ["crinkling", "scraping"]),
]


def _render(groups):
    return "\n".join(f"  {name}: {', '.join(items)}" for name, items in groups)


_RULES = f"""You are an expert at Audio-Visual Scene Graph annotation.

TASK
Given a short video clip (with audio), produce a scene graph capturing the MAIN
IDEA of the clip: what you SEE (visual nodes), what you HEAR (audio nodes), and
the relationships (edges) between them. All edges are HUMAN-CENTERED: they flow
from a human/group node out to objects and sounds (exception: autonomous objects
like a ringing phone or a revving engine are their own source).

OUTPUT
Return ONLY one JSON object, no prose and no markdown fences:
{{"nodes": [{{"id": "v1", "class": "<visual class>", "attribute": "<short phrase>"}},
           {{"id": "a1", "class": "<audio class>", "attribute": "<short phrase>"}}],
 "edges": [{{"source": "<id>", "edge": "<edge>", "target": "<id>"}}]}}
Visual ids are v1,v2,...; audio ids are a1,a2,.... Use ONLY the classes and edges
listed below; never invent new ones. Give every node a short "attribute" (free
text, not scored).

NODE COUNT
2-10 nodes total (visual+audio); target 4-6 (about 2-3 visual + 2-3 audio).
Annotate important things first; do not pad to a count; never exceed 10.

VISUAL NODES
DO annotate: people, animals, instruments being played, objects being interacted
with, spatial anchors (car, court). Include people who are part of the event even
if silent. DON'T annotate: static/uninvolved objects, background clutter, random
passersby. Put clothing/footwear/hats/jewelry in the person's attribute, not as nodes.
Visual classes (use exactly):
{_render(_VISUAL_GROUPS)}
Group nodes (do not split into individuals): crowd = >=3 people doing the same thing;
band = musicians performing together incl. the singer; team = players competing.
If everyone does the same thing -> group node; if someone has a distinct role ->
give that person an individual node.
Special rules:
  - Age/gender by appearance: baby (0-2), boy/girl (2-12), man/woman (13+); use
    baby only when gender is not identifiable.
  - hand: only when the person is NOT visible; otherwise the person is the source.
  - microphone: only when held by someone (skip a mic on a stand).
  - Instruments: one node per instrument type; sub-type goes in attribute.
Common substitutions (output the listed class, put the real item in attribute):
  banjo/ukulele/bass guitar -> guitar; cello/double bass/viola -> violin;
  saxophone -> trumpet; electronic keyboard -> piano; basketball/soccer ball -> ball;
  funnel -> cup; wheelchair -> chair; power cord/cable/leash -> rope;
  lake/river/pond -> water; wok/frying pan/skillet -> pan; bucket/pot/kettle -> bowl;
  microwave/fridge -> box; truck -> car; skateboard/surfboard/sled/snowboard/ski -> board;
  hiking pole/walking stick/shovel/umbrella -> rod; gavel/mallet/sledgehammer -> hammer;
  golf club/hockey stick/cricket bat -> bat; balloon/sculpture/toy figure -> toy;
  TV/laptop/monitor -> screen; train/subway -> bus; drone/ATV -> motorcycle;
  cash bills -> paper; vinyl record/DVD -> card; cigarette/vape/pipe -> bottle.
For anything not in the vocab or this table, substitute the closest listed class or
skip it if it is not central.

AUDIO NODES
Create nodes for the obvious sounds that capture the main idea.
Audio classes (use exactly):
{_render(_AUDIO_GROUPS)}
Key rules:
  - male_speech/female_speech = ALL linguistic speech (calm, shouting, whispering);
    tone goes in attribute. man/boy -> male_*, woman/girl -> female_*, baby -> crying
    or laughter (child_singing for cooing).
  - Visible instrument -> [instrument]_sound, with a "playing" V2A edge FROM THE
    HUMAN (never from the instrument). Instrument never visible -> background_music.
  - background_music, background_sound, sound_effect are ALWAYS isolated (no edges).
  - crowd_sound = all crowd vocalizations (cheering, chanting, singing along,
    murmuring); use applause only for distinct clapping-as-reaction.
  - One speech node per distinguishable speaker; shared sounds (laughter,
    guitar_sound) = one node.
  - Orphan audio nodes are fine: a sound with no visible source (off-screen voice,
    hidden music, ambient) is a node with no edge. A clip may have audio nodes and
    zero V2A edges.

EDGES
Most specific edge wins; action edges beat spatial edges; do not stack redundant
edges. Store symmetric edges once with the LOWER id as source.
V2V (visual -> visual):
  Action: eating, drinking_from, filming (camera visible in hand), aiming_at,
    fishing, riding (horse/motorcycle/bicycle; cars use inside), swimming_in,
    hitting (non-musical impact), kicking, throwing, playing_with [sym, mutual play],
    playing (person -> instrument), pointing_at, wearing, holding (grip; subsumes
    touching), sitting_on, standing_on, lying_on, watching (person -> screen/
    performer/animal; never person<->person in conversation), walking_toward,
    walking_through, touching (last resort).
  Spatial: inside (all people in cars), on, above, surrounded_by (encloser is
    physically all around), in_front_of (source closer to camera), next_to [sym].
  in_front_of: human -> object, human is source; band/performer -> crowd at a
    concert (this replaces "crowd watching band"); people facing each other -> use
    next_to, never in_front_of.
  NEVER use: behind, under, beneath, below, looking_at, vocalizing, striking,
    strumming, surrounding, shouting, shooting, dancing.
V2A (visual -> audio, from the HUMAN): singing, speaking, crying, laughing, panting,
  coughing, playing (all instruments), hitting, cheering, scraping, crinkling,
  kicking, clapping, barking, meowing, chirping, splashing, pouring, revving,
  slamming, clicking, stepping (-> footstep), ringing (phone -> phone_ring; phone
  is the source).
A2A (audio <-> audio): in_dialogue_with [sym, back-and-forth conversation],
  responds_to (one-way, non-conversational, e.g. a crowd reacting to a speaker).
  No edge if sounds merely co-occur; background audio is always isolated.
Shared edges: playing, hitting, kicking, singing, crying, coughing, panting,
  laughing work as BOTH V2V and V2A - visual target makes it V2V, audio target V2A.
"""

# Few-shot examples (rubric "Common Patterns" -> exact output JSON)
_EXAMPLES = [
    ("Concert (band in_front_of crowd; crowd_sound isolated)", {
        "nodes": [
            {"id": "v1", "class": "band", "attribute": "rock band on stage"},
            {"id": "v2", "class": "crowd", "attribute": "audience"},
            {"id": "v3", "class": "guitar", "attribute": "electric guitar"},
            {"id": "v4", "class": "drum", "attribute": "drum kit"},
            {"id": "a1", "class": "guitar_sound", "attribute": "guitar riff"},
            {"id": "a2", "class": "drum_sound", "attribute": "drum beat"},
            {"id": "a3", "class": "male_singing", "attribute": "lead vocalist"},
            {"id": "a4", "class": "crowd_sound", "attribute": "crowd cheering"}],
        "edges": [
            {"source": "v1", "edge": "in_front_of", "target": "v2"},
            {"source": "v1", "edge": "playing", "target": "v3"},
            {"source": "v1", "edge": "playing", "target": "v4"},
            {"source": "v1", "edge": "playing", "target": "a1"},
            {"source": "v1", "edge": "playing", "target": "a2"},
            {"source": "v1", "edge": "singing", "target": "a3"}]}),
    ("Conversation (inside, next_to, speaking, in_dialogue_with)", {
        "nodes": [
            {"id": "v1", "class": "man", "attribute": "driver"},
            {"id": "v2", "class": "woman", "attribute": "passenger"},
            {"id": "v3", "class": "car", "attribute": "car interior"},
            {"id": "a1", "class": "male_speech", "attribute": "man talking"},
            {"id": "a2", "class": "female_speech", "attribute": "woman talking"}],
        "edges": [
            {"source": "v1", "edge": "inside", "target": "v3"},
            {"source": "v2", "edge": "inside", "target": "v3"},
            {"source": "v1", "edge": "next_to", "target": "v2"},
            {"source": "v1", "edge": "speaking", "target": "a1"},
            {"source": "v2", "edge": "speaking", "target": "a2"},
            {"source": "a1", "edge": "in_dialogue_with", "target": "a2"}]}),
    ("Unboxing (hand since person not visible; off-screen narrator isolated)", {
        "nodes": [
            {"id": "v1", "class": "hand", "attribute": "hands opening box"},
            {"id": "v2", "class": "box", "attribute": "cardboard box"},
            {"id": "a1", "class": "crinkling", "attribute": "cardboard tearing"},
            {"id": "a2", "class": "male_speech", "attribute": "man narrating off-screen"}],
        "edges": [
            {"source": "v1", "edge": "holding", "target": "v2"},
            {"source": "v1", "edge": "crinkling", "target": "a1"}]}),
]


def build_system_prompt():
    parts = [_RULES, "\nEXAMPLES"]
    for title, graph in _EXAMPLES:
        parts.append(f"# {title}\n{json.dumps(graph, ensure_ascii=False)}")
    return "\n".join(parts)


def build_user_text(caption=None):
    base = "Analyze this clip and output its audio-visual scene graph as the JSON object."
    if caption:
        return base + f'\n\nReference caption (a hint only; the video and audio are the ground truth): "{caption}"'
    return base


SYSTEM_PROMPT = build_system_prompt()


if __name__ == "__main__":
    sp = SYSTEM_PROMPT
    missing = [t for t in (ALL_CLASSES | ALL_EDGES) if t not in sp]
    print(f"system prompt: {len(sp)} chars, ~{len(sp)//4} tokens")
    print(f"vocab terms in prompt: {len(ALL_CLASSES | ALL_EDGES) - len(missing)}/{len(ALL_CLASSES | ALL_EDGES)}")
    if missing:
        print("MISSING:", sorted(missing))
    print("raw user text:", build_user_text())
