# AVSG Annotation Rubric — v2.29 (final)

Capture the main idea of each video clip as a graph with visual nodes, audio nodes, and edges connecting them. Human-centered: edges flow from humans to objects, from humans to sounds.

**Note on clips:** clips are pre-segmented short video segments (median ~14s) cut from longer source videos. Each clip is annotated independently. Sounds and entities not visible in the clip's frame range may still be annotated as nodes without spatial/visual edges.

---

## 1. Visual Nodes

Annotate **only what matters** to understand the clip. **Hard limits: 2–10 nodes** (visual + audio combined). Target **4–6 typical**, with roughly 2–3 visual and 2–3 audio. Annotate important things first; don't pad with irrelevant nodes just to hit a count, and don't exceed 10 nodes.

**DO annotate:** people, animals, instruments being played, objects being interacted with, spatial anchors (car, court). Include people who are part of the event even if silent (a partner standing next to the speaker, a friend watching).

**DON'T annotate:** static uninvolved objects (sky, wall, phone in crowd's hand, bottle someone sipped from once), background clutter, objects that aren't moving or making sound, random passersby in the background who are clearly not part of the event. Skip clothing, footwear, hats, jewelry, and other worn items — describe them in the person's attribute instead.

**Inventing classes:** prefer substitution with an existing class plus an attribute when the object is functionally similar (banjo → guitar). Only invent a new class when no existing class fits AND the object is central to the clip.

### Classes

| Category | Classes |
|----------|---------|
| People | man, woman, boy, girl, baby, crowd, band, team |
| Animals | dog, cat, horse, bird, cow |
| Objects | phone, camera, microphone, guitar, ball, board, book, cup, bottle, bowl, food, bag, box, paper, drum, gun, rod, rope, bat, toy, plate, spoon, knife, brush, flag, piano, card, can, scissors, dumbbell, shoe, flute, pen, trumpet, violin, fish, sword, axe, pan, hammer, accordion |
| Vehicles | car, motorcycle, bicycle, boat, bus |
| Furniture | chair, bed, desk, door, window |
| Scene Elements | screen, sign, water, grass, tree, field, court, fire, building |
| Body Parts | hand |

### Group Nodes

- **crowd** = 3+ people doing the same thing (watching, cheering, standing). Don't split individuals.
- **band** = musicians performing together. Includes the singer. Don't split into individual members.
- **team** = players competing in a sport. Don't split into individual players unless necessary.

### When to Use Group vs Individual

If everyone is doing the same thing → group node. If someone is doing something categorically different (a dancer performing in front of a band) → separate individual node for that person.

**The test:** Does the clip focus on specific people with distinct roles? If yes → individual nodes. If they're all doing the same thing and no one stands out → `crowd`, even with just 3 people. Three people chatting equally = `crowd`. Three people in an interview (one asking, one answering, one watching) = three individual nodes because they have distinct roles.

### Special Rules

- **man / woman / boy / girl / baby:** age-based split. Roughly: baby (0-2), boy/girl (2-12), man/woman (13+). Use judgment at boundaries. Use gender-neutral `baby` only for infants/toddlers where gender isn't visually identifiable.
- **hand** = only when the person is NOT visible. If the person is visible, the person is the source of edges like `holding`, not the hand.
- **microphone** = only when held by someone. Mic on a stand → skip.
- **Instruments** = one node per instrument type. Two guitars in the band → one guitar node. Acoustic guitar and bass are both `guitar`. Use attribute for sub-type (see Common Substitutions).

### Common Substitutions

| Item | Use | Example attribute |
|------|-----|-----------|
| Banjo, ukulele, bass guitar | guitar | "banjo", "bass guitar" |
| Cello, double bass, viola | violin | "cello", "double bass" |
| Saxophone | trumpet | "saxophone" |
| Electronic keyboard | piano | "electronic keyboard" |
| Basketball, soccer ball | ball | "basketball" |
| Funnel | cup | "funnel" |
| Wheelchair | chair | "motorized wheelchair" |
| Power cord, cable, leash | rope | "power cord", "leash" |
| Lake, river, pond | water | "lake" |
| Wok, frying pan, skillet | pan | "wok" |
| Bucket, pot, kettle | bowl | "cooking pot", "bucket" |
| Microwave, fridge | box | "microwave" |
| Truck | car | "pickup truck" |
| Skateboard, surfboard, sled, snowboard, ski | board | "skateboard", "snow sled" |
| Hiking pole, walking stick, shovel, umbrella | rod | "hiking pole", "shovel" |
| Gavel, mallet, sledgehammer | hammer | "gavel", "mallet" |
| Golf club, hockey stick, cricket bat | bat | "golf club" |
| Balloon, sculpture, toy figure | toy | "balloon", "horse sculpture" |
| TV, laptop, monitor | screen | "tv", "laptop" |
| Train, subway | bus | "subway train" |
| Drone, ATV | motorcycle | "drone", "atv" |
| Cash bills | paper | "cash bills" |
| Vinyl record, DVD | card | "vinyl record" |
| Cigarette, vape, pipe | bottle | "vape pen", "glass pipe" |

For objects not in the vocab and not in this table: substitute into the closest existing class with an attribute, or skip if not central.

---

## 2. Audio Nodes

Create nodes for the **obvious sounds** that capture the main idea. Be consistent with V2A edges.

### Classes

| Category | Classes |
|----------|---------|
| Human Voice | male_speech, female_speech, laughter, crying, male_singing, female_singing, child_singing, panting, coughing |
| Music — Visible | guitar_sound, piano_sound, drum_sound, violin_sound, trumpet_sound, flute_sound, accordion_sound |
| Background | background_music, background_sound, sound_effect |
| Crowd / Group | crowd_sound, applause |
| Animal | bark, meow, chirp |
| Mechanical | engine, footstep, phone_ring, whistle, gunshot, splash, door_sound, crashing_sound |
| Environmental | wind, rain |
| Contact | crinkling, scraping |

### Key Rules

- **male_speech / female_speech** covers ALL linguistic vocal output regardless of volume — calm conversation, raised voice, shouting, yelling, whispering. Put volume/tone in attribute. Does NOT cover panting, coughing, crying, laughing, or singing — those have their own classes.
- **Visual → audio mapping:** `man`/`boy` → `male_speech`/`male_singing`. `woman`/`girl` → `female_speech`/`female_singing`. `baby` → `crying`/`laughter` for typical baby sounds; `child_singing` for cooing or melodic vocalization.
- **child_singing** = baby/toddler singing, cooing, or mixed-gender group of young children singing together. Otherwise use gendered singing classes.
- **panting** = heavy breathing, grunting, sighing, moaning, or effort sounds from a human. Gender-neutral. Use `panting` V2A edge.
- **coughing** = cough or sneeze sounds. Gender-neutral. Use `coughing` V2A edge.
- **Visible instrument** → `[instrument]_sound`. V2A `playing` edge from the HUMAN, not the instrument. If the instrument is visible at any point in the clip, use `[instrument]_sound` even during camera cuts away.
- **Can't see the instrument at any point** → `background_music`. No matter how obvious the music sounds. Always isolated.
- **background_music** = always isolated. No edges. No exceptions.
- **background_sound** = unidentifiable ambient noise. Always isolated. If you can identify it → use specific class.
- **sound_effect** = post-production audio (swoosh, ding, dramatic sting). Always isolated.
- **crowd_sound** = all crowd vocalizations: cheering, singing along, chanting, murmuring, laughter. Use `crowd_sound` for everything. Only use `applause` when it's clearly distinct clapping-as-reaction.
- **door_sound** = knocking, clicking, slamming. Specify in attribute.
- **Separate speech nodes** for each distinguishable speaker. Shared sounds (laughter, guitar_sound) → one node.
- **Band singing in unison** → one `crowd_sound` node with attribute "band singing chorus".
- **Off-screen narrator** → isolated speech audio node, no V2A edge, no visual node for the speaker.
- **Orphan audio nodes are acceptable.** Audio that has no clear human/object source in the visible frame (off-screen voices, music with hidden source, ambient sounds, sounds from things outside the visual annotation set) should be included as an audio node without any V2A edge. A clip with audio nodes but zero V2A edges is valid.

---

## 3. Edges

All edges are **human-centered**: edges flow from the human/group node outward to objects and sounds. The human is always the source. (Exception: autonomous objects like phones ringing or car engines revving.)

**Shared edges (V2V + V2A):** the following edges work as both V2V and V2A — same action, different target type. Target node type determines which: visual target → V2V, audio target → V2A.

`playing`, `hitting`, `kicking`, `singing`, `crying`, `coughing`, `panting`, `laughing`

For example, `man --playing--> guitar` is V2V (visual target), while `man --playing--> guitar_sound` is V2A (audio target). Both are valid uses of the same edge.

### V2V Edges

**Rule: Most specific edge wins. Action edges beat spatial edges. Don't stack redundant edges.**

**Action Edges (use first):**

| Edge | Notes |
|------|-------|
| eating | Subsumes holding. eating = eating from. |
| drinking_from | Subsumes holding |
| filming | Camera must be VISIBLE in hand. Never for crowd. |
| aiming_at | Weapon at target |
| fishing | Subsumes holding (rod) |
| riding | On top of and controlling: horse, motorcycle, bicycle. NOT for cars → use `inside`. |
| swimming_in | Subsumes inside (water) |
| hitting | Non-musical percussive impact on something not designed for sound. Never for instruments → use `playing`. Never for mechanical closure → use `clicking`/`slamming`. |
| kicking | Foot-based contact |
| throwing | Moment of release |
| playing_with [SYM] | Mutual play (person + dog, kids together, child + water, person + ball). Symmetric. V2V only. |
| playing | Person → instrument. Also covers repetitive grip-based motion with any object if motion is the focus. |
| pointing_at | Person points at target |
| wearing | Person wears item |
| holding | Stationary or moving grip. Subsumes touching. Covers repetitive grip-based motion (lifting, pushing, pulling) — specifics in attribute. |
| sitting_on | Subsumes on |
| standing_on | Subsumes on |
| lying_on | Subsumes on |
| watching | Sustained attention: person→screen, small-group→performer, person→animal. NEVER person→person in conversation. NEVER crowd→band (use in_front_of instead). |
| walking_toward | Moving toward target |
| walking_through | Traversing a boundary (door, gate) |
| touching | Generic contact. Last resort. |

**Spatial Edges (use when no action edge applies):**

| Edge | Notes |
|------|-------|
| inside | Contained → container. Use for all people in cars. |
| on | Small on large, contact. Subsumed by sitting_on/standing_on/lying_on. |
| above | Non-contact vertical, higher entity as source |
| surrounded_by | Human/entity is enclosed on multiple sides. Human as source. Use ONLY when encloser is physically around (crowd-surfing, mosh pit, street performer mobbed on all sides). |
| in_front_of | Source closer to camera. Human always as source when humans are involved. |
| next_to [SYM] | Side by side. One edge, lower id as source. |

**in_front_of rules:**
- Human → object: human as source. "Man in_front_of car."
- Human → human, same direction: front person as source.
- Human → human, facing each other: use `next_to`. NEVER in_front_of.
- Band/performer → crowd (standard concert, crowd facing band): band `in_front_of` crowd. This replaces crowd→watching→band.
- Performer surrounded on multiple sides: use `surrounded_by` instead.

**Deleted edges (NEVER use):** behind, under, beneath, below, looking_at, vocalizing, striking, strumming, surrounding, shouting, shooting

### V2A Edges

**Human-centered:** V2A always goes from the HUMAN → audio node. Not from the instrument.

| Edge | Notes |
|------|-------|
| singing | Melodic vocalization. Does not subsume speaking. |
| crying | Emotional |
| laughing | Emotional |
| panting | Heavy breathing, grunting, sighing, moaning, effort sounds. |
| coughing | Cough or sneeze sounds. |
| speaking | All linguistic vocal output — calm, raised voice, shouting, yelling, whispering. Volume/tone in attribute. |
| playing | ALL instruments. One verb for all music. |
| hitting | Non-musical percussive impact. |
| cheering | Crowd cheering |
| scraping | Hard surface on hard surface |
| crinkling | Paper/plastic/cardboard |
| kicking | Kick produces sound |
| clapping | Hands clapping |
| barking | Dog |
| meowing | Cat |
| chirping | Bird |

**Exception:** Objects acting on their own (car engine, door click, phone ring) → V2A from the object.

**Extended V2A edges** (canonical):

| Edge | For |
|------|-----|
| splashing | Person/object in water |
| pouring | Pouring liquid or dry items |
| revving | Vehicle engine revving |
| slamming | Door/object slamming shut |
| clicking | Mechanical click (door latch, button) |
| stepping | Walking/running footsteps. Points at `footstep` audio node. |
| ringing | Phone ringing. Points at `phone_ring` audio node. Object-as-source (phone is the visual node). |

### A2A Edges

| Edge | Type | When |
|------|------|------|
| in_dialogue_with [SYM] | Symmetric | Back-and-forth linguistic turn-taking with content. One edge, lower id as source. |
| responds_to | Directed | One-way, NON-CONVERSATIONAL. Crowd reacting to speaker, cheer after an announcement. Never for actual conversations. |

No edge if sounds merely co-occur. All background audio (background_music, background_sound, sound_effect) is always isolated.

---

## 4. Common Patterns

### Concert / Band Performance (standard — crowd in front of band)
```
v1 (band) --in_front_of--> v2 (crowd)         [performer is focus]
v1 (band) --playing--> v3 (guitar)             [V2V]
v1 (band) --playing--> v4 (drum)               [V2V]
v1 (band) --playing--> a1 (guitar_sound)       [V2A]
v1 (band) --playing--> a2 (drum_sound)         [V2A]
v1 (band) --singing--> a3 (male_singing)       [V2A]
a4 (crowd_sound, "crowd cheering")             [isolated]
```

### Performer Surrounded by Crowd (street performer, mosh pit)
```
v1 (man) --surrounded_by--> v2 (crowd)
v1 --playing--> v3 (guitar)
v1 --playing--> a1 (guitar_sound)
a2 (crowd_sound, "crowd cheering")
```

### Car Conversation
```
v1 (man, "driver") --inside--> v3 (car)
v2 (woman, "passenger") --inside--> v3 (car)
v1 --next_to--> v2
v1 --speaking--> a1 (male_speech)
v2 --speaking--> a2 (female_speech)
a1 --in_dialogue_with--> a2
```

### Interview / Conversation
```
v1 (man, "interviewer") --holding--> v3 (microphone)
v1 --next_to--> v2 (woman, "interviewee")
v1 --speaking--> a1 (male_speech)
v2 --speaking--> a2 (female_speech)
a1 --in_dialogue_with--> a2
```
People facing each other → `next_to`. Never in_front_of.

### Sports
```
v1 (team) --playing_with--> v2 (ball, "basketball")
v1 --standing_on--> v3 (court)
a1 (crowd_sound, "crowd cheering")
```

### Unboxing
```
v1 (hand) --holding--> v2 (box)
v1 --crinkling--> a1 (crinkling, "cardboard tearing")
a2 (male_speech, "man narrating off-screen")     [isolated]
```

### Off-screen Narrator
```
v1 (dog) --barking--> a1 (bark)
a2 (male_speech, "narrator voice-over")           [isolated]
```

### Effort / Workout
```
v1 (man) --holding--> v2 (dumbbell)
v1 --panting--> a1 (panting, "heavy exhale while lifting")
a2 (background_music, "gym music")                [isolated]
```

### Public Speaking / Rally
```
v1 (man) --in_front_of--> v2 (crowd)
v1 --speaking--> a1 (male_speech, "angry shouting")
a2 (crowd_sound, "crowd cheering")
a2 --responds_to--> a1
```

---

## 5. Symmetric Edges

Symmetric edges: **next_to, playing_with, in_dialogue_with**

NOT symmetric: surrounded_by, responds_to, watching

**Annotation convention:** store once with lower id as source. E.g., `v1 next_to v2`, not `v2 next_to v1`.

**Evaluation:** symmetric edges are order-agnostic. `(v1, next_to, v2)` and `(v2, next_to, v1)` are equivalent and must match either way during evaluation. Implementations should normalize symmetric edges before comparison:

```python
SYMMETRIC = {"next_to", "playing_with", "in_dialogue_with"}

def normalize_edge(e):
    src, name, tgt = e["source"], e["edge"], e["target"]
    if name in SYMMETRIC:
        src, tgt = sorted([src, tgt])
    return (src, name, tgt)
```

This ensures predicted graphs match ground truth regardless of source/target order on symmetric edges.

---

## 6. Vocab Summary

| Category | Count |
|---|---|
| Visual classes | 74 |
| Audio classes | 36 |
| V2V edges | 28 (22 action + 6 spatial) |
| V2A edges | 23 (16 core + 7 extended) |
| A2A edges | 2 |
| Shared V2V/V2A edges | 8 (playing, hitting, kicking, singing, crying, coughing, panting, laughing) |
| Symmetric edges | 3 (next_to, playing_with, in_dialogue_with) |

---

## Changelog

**v2.29 (2026-04-16) — hard caps on node count**
- Set explicit hard limits: minimum 2 nodes, maximum 10 nodes per clip. Typical target remains 4–6.
- Re-annotated one clip exceeding 10 nodes to comply with the cap.

**v2.28 (2026-04-16) — annotation guidance update**
- Updated node count target from "2-4, rarely 5" to "4-6 nodes (2-3 visual + 2-3 audio), stop at 8" to reflect actual annotation behavior across 1024 clips (median 5, avg 5.18).
- Added explicit rule that **orphan audio nodes are acceptable**. Audio without a clear visual source (off-screen voices, hidden music sources, ambient sound) is annotated as an audio node with no V2A edge. Clips with audio nodes but zero V2A edges are valid.
- Added intro note clarifying that clips are pre-segmented short video segments (median ~14s) cut from longer source videos.

**v2.27 (2026-04-16) — benchmark publication prep**
- Documented shared edges (V2V + V2A): `playing`, `hitting`, `kicking`, `singing`, `crying`, `coughing`, `panting`, `laughing` work as both, distinguished by target node type. This was implicit before; now explicit.
- Added evaluation rule for symmetric edges: order-agnostic during evaluation. Predicted and ground-truth graphs match regardless of source/target order for `next_to`, `playing_with`, `in_dialogue_with`. Includes Python normalization snippet.
- Updated Vocab Summary with shared edges count and accurate edge counts.

**v2.26 (2026-04-16) — final cleanup**
- Removed unused visual classes: `bear`, `truck`, `wall`, `sky`, `beach`, `speaker`.
- Removed unused audio classes: `bell`, `explosion`, `siren`.
- Removed unused edges: `dancing`, `dancing_with`, `groaning`, `squawking`.
- Updated Common Substitutions table to reflect final vocab: removed entries for classes that became official (fish, sword, axe, pan), added entries for common substitutions made during annotation (board, motorcycle, screen, bus, hammer, bat, toy, paper, card, bottle).
- Updated `Deleted edges` list to include `shouting` and `shooting`.
- Added Vocab Summary section.
- Updated DON'T annotate guidance to explicitly note clothing/footwear/worn items.

**v2.0 – v2.25**
Progressive refinement over annotation of 1024 clips. Major changes:
- v2.1: collapsed shouting into speech; added panting class; added dumbbell.
- v2.2–v2.5: shoe class, fishing_rod→rod, boy/girl visual classes, child_speech scoping.
- v2.6–v2.11: removed table, sofa, keyboard, floor, headphone, skateboard_sound (unused/redundant).
- v2.12–v2.15: added flute, pen; removed directed_at; consolidated child_speech/child_singing rules.
- v2.16–v2.17: added stepping V2A edge; added coughing class and edge.
- v2.18–v2.20: added trumpet, violin, fish, sword, axe, pan (formalized substitutions).
- v2.21–v2.22: added ringing V2A edge; added hammer.
- v2.23–v2.25: merged skateboard+surfboard→board; removed stage; added accordion/accordion_sound.
