"""Short, source-blind relation guidance derived from rubric v2.29.

These rules concern the displayed relation and endpoints, not whole-graph
completeness, node limits, or the origin of a candidate.
"""
import re


BACKGROUND = {"background_music", "background_sound", "sound_effect"}

V2V = {
    "eating": "The shown person is eating the target food or eating from the target container; merely holding it or bringing it near the mouth is insufficient.",
    "drinking_from": "The shown person drinks from the target container; this is more specific than holding it, and merely holding the container is insufficient.",
    "filming": "The shown person films using a camera visible in their hand, with the intended target correctly identified; a crowd cannot be the starting endpoint.",
    "aiming_at": "The shown person aims a weapon at the described target, so the arrow goes from the person doing the aiming toward that target.",
    "fishing": "The shown person uses the described target while fishing; this relation takes priority over holding the fishing rod, but merely holding a rod is insufficient.",
    "riding": "The shown person is on top of and controlling the target horse, motorcycle or bicycle; use inside rather than riding for a person in a car.",
    "swimming_in": "The shown person is swimming in the target body of water; swimming takes priority over merely being inside the water.",
    "hitting": "The shown person strikes the target with a non-musical impact; playing an instrument or closing a mechanism uses a different relationship.",
    "kicking": "The shown person makes foot-based contact with the target, with the arrow going from the person performing the kick toward what is kicked.",
    "throwing": "The shown person releases the target object in a throwing motion; merely holding, carrying or preparing to throw it is insufficient.",
    "playing_with": "The described endpoints participate in play together, such as a person playing with a dog or ball; either direction means the same relationship.",
    "playing": "The shown person or group plays the target instrument, or repeatedly moves another target object while gripping it when that motion is the focus.",
    "pointing_at": "The shown person visibly points toward the described target, with the arrow going from the person doing the pointing toward that target.",
    "wearing": "The shown person wears the target item; ordinary clothing, shoes, hats and jewelry belong in the person's description rather than separate endpoints.",
    "holding": "The shown person grips the target, including lifting, pushing or pulling it; eating, drinking or fishing should use their more specific relationship when applicable.",
    "sitting_on": "The shown person sits on the target supporting surface, with the arrow going from the seated person toward that surface rather than the reverse.",
    "standing_on": "The shown person stands on the target supporting surface, with the arrow going from the standing person toward that surface rather than the reverse.",
    "lying_on": "The shown person lies on the target supporting surface, with the arrow going from the person lying down toward that surface rather than the reverse.",
    "watching": "The starting person or small group pays sustained attention to the target; this label excludes people conversing with each other and a crowd watching a band.",
    "walking_toward": "The shown person walks in the direction of the described target, with the arrow going from the moving person toward that target.",
    "walking_through": "The shown person walks through the target boundary, such as a door or gate; simply standing beside it or moving toward it is insufficient.",
    "touching": "The shown person physically contacts the target; use this general relationship only when a more specific action such as holding or eating does not apply.",
    "inside": "The starting endpoint is contained within the target, including a person inside a car; use swimming instead when the person is swimming in water.",
    "on": "The starting endpoint rests on and touches the target; sitting, standing or lying on a surface should use those more specific relationships.",
    "above": "The starting endpoint is higher than and not touching the other endpoint, with a person as the starting endpoint whenever a person is involved.",
    "surrounded_by": "The starting person or entity is physically enclosed on multiple sides by the target group or surroundings; merely being in front of a crowd is insufficient.",
    "in_front_of": "The starting endpoint is nearer the camera, or is a performer facing an audience; a person must start the arrow when involved, and people facing each other use next_to.",
    "next_to": "The described endpoints are next to one another, including people facing each other; either direction is equivalent, and a more specific visible action takes priority.",
}

V2A = {
    "speaking": "The shown person or group must be identifiable as producing the described speech, including shouting or whispering; an unrelated or off-screen voice does not establish this link.",
    "singing": "The shown person or group must produce the described audible melodic vocalization; ordinary speech alone does not establish singing, and unrelated background vocals do not establish the link.",
    "crying": "The shown person must be identifiable as producing the described audible crying; a crying expression without a corresponding sound does not establish this link.",
    "laughing": "The shown person or group must produce the described audible laughter; several people may share one laughter endpoint, but an unrelated laugh does not establish the link.",
    "panting": "The shown person must produce the described audible heavy breathing, grunting, sighing, moaning or effort sound; visible exertion alone is insufficient.",
    "coughing": "The shown person must produce the described audible cough or sneeze; an unrelated cough or a visible movement without the sound is insufficient.",
    "playing": "A visible person or group must produce this audible instrument sound, and the instrument must appear at least once; an instrument itself cannot be the starting endpoint.",
    "hitting": "The shown person must produce the described audible non-musical impact by striking something; instrument playing and mechanical closing use different relationships.",
    "kicking": "The shown person must produce the described audible impact by kicking; seeing a kick without hearing its resulting sound does not establish this link.",
    "cheering": "The shown crowd must produce the described audible cheering or other crowd vocalization; unrelated crowd noise or merely seeing excited people is insufficient.",
    "scraping": "The shown person must produce the described scraping sound by moving a hard surface against another hard surface; visible motion without the corresponding sound is insufficient.",
    "crinkling": "The shown person must produce the described crinkling sound by handling paper, plastic or cardboard; merely seeing that material or handling it silently is insufficient.",
    "clapping": "The shown person or group must produce the described audible handclaps; seeing hands move or hearing unrelated applause does not establish this link.",
    "barking": "The shown dog must be identifiable as producing the described audible bark; merely seeing a dog or hearing another animal is insufficient.",
    "meowing": "The shown cat must be identifiable as producing the described audible meow; merely seeing a cat or hearing another animal is insufficient.",
    "chirping": "The shown bird must be identifiable as producing the described audible chirp; merely seeing a bird or hearing another animal is insufficient.",
    "splashing": "The shown person or object must produce the described audible splash while interacting with water; seeing water movement without its corresponding sound is insufficient.",
    "pouring": "The shown person must produce the described sound while pouring liquid or dry material; merely seeing the pouring motion without its sound is insufficient.",
    "revving": "The shown vehicle must be identifiable as producing the described audible engine sound; this relationship may start from the vehicle itself rather than a person.",
    "slamming": "The shown person must cause the described audible slam by closing something forcefully; an object may start the arrow only when it acts on its own.",
    "clicking": "The shown person must cause the described audible mechanical click; an object may start the arrow when it operates on its own, but an added sound effect cannot be linked.",
    "stepping": "The shown person must produce the described audible footsteps while walking or running; seeing steps without hearing the corresponding footstep sound is insufficient.",
    "ringing": "The shown phone must be identifiable as producing the described audible ring; this relationship starts from the phone itself rather than the person holding it.",
}

A2A = {
    "in_dialogue_with": "The two described voices must take turns in a back-and-forth conversation with linguistic content; either direction is equivalent, and merely overlapping or co-occurring sounds are insufficient.",
    "responds_to": "The sound at the start of the arrow must audibly react to the sound at its end, such as laughter responding to speech; conversational replies and mere co-occurrence do not qualify.",
}


def rule_for(task):
    """Return one plain-English sentence, without using provenance or answers."""
    source, target = task.get("source", {}), task.get("target", {})
    if source.get("class") in BACKGROUND or target.get("class") in BACKGROUND:
        return ("Background music, background sound and added sound effects must stay "
                "unconnected, so a relationship involving any of these labels is not "
                "allowed even when the sound is audible.")
    relation = str(task.get("relation", ""))
    family = str(task.get("family", "")).lower()
    if not family:
        family = ("v" if str(source.get("id", "")).startswith("v") else "a") + "2" + (
            "v" if str(target.get("id", "")).startswith("v") else "a")
    rules = {"v2v": V2V, "v2a": V2A, "a2a": A2A}.get(family, {})
    rule = rules.get(relation)
    if rule is None:
        return ("Judge whether the named relationship holds between the described endpoints "
                "in the shown direction; if its meaning or permitted use is unclear, "
                "choose Cannot tell.")
    if family == "v2v" and relation == "riding" and target.get("class") == "motorcycle" and re.search(
            r"\batv\b", str(target.get("attribute", "")), flags=re.IGNORECASE):
        return ("The shown person must sit on and control the described ATV, which is "
                "intentionally labeled motorcycle; riding does not describe someone "
                "sitting inside a car.")
    if family == "v2v" and source.get("class") == "hand":
        rule = rule[:-1] + "; a hand represents the person only when that person is off-screen."
    if any(re.search(r"\batv\b", str(node.get("attribute", "")), flags=re.IGNORECASE)
           for node in (source, target)):
        rule = rule[:-1] + "; an ATV uses the motorcycle label."
    return rule
