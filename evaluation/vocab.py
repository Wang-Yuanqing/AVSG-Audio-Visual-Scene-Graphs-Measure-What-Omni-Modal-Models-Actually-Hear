"""AVSG v2.29 closed vocabulary (visual classes, audio classes, edges).

Used by the benchmark parser to flag out-of-vocab predictions and by the prompt
builder to give models the allowed labels.
"""

VISUAL_CLASSES = {
    # People
    "man", "woman", "boy", "girl", "baby", "crowd", "band", "team",
    # Animals
    "dog", "cat", "horse", "bird", "cow",
    # Objects
    "phone", "camera", "microphone", "guitar", "ball", "board", "book", "cup",
    "bottle", "bowl", "food", "bag", "box", "paper", "drum", "gun", "rod", "rope",
    "bat", "toy", "plate", "spoon", "knife", "brush", "flag", "piano", "card", "can",
    "scissors", "dumbbell", "shoe", "flute", "pen", "trumpet", "violin", "fish",
    "sword", "axe", "pan", "hammer", "accordion",
    # Vehicles
    "car", "motorcycle", "bicycle", "boat", "bus",
    # Furniture
    "chair", "bed", "desk", "door", "window",
    # Scene Elements
    "screen", "sign", "water", "grass", "tree", "field", "court", "fire", "building",
    # Body Parts
    "hand",
}

AUDIO_CLASSES = {
    "male_speech", "female_speech", "laughter", "crying", "male_singing",
    "female_singing", "child_singing", "panting", "coughing",
    "guitar_sound", "piano_sound", "drum_sound", "violin_sound", "trumpet_sound",
    "flute_sound", "accordion_sound",
    "background_music", "background_sound", "sound_effect",
    "crowd_sound", "applause",
    "bark", "meow", "chirp",
    "engine", "footstep", "phone_ring", "whistle", "gunshot", "splash",
    "door_sound", "crashing_sound",
    "wind", "rain",
    "crinkling", "scraping",
}

V2V_EDGES = {
    "eating", "drinking_from", "filming", "aiming_at", "fishing", "riding",
    "swimming_in", "hitting", "kicking", "throwing", "playing_with", "playing",
    "pointing_at", "wearing", "holding", "sitting_on", "standing_on", "lying_on",
    "watching", "walking_toward", "walking_through", "touching",
    "inside", "on", "above", "surrounded_by", "in_front_of", "next_to",
}

V2A_EDGES = {
    "singing", "crying", "laughing", "panting", "coughing", "speaking", "playing",
    "hitting", "cheering", "scraping", "crinkling", "kicking", "clapping",
    "barking", "meowing", "chirping",
    "splashing", "pouring", "revving", "slamming", "clicking", "stepping", "ringing",
}

A2A_EDGES = {"in_dialogue_with", "responds_to"}

ALL_EDGES = V2V_EDGES | V2A_EDGES | A2A_EDGES
ALL_CLASSES = VISUAL_CLASSES | AUDIO_CLASSES
