

# Lista de frases explícitas
SUICIDAL_PHRASES = [
    "i want to die",
    "i want to kill myself",
    "i'm going to kill myself",
    "i'm thinking about suicide",
    "i'm considering suicide",
    "i plan to take my life",
    "i feel like ending it all",
    "i wish i were dead",
    "life is not worth living",
    "i can't go on anymore",
    "i don’t want to live",
    "there’s no point in living",
    "i want to end it",
    "kill myself",
    "commit suicide",
    "thinking of dying",
    "no reason to live",
    "die by suicide",
    "end my life"
]

def contains_suicidal_phrase(text):

    text = text.lower()
    for phrase in SUICIDAL_PHRASES:
        if phrase in text:
            return True
    return False
