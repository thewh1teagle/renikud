from languages.hebrew import HEBREW

LANG_PACKS = {
    "hebrew": HEBREW,
}


def get_lang_pack(name: str):
    if name not in LANG_PACKS:
        raise ValueError(f"Unknown language pack: {name!r}. Available: {list(LANG_PACKS)}")
    return LANG_PACKS[name]
