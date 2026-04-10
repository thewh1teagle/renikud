import functools

HEBREW_LETTER_CONSONANTS: dict[str, tuple[str, ...]] = {
    "א": ("ʔ", ""),
    "ב": ("b", "v"),
    "ג": ("ɡ", "dʒ"), 
    "ד": ("d",),
    "ה": ("h", ""),
    "ו": ("v", "w", ""),
    "ז": ("z", "ʒ"),
    "ח": ("χ",),
    "ט": ("t",),
    "י": ("j", ""),
    "כ": ("k", "χ"),
    "ך": ("k", "χ"),
    "ל": ("l", ''),
    "מ": ("m",),
    "ם": ("m",),
    "נ": ("n",),
    "ן": ("n",),
    "ס": ("s",),
    "ע": ("ʔ", ""),
    "פ": ("p", "f"),
    "ף": ("p", "f"),
    "צ": ("ts", "tʃ"),
    "ץ": ("ts", "tʃ"),
    "ק": ("k",),
    "ר": ("ʁ"),
    "ש": ("ʃ", "s", ''),
    "ת": ("t",),
}

VOWELS = ("a", "e", "i", "o", "u")
STRESS = "ˈ"
SPECIAL_CHARS = ("'", '"')

def align_word(heb_word: str, ipa_word: str) -> list[tuple[str, str]] | None:
    @functools.lru_cache(maxsize=None)
    def search(h_idx: int, i_idx: int) -> tuple | None:
        if h_idx == len(heb_word) and i_idx == len(ipa_word):
            return ()
        if h_idx == len(heb_word):
            return None

        char = heb_word[h_idx]

        if char in SPECIAL_CHARS:
            res = search(h_idx + 1, i_idx)
            if res is not None:
                return ((char, ""),) + res
            return None

        if char not in HEBREW_LETTER_CONSONANTS:
            return None

        allowed_cons = HEBREW_LETTER_CONSONANTS[char]
        rest_ipa = ipa_word[i_idx:]

        for cons in allowed_cons:
            if cons and not rest_ipa.startswith(cons):
                continue
            
            c_len = len(cons)
            for has_stress in (True, False):
                s_len = 0
                if has_stress:
                    if (c_len < len(rest_ipa)) and rest_ipa[c_len] == STRESS:
                        s_len = 1
                    else: continue
                
                for v in (*VOWELS, ""):
                    v_start = c_len + s_len
                    if v and rest_ipa[v_start:].startswith(v):
                        v_len = len(v)
                    elif not v:
                        v_len = 0
                    else: continue
                    
                    total_step = v_start + v_len
                    res = search(h_idx + 1, i_idx + total_step)
                    if res is not None:
                        return ((char, rest_ipa[:total_step]),) + res

        # Furtive Patah (למשל סוף המילה שיח)
        for cons in allowed_cons:
            if not cons: continue
            for has_stress in (True, False):
                prefix = STRESS + "a" if has_stress else "a"
                if rest_ipa.startswith(prefix + cons):
                    step = len(prefix) + len(cons)
                    res = search(h_idx + 1, i_idx + step)
                    if res is not None:
                        return ((char, rest_ipa[:step]),) + res
        
        return None

    result = search(0, 0)
    return list(result) if result is not None else None
