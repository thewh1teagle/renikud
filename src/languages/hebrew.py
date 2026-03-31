from lang_pack import LangPack

HEBREW = LangPack(
    name="hebrew",
    input_chars=frozenset(
        "אבגדהוזחטיכךלמםנןסעפףצץקרשת"
    ),
    output_tokens=(
        # 0 = null (silent, no output)
        "∅",
        # plain consonants
        "b", "v", "d", "h", "z", "χ", "t", "j", "k", "l",
        "m", "n", "s", "f", "p", "ts", "tʃ", "w", "ʔ", "ɡ", "ʁ", "ʃ", "ʒ", "dʒ",
        # plain vowels (unstressed)
        "a", "e", "i", "o", "u",
        # stressed vowels (stress mark fused into token)
        "ˈa", "ˈe", "ˈi", "ˈo", "ˈu",
    ),
    extra_chars=frozenset("\u05BE\u05F3\u05F4"),  # maqaf, geresh, gershayim
    strip_accents=True,
)
