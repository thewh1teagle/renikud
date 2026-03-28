import yaml


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def tokenize_ipa(ipa: str, atoms: list[str]) -> list[str]:
    """Split an IPA string into atoms. Digraphs take priority (listed first in config)."""
    tokens = []
    i = 0
    while i < len(ipa):
        matched = False
        for atom in atoms:
            if ipa[i:].startswith(atom):
                tokens.append(atom)
                i += len(atom)
                matched = True
                break
        if not matched:
            tokens.append(ipa[i])
            i += 1
    return tokens


def parse_sentence(
    text: str, ipa: str, atoms: list[str]
) -> tuple[list[list[str]], list[list[str]]] | None:
    words = text.split(" ")
    ipa_words = ipa.split(" ")
    if len(words) != len(ipa_words):
        return None
    letter_seqs, token_seqs = [], []
    for w, iw in zip(words, ipa_words):
        if not w:
            continue
        letter_seqs.append(list(w))
        token_seqs.append(tokenize_ipa(iw, atoms))
    return letter_seqs, token_seqs


def align_sentence(
    text: str,
    ipa: str,
    atoms: list[str],
    log_probs: dict,
    allowed: dict[str, list[str]],
    max_n: int,
) -> list[tuple[str, str]] | None:
    from dp import viterbi
    parsed = parse_sentence(text, ipa, atoms)
    if parsed is None:
        return None
    letter_seqs, token_seqs = parsed
    result = []
    for idx, (ls, ts) in enumerate(zip(letter_seqs, token_seqs)):
        aligned = viterbi(ls, ts, log_probs, allowed, max_n)
        if aligned is None:
            return None
        result.extend(aligned)
        if idx < len(letter_seqs) - 1:
            result.append((" ", " "))
    return result
