import math
from collections import defaultdict


def _log_add(a: float, b: float) -> float:
    if a == float("-inf"):
        return b
    if b == float("-inf"):
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def valid_chunk(chunk: str, allowed_starts: list[str]) -> bool:
    if not chunk:
        return "" in allowed_starts
    for atom in allowed_starts:
        if atom and chunk.startswith(atom):
            return True
    return False


def forward_backward(
    letters: list[str],
    tokens: list[str],
    log_probs: dict,
    allowed: dict[str, list[str]],
    max_n: int,
) -> dict[tuple[str, str], float] | None:
    n = len(letters)
    m = len(tokens)
    NEG_INF = float("-inf")

    fwd = [[NEG_INF] * (m + 1) for _ in range(n + 1)]
    fwd[0][0] = 0.0

    for i in range(1, n + 1):
        letter = letters[i - 1]
        allowed_starts = allowed.get(letter, [""])
        for j_prev in range(m + 1):
            if fwd[i - 1][j_prev] == NEG_INF:
                continue
            for length in range(0, max_n + 1):
                j_new = j_prev + length
                if j_new > m:
                    break
                chunk = "".join(tokens[j_prev:j_new])
                if not valid_chunk(chunk, allowed_starts):
                    continue
                lp = log_probs.get((letter, chunk), NEG_INF)
                if lp == NEG_INF:
                    continue
                fwd[i][j_new] = _log_add(fwd[i][j_new], fwd[i - 1][j_prev] + lp)

    if fwd[n][m] == NEG_INF:
        return None

    bwd = [[NEG_INF] * (m + 1) for _ in range(n + 1)]
    bwd[n][m] = 0.0

    for i in range(n - 1, -1, -1):
        letter = letters[i]
        allowed_starts = allowed.get(letter, [""])
        for j_next in range(m + 1):
            if bwd[i + 1][j_next] == NEG_INF:
                continue
            for length in range(0, max_n + 1):
                j_prev = j_next - length
                if j_prev < 0:
                    break
                chunk = "".join(tokens[j_prev:j_next])
                if not valid_chunk(chunk, allowed_starts):
                    continue
                lp = log_probs.get((letter, chunk), NEG_INF)
                if lp == NEG_INF:
                    continue
                bwd[i][j_prev] = _log_add(bwd[i][j_prev], bwd[i + 1][j_next] + lp)

    total = fwd[n][m]
    counts: dict[tuple[str, str], float] = defaultdict(float)

    for i in range(n):
        letter = letters[i]
        allowed_starts = allowed.get(letter, [""])
        for j_prev in range(m + 1):
            if fwd[i][j_prev] == NEG_INF:
                continue
            for length in range(0, max_n + 1):
                j_new = j_prev + length
                if j_new > m:
                    break
                chunk = "".join(tokens[j_prev:j_new])
                if not valid_chunk(chunk, allowed_starts):
                    continue
                lp = log_probs.get((letter, chunk), NEG_INF)
                if lp == NEG_INF:
                    continue
                if bwd[i + 1][j_new] == NEG_INF:
                    continue
                counts[(letter, chunk)] += math.exp(fwd[i][j_prev] + lp + bwd[i + 1][j_new] - total)

    return counts


def viterbi(
    letters: list[str],
    tokens: list[str],
    log_probs: dict,
    allowed: dict[str, list[str]],
    max_n: int,
) -> list[tuple[str, str]] | None:
    n = len(letters)
    m = len(tokens)
    NEG_INF = float("-inf")

    dp = [[NEG_INF] * (m + 1) for _ in range(n + 1)]
    back = [[-1] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    for i in range(1, n + 1):
        letter = letters[i - 1]
        allowed_starts = allowed.get(letter, [""])
        for j_prev in range(m + 1):
            if dp[i - 1][j_prev] == NEG_INF:
                continue
            for length in range(0, max_n + 1):
                j_new = j_prev + length
                if j_new > m:
                    break
                chunk = "".join(tokens[j_prev:j_new])
                if not valid_chunk(chunk, allowed_starts):
                    continue
                lp = log_probs.get((letter, chunk), NEG_INF)
                if lp == NEG_INF:
                    continue
                val = dp[i - 1][j_prev] + lp
                if val > dp[i][j_new]:
                    dp[i][j_new] = val
                    back[i][j_new] = j_prev

    if dp[n][m] == NEG_INF:
        return None

    result = []
    j = m
    for i in range(n, 0, -1):
        j_prev = back[i][j]
        result.append((letters[i - 1], "".join(tokens[j_prev:j])))
        j = j_prev
    result.reverse()
    return result
