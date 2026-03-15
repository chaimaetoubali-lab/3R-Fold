import RNA


def dotbracket_to_pairs(structure):
    stack = []
    pairs = []

    for i, ch in enumerate(structure):
        if ch == "(":
            stack.append(i)
        elif ch == ")" and stack:
            j = stack.pop()
            pairs.append((j, i))

    return pairs


def build_pair_frequency(query_length, neighbor_structures, weights=None):
    if not neighbor_structures:
        return {}

    if weights is None or len(weights) != len(neighbor_structures):
        weights = [1.0] * len(neighbor_structures)

    pair_scores = {}
    total_weight = sum(weights) if sum(weights) > 0 else 1.0

    for struct, w in zip(neighbor_structures, weights):
        for i, j in dotbracket_to_pairs(struct):
            if i < query_length and j < query_length:
                pair_scores[(i, j)] = pair_scores.get((i, j), 0.0) + w

    for pair in pair_scores:
        pair_scores[pair] /= total_weight

    return pair_scores


def build_unpaired_scores(query_length, neighbor_structures, weights=None):
    if not neighbor_structures:
        return [0.0] * query_length

    if weights is None or len(weights) != len(neighbor_structures):
        weights = [1.0] * len(neighbor_structures)

    scores = [0.0] * query_length
    total_weight = sum(weights) if sum(weights) > 0 else 1.0

    for struct, w in zip(neighbor_structures, weights):
        L = min(len(struct), query_length)
        for i in range(L):
            if struct[i] == ".":
                scores[i] += w

    return [s / total_weight for s in scores]


def consensus_pairs(pair_scores, min_freq=0.5):
    sorted_pairs = sorted(pair_scores.items(), key=lambda x: x[1], reverse=True)

    selected = []
    used = set()

    for (i, j), score in sorted_pairs:
        if score < min_freq:
            continue
        if i in used or j in used:
            continue
        selected.append((i, j))
        used.add(i)
        used.add(j)

    return selected


def pairs_to_constraint(length, pairs):
    constraint = ["."] * length
    for i, j in pairs:
        if 0 <= i < length and 0 <= j < length and i < j:
            constraint[i] = "("
            constraint[j] = ")"
    return "".join(constraint)


def fold_with_neighbor_guidance(
    sequence,
    neighbor_structures,
    weights=None,
    min_pair_freq=0.5,
    unpaired_bonus=0.5
):
    n = len(sequence)

    if not neighbor_structures:
        fc = RNA.fold_compound(sequence)
        structure, _ = fc.mfe()
        return structure

    pair_scores = build_pair_frequency(n, neighbor_structures, weights)
    unpaired_scores = build_unpaired_scores(n, neighbor_structures, weights)

    selected_pairs = consensus_pairs(pair_scores, min_freq=min_pair_freq)
    constraint = pairs_to_constraint(n, selected_pairs)

    fc = RNA.fold_compound(sequence)

    # Contraintes structurelles consensus
    if any(ch in "()" for ch in constraint):
        fc.constraints_add(constraint, RNA.CONSTRAINT_DB_DEFAULT)

    # Bonus léger pour garder non appariées les positions souvent "."
    for i, score in enumerate(unpaired_scores, start=1):
        if score > 0:
            fc.sc_add_up(i, -unpaired_bonus * score)

    structure, _ = fc.mfe()
    return structure