from Bio import pairwise2
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


def align_query_to_neighbor(query_seq, neighbor_seq):
    alignments = pairwise2.align.globalms(
        query_seq,
        neighbor_seq,
        2.0,   # match
        -1.0,  # mismatch
        -2.0,  # gap open
        -0.5,  # gap extend
        one_alignment_only=True
    )
    if not alignments:
        return None, None

    aln = alignments[0]
    return aln.seqA, aln.seqB


def build_alignment_maps(aligned_query, aligned_neighbor):
    query_to_aln = {}
    neighbor_to_aln = {}
    aln_to_query = {}
    aln_to_neighbor = {}

    q_idx = 0
    n_idx = 0

    for aln_idx, (q_ch, n_ch) in enumerate(zip(aligned_query, aligned_neighbor)):
        if q_ch != "-":
            query_to_aln[q_idx] = aln_idx
            aln_to_query[aln_idx] = q_idx
            q_idx += 1
        if n_ch != "-":
            neighbor_to_aln[n_idx] = aln_idx
            aln_to_neighbor[aln_idx] = n_idx
            n_idx += 1

    return query_to_aln, neighbor_to_aln, aln_to_query, aln_to_neighbor


def transfer_neighbor_pairs_to_query(query_seq, neighbor_seq, neighbor_structure):
    aligned_query, aligned_neighbor = align_query_to_neighbor(query_seq, neighbor_seq)
    if aligned_query is None:
        return []

    _, _, aln_to_query, aln_to_neighbor = build_alignment_maps(aligned_query, aligned_neighbor)

    transferred_pairs = []
    neighbor_pairs = dotbracket_to_pairs(neighbor_structure)

    neighbor_pos_to_aln = {}
    n_idx = 0
    for aln_idx, ch in enumerate(aligned_neighbor):
        if ch != "-":
            neighbor_pos_to_aln[n_idx] = aln_idx
            n_idx += 1

    for i, j in neighbor_pairs:
        if i not in neighbor_pos_to_aln or j not in neighbor_pos_to_aln:
            continue

        aln_i = neighbor_pos_to_aln[i]
        aln_j = neighbor_pos_to_aln[j]

        if aln_i in aln_to_query and aln_j in aln_to_query:
            qi = aln_to_query[aln_i]
            qj = aln_to_query[aln_j]
            if qi < qj and qj < len(query_seq):
                transferred_pairs.append((qi, qj))

    return transferred_pairs


def canonical_pair(a, b):
    pair = (a.upper().replace("T", "U"), b.upper().replace("T", "U"))
    return pair in {
        ("A", "U"), ("U", "A"),
        ("G", "C"), ("C", "G"),
        ("G", "U"), ("U", "G"),
    }


def accumulate_transferred_pair_scores(query_seq, neighbors, weights=None):
    if not neighbors:
        return {}

    if weights is None or len(weights) != len(neighbors):
        weights = [1.0] * len(neighbors)

    total_weight = sum(weights) if sum(weights) > 0 else 1.0
    pair_scores = {}

    for neighbor, w in zip(neighbors, weights):
        n_seq = neighbor["sequence"]
        n_struct = neighbor.get("structure", "")
        if not n_struct:
            continue

        transferred_pairs = transfer_neighbor_pairs_to_query(query_seq, n_seq, n_struct)

        for i, j in transferred_pairs:
            if j >= len(query_seq):
                continue
            if canonical_pair(query_seq[i], query_seq[j]):
                pair_scores[(i, j)] = pair_scores.get((i, j), 0.0) + w

    for pair in list(pair_scores.keys()):
        pair_scores[pair] /= total_weight

    return pair_scores


def select_non_conflicting_pairs(pair_scores, min_pair_freq=0.3):
    selected = []
    used = set()

    sorted_pairs = sorted(pair_scores.items(), key=lambda x: x[1], reverse=True)

    for (i, j), score in sorted_pairs:
        if score < min_pair_freq:
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


def fold_with_template_transfer(query_seq, neighbors, weights=None, min_pair_freq=0.3):
    fc = RNA.fold_compound(query_seq)

    if not neighbors:
        structure, _ = fc.mfe()
        return structure

    pair_scores = accumulate_transferred_pair_scores(query_seq, neighbors, weights=weights)
    selected_pairs = select_non_conflicting_pairs(pair_scores, min_pair_freq=min_pair_freq)
    constraint = pairs_to_constraint(len(query_seq), selected_pairs)

    if any(ch in "()" for ch in constraint):
        fc.constraints_add(constraint, RNA.CONSTRAINT_DB_DEFAULT)

    structure, _ = fc.mfe()
    return structure