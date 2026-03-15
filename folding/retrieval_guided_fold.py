import RNA

def consensus_structure(structures):

    L = len(structures[0])
    consensus = []

    for i in range(L):

        chars = [s[i] for s in structures if i < len(s)]

        if chars.count("(") + chars.count(")") > len(chars)/2:
            consensus.append("(")
        else:
            consensus.append(".")

    return "".join(consensus)


def fold_with_neighbors(sequence, neighbors):

    neighbor_structs = [n["structure"] for n in neighbors]

    constraint = consensus_structure(neighbor_structs)

    fc = RNA.fold_compound(sequence)
    fc.constraints_add(constraint, RNA.CONSTRAINT_DB)

    structure, energy = fc.mfe()

    return structure