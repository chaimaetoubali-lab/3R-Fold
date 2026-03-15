def dotbracket_to_pairs(ss):

    stack=[]
    pairs=set()

    for i,c in enumerate(ss):

        if c=="(":
            stack.append(i)

        elif c==")":

            j=stack.pop()

            pairs.add((j,i))

    return pairs


def evaluate_structure(pred,ref):

    pred_pairs=dotbracket_to_pairs(pred)

    ref_pairs=dotbracket_to_pairs(ref)

    sens=len(pred_pairs & ref_pairs)/len(ref_pairs) if ref_pairs else 0

    ppv=len(pred_pairs & ref_pairs)/len(pred_pairs) if pred_pairs else 0

    f1=2*sens*ppv/(sens+ppv) if sens+ppv>0 else 0

    return sens,ppv,f1
