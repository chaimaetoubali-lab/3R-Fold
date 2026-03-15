import RNA
import torch

def fold_with_retrieval(seq,h_ret):

    fc = RNA.fold_compound(seq)

    n=len(seq)

    bias=float(torch.norm(h_ret).item())

    soft_bp=[[bias]*n for _ in range(n)]

    fc.sc_set_bp(soft_bp)

    ss,mfe=fc.mfe()

    return ss