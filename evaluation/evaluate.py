from evaluation.metrics import evaluate_structure
from folding.vienna_fold import fold_with_retrieval

def evaluate_model(query_seq, ref_structure, kb_embeddings, topk_idx):

    h_ret = kb_embeddings[topk_idx].mean(dim=0)

    pred_ss = fold_with_retrieval(query_seq, h_ret)

    sens, ppv, f1 = evaluate_structure(pred_ss, ref_structure)

    return {
        "pred_structure": pred_ss,
        "sensitivity": sens,
        "ppv": ppv,
        "f1": f1
    }