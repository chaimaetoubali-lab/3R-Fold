import pandas as pd

def summarize(results):

    rows=[]

    for m in results:

        rows.append({
            "id":m["id"],
            "family":m["family"],
            "f1":m["f1"],
            "sensitivity":m["sensitivity"],
            "ppv":m["ppv"]
        })

    return pd.DataFrame(rows)