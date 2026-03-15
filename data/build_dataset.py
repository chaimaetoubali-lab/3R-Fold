import gzip, os, random, io
import requests
from Bio import SeqIO, AlignIO
from collections import defaultdict
from datasets import load_dataset, Dataset

random.seed(42)

def load_external_datasets():

    def process_rnastralign(ex):
        return {
            "id": ex["id"],
            "family": ex["family"],
            "sequence": ex["sequence"],
            "structure": ex["secondary_structure"]
        }

    rnastralign = load_dataset("multimolecule/rnastralign", split="train").map(process_rnastralign)

    archiveii = load_dataset("multimolecule/archiveii", split="test").map(
        lambda ex: {
            "id": ex["id"],
            "family": ex.get("family", "ArchiveII"),
            "sequence": ex["sequence"],
            "structure": ex["secondary_structure"]
        }
    )

    rnastr_512 = load_dataset("multimolecule/rnastralign.512", split="train").map(process_rnastralign)
    rnastr_1024 = load_dataset("multimolecule/rnastralign.1024", split="train").map(process_rnastralign)

    return list(rnastralign) + list(rnastr_512) + list(rnastr_1024) + list(archiveii)

def balance_dataset(all_data, threshold=200):

    families = defaultdict(list)

    for item in all_data:
        families[item["family"]].append(item)

    filtered = {fam: seqs for fam, seqs in families.items() if len(seqs) >= threshold}

    balanced = []
    for fam, seqs in filtered.items():
        balanced.extend(random.sample(seqs, threshold))

    return Dataset.from_list(balanced)

def build_dataset(remove_rfam=True):
    external = load_external_datasets()
    all_data = external

    dataset = balance_dataset(all_data, threshold=200)

    print("Before filtering:", len(dataset))
    print("Families before:", sorted(set(dataset["family"])))

    if remove_rfam:
        dataset = dataset.filter(lambda x: x["family"] != "RfamFamily")

    print("After filtering:", len(dataset))
    print("Families after:", sorted(set(dataset["family"])))

    return dataset