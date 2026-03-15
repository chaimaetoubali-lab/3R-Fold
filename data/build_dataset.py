import gzip, os, random, io
import requests
from Bio import AlignIO
from collections import defaultdict
from datasets import load_dataset, Dataset

random.seed(42)

RFAM_SEED_URL = "ftp://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.seed.gz"
SEED_LOCAL = "Rfam.seed.gz"

def download_rfam():
    if not os.path.exists(SEED_LOCAL):
        print("Downloading Rfam.seed.gz...")
        http_url = RFAM_SEED_URL.replace("ftp://", "https://")
        response = requests.get(http_url, stream=True)
        with open(SEED_LOCAL, "wb") as f:
            for chunk in response.iter_content(8192):
                f.write(chunk)

def parse_rfam():
    rfam_data = []
    if os.path.exists(SEED_LOCAL):
        with gzip.open(SEED_LOCAL, "rb") as handle:
            with io.TextIOWrapper(handle, encoding="utf-8", errors="ignore") as text_handle:
                alignments = AlignIO.parse(text_handle, "stockholm")
                for alignment in alignments:
                    struct = alignment.column_annotations.get("SS_cons", "")
                    fam = alignment.annotations.get("RFAM_ACCESSION", "RfamFamily")
                    for record in alignment:
                        rfam_data.append({
                            "id": f"{fam}_{record.id}",
                            "family": fam,
                            "sequence": str(record.seq).replace("-", ""),
                            "structure": struct,
                        })
    return rfam_data

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

def build_dataset():

    download_rfam()
    rfam = parse_rfam()

    external = load_external_datasets()

    all_data = external + rfam

    dataset = balance_dataset(all_data)

    print(dataset)

    return dataset