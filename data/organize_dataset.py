import os
import itertools
import pandas as pd
from Bio import SeqIO
from Bio import pairwise2

# Base directory for protein family folders
BASE_DIR = "data"

# List of families under BASE_DIR
FAMILIES = ["actb", "brca1", "dmd", "hbb", "leptin", "p53", "pax6", "pten"]

# Final output file
OUTPUT_FILE = "pairs_with_features.csv"

def parse_fasta_header(header_line):
    """
    Given a FASTA header line like:
    >sp|P68871|HBB_HUMAN Hemoglobin subunit beta OS=Homo sapiens GN=HBB PE=1 SV=2

    This function returns:
    - uniprot_id: P68871
    - species: Homo sapiens
    """
    header_line = header_line.strip()
    if header_line.startswith(">"):
        header_line = header_line[1:]  # remove the '>'
    parts = header_line.split()

    # Default values in case parsing fails
    uniprot_id = ""
    species = ""

    # UniProt ID is usually between first and second '|', e.g. sp|P68871|HBB_HUMAN
    if "|" in header_line:
        pipe_parts = header_line.split("|")
        if len(pipe_parts) >= 2:
            uniprot_id = pipe_parts[1].strip()

    # Species comes after OS= and before the next tag (like GN=)
    if "OS=" in header_line:
        # split on OS=
        os_part = header_line.split("OS=")[1]
        # species name is a sequence of words until we hit something like 'GN=' or 'PE=' or 'SV='
        stop_tokens = ["GN=", "PE=", "SV=", "OX="]
        species_tokens = []
        for token in os_part.split():
            if any(token.startswith(st) for st in stop_tokens):
                break
            species_tokens.append(token)
        species = " ".join(species_tokens)

    return uniprot_id, species

def build_metadata():
    rows = []

    for family in FAMILIES:
        family_dir = os.path.join(BASE_DIR, family)
        if not os.path.isdir(family_dir):
            print(f"Warning: folder '{family_dir}' not found, skipping.")
            continue

        for filename in os.listdir(family_dir):
            if not filename.lower().endswith(".fasta"):
                continue

            filepath = os.path.join(family_dir, filename)

            # Read first line (header) of the FASTA file
            try:
                with open(filepath, "r") as f:
                    header = f.readline()
            except Exception as e:
                print(f"Error reading file {filepath}: {e}")
                continue

            uniprot_id, species = parse_fasta_header(header)

            rows.append({
                "file_name": filename,
                "family": family,
                "species": species,
                "uniprot_id": uniprot_id
            })

    df = pd.DataFrame(rows, columns=["file_name", "family", "species", "uniprot_id"])
    print(f"Built metadata for {len(df)} sequences.")
    return df
    
def build_pairs(metadata_df):
    """
    Given the metadata DataFrame, build all unique pairs of sequences.

    Returns:
        pandas.DataFrame with columns:
        seqA_file, seqB_file, familyA, familyB, label
        where label = 1 if same family, else 0.
    """
    rows = []
    
    # Use combinations to avoid duplicates and skip (A, A) pairs
    for i, j in itertools.combinations(metadata_df.index, 2):
        row_i = metadata_df.loc[i]
        row_j = metadata_df.loc[j]

        seqA = row_i["file_name"]
        seqB = row_j["file_name"]
        familyA = row_i["family"]
        familyB = row_j["family"]

        # label = 1 if same family, otherwise label = 0
        label = 1 if familyA == familyB else 0
        
        rows.append({
            "seqA_file": seqA,
            "seqB_file": seqB,
            "familyA": familyA,
            "familyB": familyB,
            "label": label,
        })
        
    pairs_df = pd.DataFrame(
        rows,
        columns=["seqA_file", "seqB_file", "familyA", "familyB", "label"],
    )
    
    num_total = len(pairs_df)
    num_pos = (pairs_df["label"] == 1).sum()
    num_neg = (pairs_df["label"] == 0).sum()

    print(f"Built {num_total} sequence pairs "
          f"({num_pos} positive / {num_neg} negative).")
    return pairs_df

def load_sequences(metadata_df):
    """
    Load all sequences into a dictionary for quick lookup:
        key   = file_name (e.g. HBB_HUMAN.fasta)
        value = sequence string
    """
    seq_dict = {}

    for _, row in metadata_df.iterrows():
        family = row["family"]
        fname = row["file_name"]

        # Build path like data/hbb/HBB_HUMAN.fasta
        fpath = os.path.join(BASE_DIR, family, fname)
        
        try:
            record = next(SeqIO.parse(fpath, "fasta"))
        except StopIteration:
            raise ValueError(f"No FASTA records found in {fpath}")
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {fpath}")

        seq_dict[fname] = str(record.seq)

    return seq_dict

def compute_alignment_features(seqA, seqB):
    """
    Compute alignment-based features between two sequences using Biopython's
    pairwise2.globalxx (match = 1, mismatch/gap = 0).

    Returns a dict of:
      - lenA, lenB, abs_len_diff
      - align_score
      - align_len
      - matches
      - percent_identity
      - gap_count
      - gap_fraction
    """

    lenA = len(seqA)
    lenB = len(seqB)
    abs_len_diff = abs(lenA - lenB)

    # Global alignment with simple scoring: match=1, mismatch=0, gap=0
    alignments = pairwise2.align.globalxx(seqA, seqB)
    best_alignment = alignments[0]
    alignedA, alignedB, score, start, end = best_alignment

    align_len = len(alignedA)

    # Count matches (excluding gaps)
    matches = sum(
        1 for a, b in zip(alignedA, alignedB)
        if a == b and a != "-" and b != "-"
    )

    # Gaps in both aligned sequences
    gapsA = alignedA.count("-")
    gapsB = alignedB.count("-")
    gap_count = gapsA + gapsB

    # Percent identity and gap fraction
    if align_len > 0:
        percent_identity = matches / align_len
        gap_fraction = gap_count / align_len
    else:
        percent_identity = 0.0
        gap_fraction = 0.0

    return {
        "lenA": lenA,
        "lenB": lenB,
        "abs_len_diff": abs_len_diff,
        "align_score": score,          # in globalxx, this is the number of matches
        "align_len": align_len,
        "matches": matches,
        "percent_identity": percent_identity,
        "gap_count": gap_count,
        "gap_fraction": gap_fraction,
    }
    
def main():
    # 1. Build metadata from FASTA files
    metadata_df = build_metadata()

    # 2. Build all sequence pairs (with labels)
    pairs_df = build_pairs(metadata_df)

    # 3. Load all sequences into memory
    print("Loading sequences from FASTA files...")
    seq_dict = load_sequences(metadata_df)
    print(f"Loaded {len(seq_dict)} sequences.")

    # 4. Compute alignment-based features for each pair
    feature_rows = []

    print("Computing features for each pair...")
    for idx, row in pairs_df.iterrows():
        seqA_file = row["seqA_file"]
        seqB_file = row["seqB_file"]
        familyA = row["familyA"]
        familyB = row["familyB"]
        label = row["label"]

        seqA = seq_dict[seqA_file]
        seqB = seq_dict[seqB_file]

        feats = compute_alignment_features(seqA, seqB)

        feature_rows.append({
            "seqA_file": seqA_file,
            "seqB_file": seqB_file,
            "familyA": familyA,
            "familyB": familyB,
            "lenA": feats["lenA"],
            "lenB": feats["lenB"],
            "abs_len_diff": feats["abs_len_diff"],
            "align_score": feats["align_score"],
            "align_len": feats["align_len"],
            "matches": feats["matches"],
            "percent_identity": feats["percent_identity"],
            "gap_count": feats["gap_count"],
            "gap_fraction": feats["gap_fraction"],
            "label": label,
        })

        # Optional: small progress indicator if many pairs
        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx + 1} / {len(pairs_df)} pairs...")

    # 5. Save final DataFrame to a single CSV
    features_df = pd.DataFrame(feature_rows)
    features_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved features for {len(features_df)} pairs to {OUTPUT_FILE}.")


if __name__ == "__main__":
    main()