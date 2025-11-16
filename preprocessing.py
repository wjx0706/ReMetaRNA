import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from Bio import SeqIO
from sklearn.model_selection import train_test_split

def read_fasta_first_seq(file_path):
    """
    Read a FASTA file and return the first sequence as a dict {id: sequence}.
    """
    sequences = {}
    for record in SeqIO.parse(file_path, "fasta"):
        sequences[record.id] = str(record.seq)
        break  # 只取第一个序列
    return sequences

def generate_train_valid_csv(seq_dir="./seqs", output_dir="./", seed=42, valid_ratio=0.2):
    """
    Reads FASTA files from seq_dir, extracts first sequence from each file,
    splits into train/validation, and writes CSV files to output_dir.
    """
    fasta_files = [f for f in os.listdir(seq_dir) if f.endswith(".fasta")]
    content_dict = {"pdb_id": [], "seq": []}

    for f in tqdm(fasta_files, desc="Processing FASTA files"):
        try:
            seqs = read_fasta_first_seq(os.path.join(seq_dir, f))
            if seqs:
                pdb_id, seq = list(seqs.items())[0]
                content_dict["pdb_id"].append(pdb_id)
                content_dict["seq"].append(seq)
        except Exception as e:
            print(f"Failed to parse {f}: {e}")

    df = pd.DataFrame(content_dict)
    train_df, valid_df = train_test_split(df, test_size=valid_ratio, random_state=seed)

    train_csv = os.path.join(output_dir, "public_train_data.csv")
    valid_csv = os.path.join(output_dir, "public_valid_data.csv")

    train_df.to_csv(train_csv, index=False)
    valid_df.to_csv(valid_csv, index=False)

    print(f"Train size: {len(train_df)}, Valid size: {len(valid_df)}")
    print(f"Saved train CSV: {train_csv}")
    print(f"Saved valid CSV: {valid_csv}")

    return train_df, valid_df


if __name__ == "__main__":
    generate_train_valid_csv(seq_dir="./seqs", output_dir="./", seed=42, valid_ratio=0.2)
