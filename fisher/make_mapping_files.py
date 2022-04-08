#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import os
import numpy as np
import pandas as pd

FILE_LEN_MAP = {
    "dev": 3979,
    "dev2": 3961,
    "test": 3641,
    "train": 138819
}

def make_mappings():
    SPLITS = ["dev", "dev2", "test", "train"]
    idx_map = {}
    for cs_type in ["cs", "mono"]:
        for split in SPLITS:
            loaded_idxs = []
            with open(f"cs_corpus/fisher_{split}_{cs_type}.es") as fin:
                for line in fin:
                    loaded_idxs.append(int(line.strip()))
            idx_map[f"{split}-{cs_type}"] = set(loaded_idxs)

    # Build Eval/Test set 
    cur_split_data = []
    for split in ["dev", "dev2", "test"]:
        split_audio_path = pd.read_csv(f"fisher-callhome-corpus-tags/mapping/fisher_{split}", index_col=None, delimiter=" ", header=None)
        split_audio_path.columns = ["AudioFile", "LineNum"]
        for idx in range(FILE_LEN_MAP[split]):
            audio_details = split_audio_path.loc[idx]
            details = {"file": f"fisher_{split}", "file_line_num": idx, "split": "test",
                         "audio_file": audio_details["AudioFile"], "audio_file_line_num": audio_details["LineNum"]}
            if idx in idx_map[f"{split}-cs"]:
                details["cs_type"] = "cs"
            else:
                details["cs_type"] = "mono"
            cur_split_data.append(details)

    # Training and Dev sets
    cs_count = 0
    cs_is_dev = []
    with open("train_vs_dev_cs.txt", "r") as fin:
        for line in fin:
            cs_is_dev.append(int(line.strip()))
    cs_is_dev = set(cs_is_dev)
    split_audio_path = pd.read_csv(f"fisher-callhome-corpus-tags/mapping/fisher_train", index_col=None, delimiter=" ", header=None)
    split_audio_path.columns = ["AudioFile", "LineNum"]
    for idx in range(FILE_LEN_MAP["train"]):
        audio_details = split_audio_path.loc[idx]
        details = {"file": f"fisher_train", "file_line_num": idx, "split": "train",
                        "audio_file": audio_details["AudioFile"], "audio_file_line_num": audio_details["LineNum"]}
        if idx in idx_map[f"train-cs"]:
            details["cs_type"] = "cs"
            if cs_count in cs_is_dev:
                details["split"] = "dev"
            cs_count += 1
        else:
            details["cs_type"] = "mono"
        cur_split_data.append(details)

    df = pd.DataFrame(cur_split_data)
    df.to_csv(f"fisher_mapping.csv")
    print("Made mapping files for Fisher")

if __name__ == "__main__":
    make_mappings()
