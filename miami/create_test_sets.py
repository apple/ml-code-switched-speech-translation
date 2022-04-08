#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

# this file takes all of the Miami data and turns it into splits
import os
import yaml
import json
from tqdm import tqdm
import shutil
import numpy as np
import random
import pandas as pd

random.seed(1)

DATASET_NAMES = ["cs", "mono"]

def map_cs(x: str) -> str:
    if x == "n/a":
        return x
    elif "cs" in x:
        return "cs"
    else:
        return "mono"

def map_split(x: str) -> str:
    if x == "n/a":
        return x
    elif "train" in x:
        return "train"
    else:
        return "test"

def split_data():
    print("Loading the data...")
    base_path = "output/miami/all"
    base_output_path = "output/miami"
    transcript = []
    translation = []
    with open(f"{base_path}/miami.yaml", "r") as fin:
        yaml_data = yaml.safe_load(fin)
    with open(f"{base_path}/miami.transcript", "r") as fin:
        for line in fin:
            transcript.append(line.strip())
    with open(f"{base_path}/miami.translation", "r") as fin:
        for idx, line in enumerate(fin):
            translation.append(line.strip())
    assert len(transcript) == len(yaml_data) == len(translation), [len(transcript), len(yaml_data), len(translation)]
    print(f"Length of the original data is {len(transcript)}")

    mono = [[], [], []]
    cs = [[], [], []]
    print("Separating the data...")
    data_type = []
    mono_count = 0
    mono_map = {}
    offsets = []
    durations = []
    files = []
    local_file_lines = []
    for idx in tqdm(range(len(yaml_data)), leave=True):
        # get values for making a mapping file
        local_line_num = yaml_data[idx]["wav"].split("/")[1].split("_")[-1].split(".")[0].replace("p", "")
        files.append(yaml_data[idx]["wav"].split("/")[1].split("_")[0])
        offsets.append(yaml_data[idx]["offset"])
        durations.append(yaml_data[idx]["duration"])
        local_file_lines.append(local_line_num)

        if (
            translation[idx] not in ["", "\n"] and transcript[idx] != translation[idx]
        ):  # same would be not helpful

            yaml_instance = yaml_data[idx]
            if yaml_instance["duration"] < 0.3:  # remove instances that are too short
                data_type.append("n/a")
                continue

            yaml_instance["offset"] = 0  # we're making each their own file

            if yaml_data[idx]["code_switched"]:
                cs[0].append(yaml_instance)
                cs[1].append(transcript[idx])
                cs[2].append(translation[idx])
                data_type.append("cs")
            else:
                mono[0].append(yaml_instance)
                mono[1].append(transcript[idx])
                mono[2].append(translation[idx])
                data_type.append("mono")
                mono_map[mono_count] = idx
                mono_count += 1
        else:
            data_type.append("n/a")

    # split the mono data
    mono[0] = np.array(mono[0])
    mono[1] = np.array(mono[1])
    mono[2] = np.array(mono[2])
    split_mono_idx = np.array(
        random.sample(list(range(len(mono[0]))), len(mono[0]) // 2)
    )
    global_map_from_mono = sorted([mono_map[cur_idx] for cur_idx in split_mono_idx])
    bool_split = np.isin(np.arange(len(mono[0])), split_mono_idx)
    mono_train = [
        mono[0][bool_split].tolist(),
        mono[1][bool_split].tolist(),
        mono[2][bool_split].tolist(),
    ]
    mono = [
        mono[0][~bool_split].tolist(),
        mono[1][~bool_split].tolist(),
        mono[2][~bool_split].tolist(),
    ]

    # make a mapping file for others to use
    data_type = [dtype if idx not in global_map_from_mono else "mono_train" \
                    for idx, dtype in enumerate(data_type)]
    mapping_val = pd.DataFrame({"global_idx": list(range(len(data_type))), "split": data_type,
                                "file": files, "file_line_num": local_file_lines, 
                                "offset": offsets, "duration": durations})

    mapping_val["cs_type"] = mapping_val.split.apply(lambda x: map_cs(x))
    mapping_val["split"] = mapping_val.split.apply(lambda x: map_split(x))                        
    mapping_val.to_csv("miami_mapping.csv", index=None)

    print("Writing the data out...")
    for (name, datasets) in zip(DATASET_NAMES + ["mono_train"], [cs, mono, mono_train]):
        print(f"Length of the data {name} is {len(datasets[0])}")
        if not os.path.isdir(os.path.join(base_output_path, name)):
            os.makedirs(os.path.join(base_output_path, name))
        with open(os.path.join(base_output_path, name, f"miami.jsonl"), "w") as fout:
            for segment in datasets[0]:
                fout.write(json.dumps(segment))
                fout.write("\n")
        with open(os.path.join(base_output_path, name, f"miami.yaml"), "w") as fout:
            fout.write(yaml.dump(datasets[0], allow_unicode=True))
        with open(
            os.path.join(base_output_path, name, f"miami.transcript"), "w"
        ) as fout:
            for line in datasets[1]:
                assert "\n" not in line, line
                fout.write(line)
                fout.write("\n")
        with open(
            os.path.join(base_output_path, name, f"miami.translation"), "w"
        ) as fout:
            for line in datasets[2]:
                assert "\n" not in line, line
                fout.write(line)
                fout.write("\n")

    print("Moving clip data...")
    mono_clips = [item["wav"] for item in mono[0]]
    mono_train_clips = [item["wav"] for item in mono_train[0]]
    cs_clips = [item["wav"] for item in cs[0]]
    assert len(mono_clips) == len(mono[0])
    assert len(cs_clips) == len(cs[0])
    assert len(mono_train_clips) == len(mono_train[0])
    for (name, file_paths) in zip(
        DATASET_NAMES + ["mono_train"], [cs_clips, mono_clips, mono_train_clips]
    ):
        for file_path in file_paths:
            if not os.path.isdir(os.path.join(base_output_path, name, "clips")):
                os.makedirs(os.path.join(base_output_path, name, "clips"))
            shutil.copy(
                os.path.join(base_path, file_path),
                os.path.join(base_output_path, name, file_path),
            )

        # make it a zip file
        shutil.make_archive(
            os.path.join(base_output_path, name, "clips"),
            "zip",
            os.path.join(base_output_path, name, "clips"),
        )


if __name__ == "__main__":
    split_data()
