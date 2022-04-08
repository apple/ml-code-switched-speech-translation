#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

# The Fisher data is already split into dev/dev2/test
# this script compiles these threeinto one `eval` test set
import os
import yaml
import shutil
from distutils.dir_util import copy_tree

DATASET_NAMES = ["cs", "mono"]
SPLITS = ["dev", "dev2", "test"]

# combine the evaluation sets
name = "eval"
base_output_path = f"output/fisher/{name}"

all_eval = {"cs": None, "mono": None}
for name in DATASET_NAMES:
    data_for_type = [[], [], []]
    for split in SPLITS:
        print(f"Loading the data for {name}, {split}...")
        base_path = f"output/fisher/{split}/{name}"
        transcript = []
        translation = []
        with open(f"{base_path}/fisher.yaml", "r") as fin:
            yaml_data = yaml.safe_load(fin)
        with open(f"{base_path}/fisher.transcript", "r") as fin:
            for line in fin:
                transcript.append(line.strip())
        with open(f"{base_path}/fisher.translation", "r") as fin:
            for line in fin:
                translation.append(line.strip())
        assert len(transcript) == len(yaml_data) == len(translation)
        print(f"Length of the original data is {len(transcript)}")

        data_for_type[0].extend(yaml_data)
        data_for_type[1].extend(transcript)
        data_for_type[2].extend(translation)

    all_eval[name] = data_for_type


print("Writing the combined data out...")
for (name, datasets) in zip(DATASET_NAMES, [all_eval["cs"], all_eval["mono"]]):
    print(f"Length of the data {name} is {len(datasets[0])}")

    if not os.path.isdir(os.path.join(base_output_path, name, "clips")):
        os.makedirs(os.path.join(base_output_path, name, "clips"))

    with open(os.path.join(base_output_path, name, f"fisher.yaml"), "w") as fout:
        fout.write(yaml.dump(datasets[0]))
    with open(os.path.join(base_output_path, name, f"fisher.transcript"), "w") as fout:
        for line in datasets[1]:
            assert "\n" not in line, line
            fout.write(line)
            fout.write("\n")
    with open(os.path.join(base_output_path, name, f"fisher.translation"), "w") as fout:
        for line in datasets[2]:
            assert "\n" not in line, line
            fout.write(line)
            fout.write("\n")

    print("Moving clip data...")
    for eval_split in SPLITS:
        copy_tree(
            os.path.join(base_output_path.replace("eval", eval_split), name, "clips"),
            os.path.join(base_output_path, name, "clips"),
        )

    # make it a zip file
    shutil.make_archive(
        os.path.join(base_output_path, name, "clips"),
        "zip",
        os.path.join(base_output_path, name, "clips"),
    )
