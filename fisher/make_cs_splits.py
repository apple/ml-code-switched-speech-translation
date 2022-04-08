#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

# This file does the initial splitting from the Fisher ASR splits into a CS and monolingual sets
import os
import yaml
import shutil

DATASET_NAMES = ["cs", "mono"]
SPLITS = ["dev", "dev2", "test", "train"]


def split_data():
    for split in SPLITS:
        print(f"Loading the data for {split}...")
        base_path = f"splits_data/{split}"
        base_output_path = f"output/fisher/{split}"
        transcript = []
        translation = []
        with open(f"{base_path}/fisher_{split}.yaml", "r") as fin:
            yaml_data = yaml.safe_load(fin)
        with open(f"{base_path}/fisher_{split}.es", "r") as fin:
            for line in fin:
                transcript.append(line.strip())
        translation_path = (
            f"{base_path}/fisher_{split}.en.0"
            if split != "train"
            else f"{base_path}/fisher_{split}.en"
        )
        with open(
            translation_path, "r"
        ) as fin:  # many refs, we use en.0 although others are possible
            for line in fin:
                translation.append(line.strip())

        assert len(transcript) == len(yaml_data) == len(translation)
        print(f"Length of the original data is {len(transcript)}")

        ## Load code switched indexes ##
        mono_idxs = []
        cs_idxs = []
        with open(f"cs_corpus/fisher_{split}_mono.es", "r") as fin:
            for line in fin:
                mono_idxs.append(line.strip())
        with open(f"cs_corpus/fisher_{split}_cs.es", "r") as fin:
            for line in fin:
                cs_idxs.append(line.strip())
        assert len(cs_idxs) + len(mono_idxs) == len(transcript)
        cs_idxs = set(cs_idxs)

        mono = [[], [], []]
        cs = [[], [], []]
        print("Separating the data...")
        for idx in range(len(yaml_data)):
            new_yaml_instance = yaml_data[idx]
            new_yaml_instance["old_wav"] = new_yaml_instance["wav"]
            new_yaml_instance["wav"] = (
                "clips/" + new_yaml_instance["wav"].split("/")[-1]
            )

            if str(idx) in cs_idxs:
                cs[0].append(new_yaml_instance)
                cs[1].append(transcript[idx])
                cs[2].append(translation[idx])
            else:
                mono[0].append(new_yaml_instance)
                mono[1].append(transcript[idx])
                mono[2].append(translation[idx])

        print("Writing the data out...")
        for (name, datasets) in zip(DATASET_NAMES, [cs, mono]):
            print(f"Length of the data {name} is {len(datasets[0])}")
            if not os.path.isdir(os.path.join(base_output_path, name)):
                os.makedirs(os.path.join(base_output_path, name))
            with open(
                os.path.join(base_output_path, name, f"fisher.yaml"), "w"
            ) as fout:
                fout.write(yaml.dump(datasets[0]))
            with open(
                os.path.join(base_output_path, name, f"fisher.transcript"), "w"
            ) as fout:
                for line in datasets[1]:
                    assert "\n" not in line, line
                    fout.write(line)
                    fout.write("\n")
            with open(
                os.path.join(base_output_path, name, f"fisher.translation"), "w"
            ) as fout:
                for line in datasets[2]:
                    assert "\n" not in line, line
                    fout.write(line)
                    fout.write("\n")

        print("Moving clip data...")
        mono_clips = [item["old_wav"] for item in mono[0]]
        cs_clips = [item["old_wav"] for item in cs[0]]
        audio_path = "speech"
        assert len(mono_clips) == len(mono[0])
        assert len(cs_clips) == len(cs[0])
        for (name, file_paths) in zip(DATASET_NAMES, [cs_clips, mono_clips]):
            for file_path in file_paths:
                file_ending = "/".join(
                    file_path.split("/")[-2:]
                )  # last two are the ones we need
                if not os.path.isdir(os.path.join(base_output_path, name, "clips")):
                    os.makedirs(os.path.join(base_output_path, name, "clips"))

                shutil.copy(
                    os.path.join(audio_path, f"fisher_{split}", file_ending),
                    os.path.join(
                        base_output_path, name, "clips", file_path.split("/")[-1]
                    ),
                )

            # make it a zip file
            shutil.make_archive(
                os.path.join(base_output_path, name, "clips"),
                "zip",
                os.path.join(base_output_path, name, "clips"),
            )


if __name__ == "__main__":
    split_data()
