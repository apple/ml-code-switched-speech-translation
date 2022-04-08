#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

# This file creates the LID labels for training/dev as well as splitting the training CS set into dev/train
import os
import random
import shutil
import yaml
import string
import numpy as np

random.seed(1)


def create_and_save_labels_for_cs_train_data(
    transcript, transcript_train, cs_words, output_path, desc, name
) -> list:
    """Only used for fisher_train_cs to save train and dev set labels"""
    labels1 = []
    labels2 = []
    cs_words1, cs_words2 = cs_words
    for idx, instance in enumerate(transcript):
        transcript = instance.translate(str.maketrans("", "", string.punctuation))
        cs_words = cs_words1[idx].translate(str.maketrans("", "", string.punctuation))

        cs_count = len(cs_words.strip().split(" "))
        all_count = len(transcript.strip().split(" "))
        if cs_count / all_count > 0.5:
            labels1.append(0)  # english
        elif cs_count / all_count < 0.5:
            labels1.append(1)  # spanish
        else:
            labels1.append(int(random.random() > 0.5))

    for idx, instance in enumerate(transcript_train):
        transcript = instance.translate(str.maketrans("", "", string.punctuation))
        cs_words = cs_words2[idx].translate(str.maketrans("", "", string.punctuation))

        cs_count = len(cs_words.strip().split(" "))
        all_count = len(transcript.strip().split(" "))
        if cs_count / all_count > 0.5:
            labels2.append(0)  # english
        elif cs_count / all_count < 0.5:
            labels2.append(1)  # spanish
        else:
            labels2.append(int(random.random() > 0.5))

    print(
        f"Averages: labels1={np.array(labels1).mean()} labels2={np.array(labels2).mean()}"
    )

    if not os.path.isdir(os.path.join(output_path, desc + "_dev")):
        os.makedirs(os.path.join(output_path, desc + "_dev"))
    if not os.path.isdir(os.path.join(output_path, desc + "_train")):
        os.makedirs(os.path.join(output_path, desc + "_train"))

    with open(os.path.join(output_path, desc + "_dev", "lid_labels.txt"), "w") as fout:
        for label in labels1:
            fout.write(str(label))
            fout.write("\n")

    with open(
        os.path.join(output_path, desc + "_train", "lid_labels.txt"), "w"
    ) as fout:
        for label in labels2:
            fout.write(str(label))
            fout.write("\n")

    return labels1, labels2


def write_out_data(
    yaml_data, transcript, translation, base_path, output_path, desc, name
):
    """A helper function for writing out all the data"""
    if not os.path.isdir(os.path.join(output_path, desc, "clips")):
        os.makedirs(os.path.join(output_path, desc, "clips"))

    with open(os.path.join(output_path, desc, f"{name}.yaml"), "w") as fout:
        fout.write(yaml.dump(yaml_data))
    with open(os.path.join(output_path, desc, f"{name}.transcript"), "w") as fout:
        for line in transcript:
            assert "\n" not in line, line
            fout.write(line)
            fout.write("\n")
    with open(os.path.join(output_path, desc, f"{name}.translation"), "w") as fout:
        for line in translation:
            assert "\n" not in line, line
            fout.write(line)
            fout.write("\n")

    for instance in yaml_data:
        audio_path = instance["wav"]
        shutil.copy(
            os.path.join(base_path, audio_path),
            os.path.join(output_path, desc, "clips", audio_path.split("/")[-1]),
        )

    # make it a zip file
    shutil.make_archive(
        os.path.join(output_path, desc, "clips"),
        "zip",
        os.path.join(output_path, desc, "clips"),
    )


def sample_yaml_data(
    yaml_data, transcript, translation, num_idxs_to_sample, return_both: bool = False, should_write_out: bool = False
):
    """A helper function for sampling from the data"""
    split_idx = np.array(
        random.sample(list(range(len(yaml_data))), k=num_idxs_to_sample)
    )
    if should_write_out:
        with open("train_vs_dev_cs.txt", "w") as fout:
            for line in split_idx.tolist():
                fout.write(str(line))
                fout.write("\n")
    bool_split = np.isin(np.arange(len(transcript)), split_idx)
    transcript1 = np.array(transcript)[bool_split].tolist()
    translation1 = np.array(translation)[bool_split].tolist()
    yaml_data1 = np.array(yaml_data)[bool_split].tolist()
    if return_both:
        cs_words = []
        with open("cs_corpus/fisher_train_cs_words_cs_only.es", "r") as fin:
            for line in fin:
                cs_words.append(line.strip())
        assert len(cs_words) == len(yaml_data)
        cs_words1 = np.array(cs_words)[bool_split].tolist()
        cs_words2 = np.array(cs_words)[~bool_split].tolist()

        transcript2 = np.array(transcript)[~bool_split].tolist()
        translation2 = np.array(translation)[~bool_split].tolist()
        yaml_data2 = np.array(yaml_data)[~bool_split].tolist()
        return (
            yaml_data1,
            transcript1,
            translation1,
            yaml_data2,
            transcript2,
            translation2,
            (cs_words1, cs_words2),
        )
    else:
        return yaml_data1, transcript1, translation1


def create_and_save_cs_labels_only(yaml_data, transcript, translation):
    """A function that only creates the LID labels and saves them (only used for Fisher Eval CS)"""
    assert len(yaml_data) == len(transcript) == len(translation)

    cs_words_list = []
    for file_type in ["dev", "dev2", "test"]:
        with open(f"cs_corpus/fisher_{file_type}_cs_words_cs_only.es", "r") as fin:
            for line in fin:
                cs_words_list.append(line.strip())

    assert len(cs_words_list) == len(
        yaml_data
    ), f"CS words: {len(cs_words_list)} len_data={len(yaml_data)}"

    labels = []
    for idx, instance in enumerate(transcript):
        transcript_str = instance.translate(str.maketrans("", "", string.punctuation))
        cs_words = cs_words_list[idx].translate(
            str.maketrans("", "", string.punctuation)
        )

        cs_count = len(cs_words.strip().split(" "))
        all_count = len(transcript_str.strip().split(" "))
        if cs_count / all_count > 0.5:
            labels.append(0)  # english
        elif cs_count / all_count < 0.5:
            labels.append(1)  # spanish
        else:
            labels.append(int(random.random() > 0.5))

    assert len(labels) == len(yaml_data)

    with open(os.path.join("output/fisher/eval/cs/fisher.labels"), "w") as fout:
        for label in labels:
            fout.write(str(label))
            fout.write("\n")


def gather_lid_data():
    output_path = "output/lid"
    num_idxs_to_sample = None
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    data_paths = [
        ("fisher_eval_cs", "fisher", "output/fisher/eval/cs"),
        ("fisher_train_cs", "fisher", "output/fisher/train/cs"),
        # for the monolingual ones, sample some of them for use in LID training
        ("fisher_train_mono", "fisher", "output/fisher/train/mono"),
        ("miami_train_mono", "miami", "../miami/output/miami/mono_train"),
    ]
    for (desc, name, base_path) in data_paths:
        print(f"Working on {desc}")
        transcript = []
        translation = []
        with open(f"{base_path}/{name}.yaml", "r") as fin:
            yaml_data = yaml.safe_load(fin)
        with open(f"{base_path}/{name}.transcript", "r") as fin:
            for line in fin:
                transcript.append(line.strip())
        with open(f"{base_path}/{name}.translation", "r") as fin:
            for line in fin:
                translation.append(line.strip())
        assert len(transcript) == len(yaml_data) == len(translation)

        if desc == "fisher_eval_cs":
            create_and_save_cs_labels_only(yaml_data, transcript, translation)
        elif desc == "fisher_train_cs":
            # need to split this into train and dev, then save
            yaml_data, transcript, translation, yaml_data_train, transcript_train, translation_train, cs_words = sample_yaml_data(yaml_data, transcript, translation, 
                                                                                                                            int(0.1 * len(yaml_data)), return_both=True,
                                                                                                                            should_write_out=True)
            print(f"Length of the data {base_path}/{name + '_dev'} is {len(yaml_data)}")
            create_and_save_labels_for_cs_train_data(transcript, transcript_train, cs_words, output_path, desc, name)
            write_out_data(yaml_data, transcript, translation, base_path, output_path, desc + "_dev", name)

            print(f"Length of the data {base_path}/{name + '_train'} is {len(yaml_data_train)}")
            write_out_data(yaml_data_train, transcript_train, translation_train, base_path, output_path, desc + "_train", name)
            num_idxs_to_sample = len(yaml_data_train) # make fisher cs the base
        else: # is monolingual
            yaml_data, transcript, translation = sample_yaml_data(yaml_data, transcript, translation, min(len(yaml_data), num_idxs_to_sample))
            print(f"Length of the data {base_path}/{name} is {len(yaml_data)}")
            write_out_data(yaml_data, transcript, translation, base_path, output_path, desc, name)


if __name__ == "__main__":
    gather_lid_data()
