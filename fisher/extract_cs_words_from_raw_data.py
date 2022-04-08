#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

# this file takes the raw Fisher data with the code-switched annotations and processes it
import glob
import os
from bs4 import BeautifulSoup
import numpy as np


def rawcount(filename):
    f = open(filename, "rb")
    lines = 0
    buf_size = 1024 * 1024
    read_f = f.raw.read

    buf = read_f(buf_size)
    while buf:
        lines += buf.count(b"\n")
        buf = read_f(buf_size)

    return lines


def fix_small_errors(line) -> str:
    """The data has some small errors to fix"""
    if 'lang+"English"' in line:
        line = line.replace('lang+"English"', 'lang="English"')
    if 'lan="English"' in line:
        line = line.replace('lan="English"', 'lang="English"')
    if " /foreign>" in line:
        line = line.replace(" /foreign>", "</foreign>")
    if '<foreign lang="English"> meeting <foreign lang="English">' in line:
        line = line.replace(
            '<foreign lang="English"> meeting <foreign lang="English">',
            '<foreign lang="English"> meeting </foreign>',
        )

    return line


# go through all spanish files, english don't have any markup since it
# generated from AMT and kept the text
file_info = {}
for file_path in glob.glob("fisher-callhome-corpus-tags/corpus/ldc/fisher_*.es"):
    file_name = file_path.split("/")[-1]
    cs_info = []
    file_info[file_name] = {
        "line_count": rawcount(file_path),
    }

    tokens_per_line = []
    with open(file_path, "r") as fin:
        for line_idx, line in enumerate(fin):
            line = line.strip()  # remove newline
            if (
                "<foreign" in line
            ):  # other tags exist like laughs, but we are only looking for code-switching

                line = fix_small_errors(line)
                soup = BeautifulSoup(line, features="html.parser")
                try:
                    inside_tags = soup.find_all(
                        "foreign"
                    )  # finds all foreign tags, in case of multiples
                    inside_texts = [
                        item.get_text() for item in inside_tags
                    ]  # extracts inside text
                    langs = [item["lang"] for item in inside_tags]
                    to_keep = [
                        idx
                        for (idx, item) in enumerate(inside_texts)
                        if item.strip() not in ["", "(())"]
                    ]
                    assert (
                        len(inside_tags) == len(inside_texts) == len(langs)
                    ), "got different lengths for same tags"

                    # get the CS text and dataset statistics
                    total_cs_text = [
                        item
                        for item in " ".join(inside_texts).strip().split(" ")
                        if item != ""
                    ]
                    line_tokens = [
                        item for item in soup.get_text().split(" ") if item != ""
                    ]
                    percent_cs = len(total_cs_text) / len(line_tokens)
                    tokens_per_line.append(percent_cs)
                    assert percent_cs <= 1.0, f"got {total_cs_text} for line {line}"

                except Exception as e:
                    breakpoint()
                    print(e)

                for idx in to_keep:
                    info = {
                        "cs_text": inside_texts[idx].strip(),
                        "lang": langs[idx],
                        "line_idx": line_idx,  # to map back
                    }
                    cs_info.append(info)

    file_info[file_name]["total_cs"] = len(cs_info)
    file_info[file_name]["cs_info"] = cs_info
    file_info[file_name]["cs_tokens_per_line"] = tokens_per_line

#### Analysis Section ####
output_path = "./cs_corpus"
if not os.path.isdir(output_path):
    os.makedirs(output_path)

for file_name in file_info.keys():
    print(f"\n## For file {file_name} ##")
    tokens_per_instance = np.array(file_info[file_name]["cs_tokens_per_line"])

    # ### make code-switched sets ###
    cs_file_name = file_name.replace(".es", "_cs.es")
    set_of_cs_idxs = set(
        [item["line_idx"] for item in file_info[file_name]["cs_info"]]
    )  # are duplicate line_nums
    with open(os.path.join(output_path, cs_file_name), "w") as fout:
        for idx in range(file_info[file_name]["line_count"]):
            if idx in set_of_cs_idxs:
                fout.write(str(idx))
                fout.write("\n")

    ### make non-code-switched sets ###
    mono_file_name = file_name.replace(".es", "_mono.es")
    with open(os.path.join(output_path, mono_file_name), "w") as fout:
        for idx in range(file_info[file_name]["line_count"]):
            if idx not in set_of_cs_idxs:
                fout.write(str(idx))
                fout.write("\n")

    # save CS words only for word tagging
    cs_words_file_name_only = file_name.replace(".es", "_cs_words_cs_only.es")
    cs_count = 0
    with open(os.path.join(output_path, cs_words_file_name_only), "w") as fout:
        for idx in range(file_info[file_name]["line_count"]):
            if idx in set_of_cs_idxs:
                cs_words = ""
                while (
                    cs_count < len(file_info[file_name]["cs_info"])
                    and file_info[file_name]["cs_info"][cs_count]["line_idx"] == idx
                ):
                    instance = file_info[file_name]["cs_info"][cs_count]
                    cs_count += 1
                    cs_words += instance["cs_text"] + " "

                assert (
                    instance["line_idx"] == idx
                ), f'Line idx at {instance["line_idx"]} with idx {idx}'
                fout.write(cs_words)
                fout.write("\n")
