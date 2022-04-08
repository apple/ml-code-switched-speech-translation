# Overview
This repository contains the code and instructions needed to reproduce the dataset splits for ["Speech Translation for Code-Switched Speech"](LINK_TODO).

You can create both datasets with the `bash create_datasets.sh` command, following the instructions in the [Instructions Section](#Instructions). The `fisher` and `miami` directories contain the scripts needed to for each dataset used by `bash create_datasets.sh`. 

A mapping between the original data and the new code-switched and monolingual splits used in the paper can be found in `mapping_files`. Note that running `bash create_datasets.sh` will create these mappings.

## Instructions
0. Install the prerequisite libraries for linux/macOS.  This includes `ffmpeg`, `sox`, `wget`, and `python` (e.g. `apt-get install sox`).
1. Run `pip install -r requirement.txt` to setup the python enviroment
2. Collect the data needed for the Fisher corpus ([LDC2010T04](https://catalog.ldc.upenn.edu/LDC2010T04) and [LDC2010S01](https://catalog.ldc.upenn.edu/LDC2010S01)) and export them: `export LDC2010S01={path_to_LDC2010S01}` and `export LDC2010T04={path_to_LDC2010T04}/fisher_spa_tr`.
3. Run `bash create_datasets.sh` to generate both Miami and Fisher datasets. 


## Example

Example utterance:
- (Audio clip)
- Transcript (code-switched): *y ti bueno tiene dos papás **which can be a little can be a little challenging**.*
- Translation (English only): *and she has two fathers which can be a little, can be a little challenging.*

The data files are composed of three parts:
1. The transcript for the dataset split (in `{dataset_name}.translation`)
2. The translation for the dataset split (in `{dataset_name}.translation`)
3. The audio for the dataset split (in `{dataset_name}.yaml` and `{dataset_name}/clips/*.wav` or `{dataset_name}/clips.zip`)

## Citation
If you found this repository helpful in your research, please consider citing
```
Orion Weller, Matthias Sperber, Telmo Pessoa Pires, Hendra Setiawan, Christian Gollan, Dominic Telaar, Matthias Paulik: End-to-End Speech Translation for Code Switched Speech (Findings of the Association for Computational Linguistics: ACL 2022)
```
