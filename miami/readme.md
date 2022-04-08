# Overview
This repository contains all the scripts needed to download the Bangor Miami Corpus and preprocess it for Speech Translation

## 1-Step Setup
0. Run `setup_all.sh` to download the data, and process it. For granular instructions, see the `Multi-Step Setup`

## Multi-Step Setup
0. Gather the data by running `bash download_miami_dataset.sh` which will place the data in `./data`
1. Format the data by running `python reformat_miami_data.py` which will output the data in `output/miami/*`. It will contain three files: a `yaml` file containing the timesteps, a `miami.transcript` containing the transcripts, and `miami.translation` containing the translations
2. Create code-switched and non-code-switched sections by running `python create_test_sets.py`
3. To create LID data, run `fisher/split_train_and_make_lid.py`


## Paper Reference
The Bangor Miami corpus is found [here](https://biling.talkbank.org/access/Bangor/Miami.html) and was published as part of [this paper](https://www.researchgate.net/publication/292243516_Building_bilingual_corpora)