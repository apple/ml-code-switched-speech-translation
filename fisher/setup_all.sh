#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

# get the Fisher data with CS tags
git clone https://github.com/orionw/fisher-callhome-corpus.git
mv fisher-callhome-corpus fisher-callhome-corpus-tags
cd fisher-callhome-corpus-tags
make
cd ../
python extract_cs_words_from_raw_data.py # makes indexes of CS data and keeps the CS words

# make the clean data without the CS tags to use
git clone -b keep_tags https://github.com/orionw/fisher-callhome-corpus.git
cd fisher-callhome-corpus
make
cp corpus/ldc/fisher_dev.{en,es}* ../splits_data/dev/
cp corpus/ldc/fisher_train.{en,es}* ../splits_data/train/
cp corpus/ldc/fisher_dev2.{en,es}* ../splits_data/dev2/
cp corpus/ldc/fisher_test.{en,es}* ../splits_data/test/
cd ../

# prepare the speech data (process to 16K, match to the other data lines)
bash prepare-sets.sh
cp fisher_train.yaml splits_data/train/
cp fisher_test.yaml splits_data/test/
cp fisher_dev.yaml splits_data/dev/
cp fisher_dev2.yaml splits_data/dev2/
mkdir speech
mv fisher_train speech
mv fisher_dev speech
mv fisher_dev2 speech
mv fisher_test speech

# make the CS and Monolingual splits
sed -i "s/\r//g" splits_data/*/*  # something adds extra carriage returns
python make_cs_splits.py
# make the `eval` set consisting of dev dev2 test
python combine_eval_splits.py
python split_train_and_make_lid.py # split into training and dev CS sets and determine the LID
python make_mapping_files.py # if you want the mapping files, optional
