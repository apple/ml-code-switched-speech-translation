#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

# this script should set everything up
mkdir output
mkdir output/miami
mkdir data

bash download_miami_data.sh
python process_miami_data.py
python create_test_sets.py
