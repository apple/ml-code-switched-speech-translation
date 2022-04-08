#! /bin/bash

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

# This script downloads the miami corpus from its repository and converts it into 16K audio

# start by downloading their repository
DIRECTORY="data/miami"
if [ ! -d "$DIRECTORY" ]; then
    echo "cloning corpus, which contains the CHAT files with text and mappings"
    cd data
    git clone https://github.com/donnekgit/miami.git
    mkdir miami/audio
    cd ../
fi

echo "downloading audio files"

declare -a audio=("herring1" "herring2" "herring3" 
 "herring5" "herring6" "herring7" "herring8" 
 "herring9" "herring10" "herring11" "herring12"
 "herring13" "herring14" "herring15" "herring16"
 "herring17" "maria1" "maria2" "maria3" "maria4"
 "maria7" "maria10" "maria16" "maria18" "maria19"
 "maria20" "maria21" "maria24" "maria27" "maria30"
 "maria31" "maria40" "sastre1" "sastre2" "sastre3"
 "sastre4" "sastre5" "sastre6" "sastre7" "sastre8" 
 "sastre9" "sastre10" "sastre11" "sastre12" "sastre13"
 "zeledon1" "zeledon2" "zeledon3" "zeledon4" "zeledon5"
 "zeledon6" "zeledon7" "zeledon8" "zeledon9" "zeledon11"
 "zeledon13" "zeledon14")

# Download each of the above files
for i in "${audio[@]}"
do
   if [ ! -f "data/miami/audio/$i.mp3" ]; then
    echo "Downloading $i"
    wget -P data/miami/audio/ http://bangortalk.bangor.ac.uk/$i.mp3
   fi
done

# convert each file to 16 bit wav files
for i in "${audio[@]}"
do
   if [ ! -f "data/miami/audio/$i.wav" ]; then
    echo "converting mp3 to wav $i"
    ffmpeg -i data/miami/audio/$i.mp3 -acodec pcm_s16le -ac 1 -ar 16000 data/miami/audio/$i.wav
   fi
done

