#!/bin/bash

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#


# This file converts the speech data into 16K audio and creates the YAML files containing the mapping

FISHER_TDF_DIR=${LDC2010T04}/data/transcripts
FISHER_SPEECH_DIR=${LDC2010S01}/data/speech
PARALLEL_DATA_DIR=fisher-callhome-corpus

process_audio()
{
  C=${1}
  mkdir -p ${C}
  echo  ${PARALLEL_DATA_DIR}/mapping/${C}
  ufiles=$(cat ${PARALLEL_DATA_DIR}/mapping/$C | cut -d' ' -f1 | uniq)
  # echo $ufiles
  for i in $ufiles; do
      echo "## FILE $i"
      cat ${FISHER_TDF_DIR}/$i.tdf | 
	grep -v ";;MM" | grep -v "file;unicode" |
	cut -f 1-5 | tr '\t' '+'
  done | python extract-utterance-audios.py ${PARALLEL_DATA_DIR}/mapping/$C ${FISHER_SPEECH_DIR} > ${C}/ids \
	   2>${C}.prepare-audio.log
}

for SET in fisher_{train,dev,dev2,test}; do
  process_audio ${SET}
done

# make YAML audio mapping
for convname in fisher_{train,dev,dev2,test}/*fsp; do
    for filename in $convname/*.wav; do
        echo "- { wav: $filename }" >> $(dirname $convname).yaml
    done
done
