#! /usr/bin/env python3

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

# this file extracts the utterance from the larger audio file given the mapping

import sys
import os
import subprocess

if len(sys.argv) != 3:
  print("Usage: %s <mapping file> <LDC-speech-dir>" % sys.argv[0])
  sys.exit(1)
srcAudioDir=sys.argv[2]

utterance = None
mapping = {}
for line in sys.stdin:
  if line.startswith('##'):
    utterance = line.strip().split(' ')[2]
    lineno = 1
  else:
    mapping[(utterance,repr(lineno))] = line.strip()
    lineno += 1

for lineno, line in enumerate(open(sys.argv[1])):
  utterances, ids = line.split()
  output = " ".join(mapping[(utterances,x)] for x in ids.split('_'))
  uttList=[mapping[(utterances,x)] for x in ids.split('_')]
  firstToks=uttList[0].split('+')
  firstToks[4] = firstToks[4].replace(' ', '~')
  uttStart=float(firstToks[2])
  uttDur=float(uttList[-1].split('+')[3])-uttStart
  audioName="%s-utt%06d" % (os.path.basename(sys.argv[1]), lineno+1)
  uttID="%s-%s-c%s-%s" % (audioName, firstToks[0], firstToks[1], firstToks[4])
  spkID="%s-c%s-%s" % (firstToks[0], firstToks[1], firstToks[4])
  wavFilename=os.path.join(os.path.basename(sys.argv[1]), os.path.join(firstToks[0][:-4], audioName))
  print(uttID, wavFilename, spkID, lineno+1, output, uttStart, uttDur) # used in the `prepare-sets.sh bash script`
  directory = os.path.dirname(wavFilename)
  try:
    os.stat(directory)
  except:
    os.mkdir(directory)       
  cmd="/usr/bin/sox %s -c 1 --encoding signed-integer %s.wav remix %d trim %f %f rate 16000" % (os.path.join(srcAudioDir, firstToks[0]), wavFilename, int(firstToks[1])+1, uttStart, uttDur)
  print(uttID, repr(subprocess.check_output(cmd.split(" "))), file=sys.stderr)
