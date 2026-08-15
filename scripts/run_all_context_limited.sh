#!/bin/bash

# CONTEXT-LIMITED: Keep costly fixed-window inference separate from run_all.sh;
# invoke this script explicitly when every English model/dataset result is wanted.
for model in 'gpt2-small' 'gpt2-medium' 'gpt2-large' 'gpt2-xl' \
             'pythia-70m' 'pythia-160m' 'pythia-410m' 'pythia-14b' 'pythia-28b' 'pythia-69b' 'pythia-120b'
do
    for dataset in 'natural_stories' 'provo' 'dundee' 'brown'
    do
        # CONTEXT-LIMITED: Each command loads one model once and computes all
        # configured contexts before merging and running the RT analysis.
        make -f MakefileContextLimited get_context_limited_llh MODEL=${model} DATASET=${dataset}
    done
done
