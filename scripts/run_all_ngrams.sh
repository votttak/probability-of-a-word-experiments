#!/bin/bash

# N-GRAM: This is intentionally separate from run_all.sh because generating
# remote Infini-gram counts is expensive. Run it explicitly when n-gram results
# are wanted; existing experiments remain unchanged.
for model in 'gpt2-small' 'gpt2-medium' 'gpt2-large' 'gpt2-xl' \
             'pythia-70m' 'pythia-160m' 'pythia-410m' 'pythia-14b' 'pythia-28b' 'pythia-69b' 'pythia-120b'
do
    for dataset in 'natural_stories' 'provo' 'dundee' 'brown'
    do
        # N-GRAM: Make reuses the dataset-level count cache and n-gram TSV for
        # every neural model, while producing model-specific baseline results.
        make -f MakefileNgrams get_ngram_llh MODEL=${model} DATASET=${dataset}
    done
done
