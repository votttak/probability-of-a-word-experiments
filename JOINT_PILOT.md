# Joint n-gram/context-limited reading-time pilot

This pilot tests the unique held-out contribution of n-gram surprisal (`A`) and
context-limited transformer surprisal (`B`) over the existing lexical controls
(`M0`):

```text
delta(A | B) = LL(M0 + A + B) - LL(M0 + B)
delta(B | A) = LL(M0 + A + B) - LL(M0 + A)
```

Run the default Natural Stories/GPT-2-small pilot with:

```bash
make -f MakefileJointPilot
```

The default sample contains the first 50 words of each passage (500 words
before complete-case filtering). Both predictor families are generated from
that exact text file, merged with the established averaged RT/control table,
and expanded to three preceding-word spillover positions.

Evaluation uses one shared complete-case sample, one deterministic set of ten
folds, and the controls used by the existing project:

```text
word_len * freq at the current word and preceding three words
```

Every cell compares paired held-out scores from identical folds. Results are
written under `results/rt/joint_pilot/`; the two `*-mean.tsv` files are the
requested 3-by-5 pivots, in mean held-out log density per observation. Matching
`*-se.tsv` files contain fold-level standard errors, and `fold-results.tsv`
retains all paired scores and both delta definitions.
