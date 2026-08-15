import os
import sys
import argparse
import re
import numpy as np
import pandas as pd

sys.path.insert(1, os.path.join(sys.path[0], '..'))
# from dataset import NaturalStoriesDataset
from h01_data.models import unigram
from utils import utils


# N-GRAM: Reproduce GPT-2's reversible bytes-to-Unicode alphabet only for
# alignment checks. wordsprobability uses this representation for some
# non-ASCII words, while Infini-gram must receive the original raw text.
def get_gpt2_byte_encoder():
    byte_values = (
        list(range(ord('!'), ord('~') + 1))
        + list(range(ord('¡'), ord('¬') + 1))
        + list(range(ord('®'), ord('ÿ') + 1))
    )
    unicode_values = byte_values.copy()
    extra_index = 0
    for byte_value in range(256):
        if byte_value not in byte_values:
            byte_values.append(byte_value)
            unicode_values.append(256 + extra_index)
            extra_index += 1
    return dict(zip(byte_values, map(chr, unicode_values)))


GPT2_BYTE_ENCODER = get_gpt2_byte_encoder()


def gpt2_byte_encode(text):
    return ''.join(GPT2_BYTE_ENCODER[byte] for byte in text.encode('utf8'))


def get_args():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--surprisal-fname', type=str, required=True)
    # N-GRAM: Optional keyed n-gram predictors keep the original LM-only
    # pipeline backward compatible.
    parser.add_argument('--ngram-surprisal-fname', type=str)
    # CONTEXT-LIMITED: The fixed-window LM table is also optional, preserving
    # both the original pipeline and the independently runnable n-gram branch.
    parser.add_argument('--context-limited-surprisal-fname', type=str)
    parser.add_argument('--rt-fname', type=str, required=True)
    parser.add_argument('--language', type=str, default='english')
    # Output
    parser.add_argument('--output-fname', type=str, required=True)

    return parser.parse_args()


def merge_ngram_surprisals(df_surprisals, ngram_surprisal_fname):
    """Validate and add keyed n-gram columns to the LM predictor table."""

    # N-GRAM: Join on stable IDs rather than row position, and retain a second
    # word column until token alignment has been checked.
    merge_columns = ['text_id', 'word_id']
    df_ngram = pd.read_csv(
        ngram_surprisal_fname, sep='\t', keep_default_na=False)
    required_columns = set(merge_columns + ['word'])
    missing_columns = required_columns - set(df_ngram.columns)
    if missing_columns:
        raise ValueError(
            'N-gram file is missing required columns: %s' %
            ', '.join(sorted(missing_columns)))

    ngram_columns = [
        column for column in df_ngram.columns
        if re.fullmatch(r'ngram_surprisal_context_\d+', column)
    ]
    if not ngram_columns:
        raise ValueError('N-gram file contains no ngram_surprisal_context_* columns')

    # N-GRAM: Duplicate keys would make spillover alignment ambiguous.
    for table_name, dataframe in [
            ('LM surprisal', df_surprisals), ('n-gram', df_ngram)]:
        duplicate_rows = dataframe.duplicated(merge_columns, keep=False)
        if duplicate_rows.any():
            duplicate_keys = dataframe.loc[duplicate_rows, merge_columns].head()
            raise ValueError(
                '%s file contains duplicate predictor keys:\n%s' %
                (table_name, duplicate_keys.to_string(index=False)))

    # N-GRAM: Reject blank, non-numeric, infinite, or missing predictor values
    # before R's global na.omit can silently change the evaluation sample.
    for column in ngram_columns:
        numeric_values = pd.to_numeric(df_ngram[column], errors='coerce')
        if numeric_values.isna().any() or not np.isfinite(numeric_values).all():
            raise ValueError('N-gram column %s contains non-finite values' % column)
        df_ngram[column] = numeric_values

    df_ngram = df_ngram[merge_columns + ['word'] + ngram_columns].rename(
        columns={'word': 'ngram_word'})
    merged = df_surprisals.merge(
        df_ngram,
        on=merge_columns,
        how='outer',
        validate='one_to_one',
        indicator=True)

    # N-GRAM: Both predictor generators must cover exactly the same word IDs.
    unmatched = merged['_merge'] != 'both'
    if unmatched.any():
        unmatched_keys = merged.loc[unmatched, merge_columns + ['_merge']].head()
        raise ValueError(
            'LM and n-gram predictor keys do not match:\n%s' %
            unmatched_keys.to_string(index=False))

    # N-GRAM: Accept raw equality or the deterministic GPT byte representation
    # used by wordsprobability; all other word mismatches remain fatal.
    encoded_ngram_words = merged['ngram_word'].map(gpt2_byte_encode)
    word_mismatch = (
        (merged['word'] != merged['ngram_word'])
        & (merged['word'] != encoded_ngram_words)
    )
    if word_mismatch.any():
        mismatches = merged.loc[
            word_mismatch, merge_columns + ['word', 'ngram_word']].head()
        raise ValueError(
            'LM and n-gram words do not match:\n%s' %
            mismatches.to_string(index=False))

    return merged.drop(columns=['ngram_word', '_merge'])


# CONTEXT-LIMITED: Keep this validation explicit rather than relying on row
# order; each generated score must match the existing LM word checkpoint.
def merge_context_limited_surprisals(
        df_surprisals, context_limited_surprisal_fname):
    """Validate and add keyed context-limited predictor columns."""

    # CONTEXT-LIMITED: Preserve stable zero-based IDs until after this merge.
    merge_columns = ['text_id', 'word_id']
    df_context = pd.read_csv(
        context_limited_surprisal_fname, sep='\t', keep_default_na=False)
    required_columns = set(merge_columns + ['word'])
    missing_columns = required_columns - set(df_context.columns)
    if missing_columns:
        raise ValueError(
            'Context-limited file is missing required columns: %s' %
            ', '.join(sorted(missing_columns)))

    # CONTEXT-LIMITED: Discover configured word windows dynamically so adding
    # another length needs no merge-code change.
    context_columns = [
        column for column in df_context.columns
        if re.fullmatch(
            r'context_limited_surprisal_context_\d+', column)
    ]
    if not context_columns:
        raise ValueError(
            'Context-limited file contains no '
            'context_limited_surprisal_context_* columns')

    # CONTEXT-LIMITED: Duplicate IDs would make current-word and spillover
    # predictors ambiguous, so reject them before joining.
    for table_name, dataframe in [
            ('LM/auxiliary surprisal', df_surprisals),
            ('context-limited', df_context)]:
        duplicate_rows = dataframe.duplicated(merge_columns, keep=False)
        if duplicate_rows.any():
            duplicate_keys = dataframe.loc[
                duplicate_rows, merge_columns].head()
            raise ValueError(
                '%s file contains duplicate predictor keys:\n%s' %
                (table_name, duplicate_keys.to_string(index=False)))

    # CONTEXT-LIMITED: Validate all scores before R's global na.omit can hide a
    # failed or partially written predictor condition.
    for column in context_columns:
        numeric_values = pd.to_numeric(df_context[column], errors='coerce')
        if numeric_values.isna().any() or not np.isfinite(numeric_values).all():
            raise ValueError(
                'Context-limited column %s contains non-finite values' % column)
        if (numeric_values < 0).any():
            raise ValueError(
                'Context-limited column %s contains negative values' % column)
        df_context[column] = numeric_values

    # CONTEXT-LIMITED: Retain a separate word column until equality has been
    # checked, then discard it so the merged schema has one canonical word.
    df_context = df_context[
        merge_columns + ['word'] + context_columns].rename(
            columns={'word': 'context_limited_word'})
    merged = df_surprisals.merge(
        df_context,
        on=merge_columns,
        how='outer',
        validate='one_to_one',
        indicator=True)

    # CONTEXT-LIMITED: Require complete coverage in both directions; a missing
    # first or final word must never silently shorten the regression sample.
    unmatched = merged['_merge'] != 'both'
    if unmatched.any():
        unmatched_keys = merged.loc[
            unmatched, merge_columns + ['_merge']].head()
        raise ValueError(
            'LM and context-limited predictor keys do not match:\n%s' %
            unmatched_keys.to_string(index=False))

    # CONTEXT-LIMITED: As with n-grams, accept the deterministic GPT-2 byte
    # alphabet used by wordsprobability while rejecting all real mismatches.
    encoded_context_words = merged['context_limited_word'].map(gpt2_byte_encode)
    word_mismatch = (
        (merged['word'] != merged['context_limited_word'])
        & (merged['word'] != encoded_context_words)
    )
    if word_mismatch.any():
        mismatches = merged.loc[
            word_mismatch,
            merge_columns + ['word', 'context_limited_word']].head()
        raise ValueError(
            'LM and context-limited words do not match:\n%s' %
            mismatches.to_string(index=False))

    return merged.drop(columns=['context_limited_word', '_merge'])


def merge_rt_and_surprisal(args):
    merge_columns = ['text_id', 'word_id']

    df_surprisals = pd.read_csv(args.surprisal_fname, sep='\t', keep_default_na=False)
    # N-GRAM: Validate n-gram alignment while both predictor tables still use
    # the same zero-based text and word IDs.
    if args.ngram_surprisal_fname:
        df_surprisals = merge_ngram_surprisals(
            df_surprisals, args.ngram_surprisal_fname)
    # CONTEXT-LIMITED: Validate fixed-window scores on the same zero-based keys
    # before the legacy one-based RT text-index adjustment below.
    if args.context_limited_surprisal_fname:
        df_surprisals = merge_context_limited_surprisals(
            df_surprisals, args.context_limited_surprisal_fname)
    df_surprisals['text_id'] = df_surprisals['text_id'] + 1 # Fix text indexing

    df_rt = pd.read_csv(args.rt_fname, sep='\t', index_col=0, keep_default_na=False)
    df_rt = df_rt[~df_rt.outlier]
    del df_rt['outlier']
    df_rt = df_rt.groupby(merge_columns + ['ref_token']).agg('mean').reset_index()

    df = df_rt.set_index(merge_columns).join(
        df_surprisals.set_index(merge_columns),
        how='outer').reset_index()
    
    assert not df.surprisal.isna().any()
    if args.language not in {'he', 'gr'}:
        assert (df.word == df.ref_token).all()
    return df


def get_frequencies(df, language):
    df['freq'] = df['word'].apply(
        lambda x: unigram.frequency(x, lang=language))


def get_spillover_vars(df):
    # N-GRAM: Sort explicitly so every shift follows reading order even if an
    # upstream merge changes row ordering.
    df.sort_values(['text_id', 'word_id'], kind='stable', inplace=True)

    # N-GRAM: Discover configured context lengths from the merged columns, so
    # no Python code change is needed when NGRAM_CONTEXT_LENGTHS changes.
    ngram_variables = [
        column for column in df.columns
        if re.fullmatch(r'ngram_surprisal_context_\d+', column)
    ]
    # CONTEXT-LIMITED: Discover every configured fixed word window and give it
    # the same three within-text spillover positions as existing predictors.
    context_limited_variables = [
        column for column in df.columns
        if re.fullmatch(
            r'context_limited_surprisal_context_\d+', column)
    ]
    variables = [
        'word', 'surprisal', 'surprisal_buggy', 'freq', 'word_len'
    ] + ngram_variables + context_limited_variables
    for variable in variables:
        df['prev_' + variable] = df.groupby("text_id", sort=False)[variable].shift(periods=1, fill_value=None)
        df['prev2_' + variable] = df.groupby("text_id", sort=False)[variable].shift(periods=2, fill_value=None)
        df['prev3_' + variable] = df.groupby("text_id", sort=False)[variable].shift(periods=3, fill_value=None)


def get_rt_with_surprisal_dataset(args):
    df = merge_rt_and_surprisal(args)
    get_frequencies(df, args.language)
    get_spillover_vars(df)

    return df


def main():
    args = get_args()
    df = get_rt_with_surprisal_dataset(args)
    utils.write_tsv(df, args.output_fname)


if __name__ == '__main__':
    main()
