#!/usr/bin/env Rscript

# Paired held-out conditional tests for N vs L and C vs L surprisal.

source('src/h02_rt_model/rt_vs_joint_pilot_surprisal.R')


score_single_predictors <- function(predictors, df, folds) {
  scores <- list()
  for (index in seq_len(nrow(predictors))) {
    variable <- predictors$variable[[index]]
    scores[[variable]] <- score_formula(
      make_model_formula(variable), df, folds
    )
  }
  scores
}


validate_internal_layers <- function(layers) {
  if (any(layers$context < 1L)) {
    stop('Internal-layer indices must start at transformer layer 1')
  }
  layers
}


make_comparison_rows <- function(predictors, layers, predictor_scores,
                                 layer_scores, m0_scores, df, folds,
                                 predictor_family) {
  rows <- list()
  row_index <- 1
  for (predictor_index in seq_len(nrow(predictors))) {
    for (layer_index in seq_len(nrow(layers))) {
      predictor_variable <- predictors$variable[[predictor_index]]
      layer_variable <- layers$variable[[layer_index]]
      joint_scores <- score_formula(
        make_model_formula(c(predictor_variable, layer_variable)),
        df,
        folds
      )
      predictor_only <- predictor_scores[[predictor_variable]]
      layer_only <- layer_scores[[layer_variable]]
      delta_predictor_mean <- joint_scores$ll_mean - layer_only$ll_mean
      delta_layer_mean <- joint_scores$ll_mean - predictor_only$ll_mean
      delta_predictor_sum <- joint_scores$ll_sum - layer_only$ll_sum
      delta_layer_sum <- joint_scores$ll_sum - predictor_only$ll_sum
      is_ngram <- predictor_family == 'ngram'
      rows[[row_index]] <- data.frame(
        comparison=paste0(predictor_family, '_vs_internal_layer'),
        predictor_family=predictor_family,
        predictor_context=predictors$context[[predictor_index]],
        ngram_context=if (is_ngram) {
          predictors$context[[predictor_index]]
        } else {
          NA_integer_
        },
        context_limited_context=if (!is_ngram) {
          predictors$context[[predictor_index]]
        } else {
          NA_integer_
        },
        layer=layers$context[[layer_index]],
        fold=joint_scores$fold,
        n_train=joint_scores$n_train,
        n_test=joint_scores$n_test,
        ll_m0_mean=m0_scores$ll_mean,
        ll_predictor_mean=predictor_only$ll_mean,
        ll_layer_mean=layer_only$ll_mean,
        ll_joint_mean=joint_scores$ll_mean,
        delta_predictor_given_layer_mean=delta_predictor_mean,
        delta_layer_given_predictor_mean=delta_layer_mean,
        delta_n_given_l_mean=if (is_ngram) delta_predictor_mean else NA_real_,
        delta_l_given_n_mean=if (is_ngram) delta_layer_mean else NA_real_,
        delta_c_given_l_mean=if (!is_ngram) delta_predictor_mean else NA_real_,
        delta_l_given_c_mean=if (!is_ngram) delta_layer_mean else NA_real_,
        ll_m0_sum=m0_scores$ll_sum,
        ll_predictor_sum=predictor_only$ll_sum,
        ll_layer_sum=layer_only$ll_sum,
        ll_joint_sum=joint_scores$ll_sum,
        delta_predictor_given_layer_sum=delta_predictor_sum,
        delta_layer_given_predictor_sum=delta_layer_sum,
        delta_n_given_l_sum=if (is_ngram) delta_predictor_sum else NA_real_,
        delta_l_given_n_sum=if (is_ngram) delta_layer_sum else NA_real_,
        delta_c_given_l_sum=if (!is_ngram) delta_predictor_sum else NA_real_,
        delta_l_given_c_sum=if (!is_ngram) delta_layer_sum else NA_real_,
        stringsAsFactors=FALSE
      )
      row_index <- row_index + 1
    }
  }
  do.call(rbind, rows)
}


standard_error <- function(values) {
  if (length(values) < 2) {
    return(NA_real_)
  }
  sd(values) / sqrt(length(values))
}


aggregate_conditional_deltas <- function(fold_results) {
  keys <- unique(fold_results[, c(
    'comparison', 'predictor_family', 'predictor_context', 'layer'
  )])
  keys <- keys[order(
    keys$predictor_family, keys$predictor_context, keys$layer
  ), , drop=FALSE]
  rows <- vector('list', nrow(keys))
  for (index in seq_len(nrow(keys))) {
    key <- keys[index, , drop=FALSE]
    selected <- fold_results[
      fold_results$comparison == key$comparison &
      fold_results$predictor_context == key$predictor_context &
      fold_results$layer == key$layer,
      ,
      drop=FALSE
    ]
    rows[[index]] <- data.frame(
      comparison=key$comparison,
      predictor_family=key$predictor_family,
      predictor_context=key$predictor_context,
      ngram_context=if (key$predictor_family == 'ngram') {
        key$predictor_context
      } else {
        NA_integer_
      },
      context_limited_context=if (
        key$predictor_family == 'context_limited'
      ) {
        key$predictor_context
      } else {
        NA_integer_
      },
      layer=key$layer,
      folds=nrow(selected),
      delta_predictor_given_layer_mean=mean(
        selected$delta_predictor_given_layer_mean
      ),
      delta_predictor_given_layer_se=standard_error(
        selected$delta_predictor_given_layer_mean
      ),
      delta_layer_given_predictor_mean=mean(
        selected$delta_layer_given_predictor_mean
      ),
      delta_layer_given_predictor_se=standard_error(
        selected$delta_layer_given_predictor_mean
      ),
      delta_n_given_l_mean=if (key$predictor_family == 'ngram') {
        mean(selected$delta_predictor_given_layer_mean)
      } else {
        NA_real_
      },
      delta_n_given_l_se=if (key$predictor_family == 'ngram') {
        standard_error(selected$delta_predictor_given_layer_mean)
      } else {
        NA_real_
      },
      delta_l_given_n_mean=if (key$predictor_family == 'ngram') {
        mean(selected$delta_layer_given_predictor_mean)
      } else {
        NA_real_
      },
      delta_l_given_n_se=if (key$predictor_family == 'ngram') {
        standard_error(selected$delta_layer_given_predictor_mean)
      } else {
        NA_real_
      },
      delta_c_given_l_mean=if (
        key$predictor_family == 'context_limited'
      ) {
        mean(selected$delta_predictor_given_layer_mean)
      } else {
        NA_real_
      },
      delta_c_given_l_se=if (
        key$predictor_family == 'context_limited'
      ) {
        standard_error(selected$delta_predictor_given_layer_mean)
      } else {
        NA_real_
      },
      delta_l_given_c_mean=if (
        key$predictor_family == 'context_limited'
      ) {
        mean(selected$delta_layer_given_predictor_mean)
      } else {
        NA_real_
      },
      delta_l_given_c_se=if (
        key$predictor_family == 'context_limited'
      ) {
        standard_error(selected$delta_layer_given_predictor_mean)
      } else {
        NA_real_
      },
      stringsAsFactors=FALSE
    )
  }
  do.call(rbind, rows)
}


run_layer_evaluation <- function(input_fname, fold_output_fname,
                                 delta_output_fname, summary_fname,
                                 n_folds=10, seed=42,
                                 model='unspecified') {
  df_raw <- read.csv(input_fname, header=TRUE, sep='\t', check.names=FALSE)
  n_input <- nrow(df_raw)
  ngram <- discover_predictors(df_raw, 'ngram_surprisal_context_')
  context <- discover_predictors(
    df_raw, 'context_limited_surprisal_context_'
  )
  layers <- validate_internal_layers(discover_predictors(
    df_raw, 'internal_layer_surprisal_layer_'
  ))

  all_nonlayer_variables <- c(ngram$variable, context$variable)
  df <- prepare_analysis_data(
    df_raw, all_nonlayer_variables, layers$variable
  )
  n_complete <- nrow(df)

  joint_coefficient_count <- 21
  smallest_training_fold <- n_complete - ceiling(n_complete / n_folds)
  if (smallest_training_fold <= joint_coefficient_count) {
    stop(paste0(
      'Pilot is too small: the smallest training fold has ',
      smallest_training_fold, ' rows for ', joint_coefficient_count,
      ' joint-model coefficients'
    ))
  }

  fold_definition <- make_folds(n_complete, n_folds=n_folds, seed=seed)
  df <- df[fold_definition$order, , drop=FALSE]
  folds <- fold_definition$folds

  m0_scores <- score_formula(make_model_formula(), df, folds)
  ngram_scores <- score_single_predictors(ngram, df, folds)
  context_scores <- score_single_predictors(context, df, folds)
  layer_scores <- score_single_predictors(layers, df, folds)

  n_vs_l <- make_comparison_rows(
    ngram, layers, ngram_scores, layer_scores, m0_scores, df, folds,
    'ngram'
  )
  c_vs_l <- make_comparison_rows(
    context, layers, context_scores, layer_scores, m0_scores, df, folds,
    'context_limited'
  )
  fold_results <- rbind(n_vs_l, c_vs_l)
  fold_results <- cbind(
    model=rep(model, nrow(fold_results)),
    fold_results,
    stringsAsFactors=FALSE
  )
  conditional_deltas <- aggregate_conditional_deltas(fold_results)
  conditional_deltas <- cbind(
    model=rep(model, nrow(conditional_deltas)),
    conditional_deltas,
    stringsAsFactors=FALSE
  )

  summary <- data.frame(
    key=c(
      'model', 'input_rows', 'complete_case_rows', 'excluded_rows',
      'folds', 'seed',
      'response', 'delta_unit', 'ngram_contexts',
      'context_limited_contexts', 'internal_layers', 'layer_decoder',
      'layer_context', 'layer_chunking', 'controls'
    ),
    value=c(
      model, n_input, n_complete, n_input - n_complete, n_folds, seed,
      'raw reading time (ms)', 'mean held-out log density per observation',
      paste(ngram$context, collapse=','),
      paste(context$context, collapse=','),
      paste(layers$context, collapse=','), 'logit lens',
      paste(
        'wordsprobability-style overlapping context; BOS is reset for each',
        'chunk and continuation predictions retain a 199-token overlap'
      ),
      'wordsprobability 0.17: 1022 encoded tokens, stride 200',
      lexical_control_formula
    ),
    stringsAsFactors=FALSE
  )

  write_tsv_atomic(fold_results, fold_output_fname)
  write_tsv_atomic(conditional_deltas, delta_output_fname)
  write_tsv_atomic(summary, summary_fname)
  invisible(list(
    folds=fold_results, deltas=conditional_deltas, summary=summary
  ))
}


main <- function(args=commandArgs(trailingOnly=TRUE)) {
  if (length(args) < 4) {
    stop(paste(
      'Usage: rt_vs_internal_layer_surprisal.R INPUT FOLDS DELTAS SUMMARY',
      '[N_FOLDS] [SEED] [MODEL]'
    ))
  }
  n_folds <- if (length(args) >= 5) as.integer(args[[5]]) else 10
  seed <- if (length(args) >= 6) as.integer(args[[6]]) else 42
  model <- if (length(args) >= 7) args[[7]] else 'unspecified'
  run_layer_evaluation(
    args[[1]], args[[2]], args[[3]], args[[4]],
    n_folds=n_folds, seed=seed, model=model
  )
}


if (sys.nframe() == 0) {
  main()
}
