#!/usr/bin/env Rscript

# Paired held-out evaluation of n-gram (A) and context-limited (B) surprisal.

lexical_control_variables <- c(
  'word_len', 'freq',
  'prev_word_len', 'prev_freq',
  'prev2_word_len', 'prev2_freq',
  'prev3_word_len', 'prev3_freq'
)

lexical_control_formula <- paste(
  'word_len*freq',
  'prev_word_len*prev_freq',
  'prev2_word_len*prev2_freq',
  'prev3_word_len*prev3_freq',
  sep=' + '
)

discover_predictors <- function(df, prefix) {
  variables <- grep(
    paste0('^', prefix, '[0-9]+$'), colnames(df), value=TRUE
  )
  if (length(variables) == 0) {
    stop(paste0('No predictors found with prefix ', prefix))
  }
  contexts <- as.integer(sub(prefix, '', variables, fixed=TRUE))
  ordering <- order(contexts)
  data.frame(
    context=contexts[ordering],
    variable=variables[ordering],
    stringsAsFactors=FALSE
  )
}

spillover_terms <- function(variable) {
  c(
    variable,
    paste0('prev_', variable),
    paste0('prev2_', variable),
    paste0('prev3_', variable)
  )
}

make_model_formula <- function(extra_variables=character()) {
  extra_terms <- unlist(lapply(extra_variables, spillover_terms), use.names=FALSE)
  right_hand_side <- lexical_control_formula
  if (length(extra_terms) > 0) {
    right_hand_side <- paste(
      right_hand_side, paste(extra_terms, collapse=' + '), sep=' + '
    )
  }
  as.formula(paste('time ~', right_hand_side))
}

required_analysis_columns <- function(ngram_variables, context_variables) {
  unique(c(
    'time', lexical_control_variables,
    unlist(lapply(c(ngram_variables, context_variables), spillover_terms),
           use.names=FALSE)
  ))
}

prepare_analysis_data <- function(df, ngram_variables, context_variables) {
  required <- required_analysis_columns(ngram_variables, context_variables)
  missing <- setdiff(required, colnames(df))
  if (length(missing) > 0) {
    stop(paste('Input is missing required columns:', paste(missing, collapse=', ')))
  }

  for (column in required) {
    if (!is.numeric(df[[column]])) {
      df[[column]] <- suppressWarnings(as.numeric(df[[column]]))
    }
  }
  complete <- complete.cases(df[, required, drop=FALSE])
  finite <- apply(
    df[, required, drop=FALSE], 1,
    function(row) all(is.finite(row))
  )
  df[complete & finite, , drop=FALSE]
}

make_folds <- function(n_rows, n_folds=10, seed=42) {
  if (n_folds < 2 || n_rows < n_folds) {
    stop('Need at least one observation in each of two or more folds')
  }
  set.seed(seed)
  shuffled_order <- sample(seq_len(n_rows))
  folds <- cut(seq_len(n_rows), breaks=n_folds, labels=FALSE)
  list(order=shuffled_order, folds=folds)
}

score_fold <- function(formula, train_data, test_data) {
  model <- lm(formula, data=train_data)
  design <- model.matrix(model)
  if (model$rank < ncol(design)) {
    stop(paste('Rank-deficient training model:', deparse(formula)))
  }
  sigma_squared <- mean(residuals(model)^2)
  if (!is.finite(sigma_squared) || sigma_squared <= 0) {
    stop(paste('Invalid residual variance for model:', deparse(formula)))
  }
  predictions <- predict(model, newdata=test_data)
  log_density <- dnorm(
    test_data$time,
    mean=predictions,
    sd=sqrt(sigma_squared),
    log=TRUE
  )
  if (any(!is.finite(log_density))) {
    stop(paste('Non-finite held-out likelihood for model:', deparse(formula)))
  }
  c(mean=mean(log_density), sum=sum(log_density))
}

score_formula <- function(formula, df, folds) {
  results <- vector('list', max(folds))
  for (fold in seq_len(max(folds))) {
    is_test <- folds == fold
    score <- score_fold(formula, df[!is_test, , drop=FALSE], df[is_test, , drop=FALSE])
    results[[fold]] <- data.frame(
      fold=fold,
      n_train=sum(!is_test),
      n_test=sum(is_test),
      ll_mean=unname(score[['mean']]),
      ll_sum=unname(score[['sum']])
    )
  }
  do.call(rbind, results)
}

make_pivot <- function(fold_results, metric, ngram_contexts, context_contexts,
                       statistic=c('mean', 'se')) {
  statistic <- match.arg(statistic)
  pivot <- data.frame(context_limited_context=context_contexts)
  for (ngram_context in ngram_contexts) {
    values <- sapply(context_contexts, function(context_context) {
      selected <- fold_results[
        fold_results$ngram_context == ngram_context &
        fold_results$context_limited_context == context_context,
        metric
      ]
      if (statistic == 'mean') {
        mean(selected)
      } else {
        sd(selected) / sqrt(length(selected))
      }
    })
    pivot[[paste0('ngram_context_', ngram_context)]] <- values
  }
  pivot
}

write_tsv_atomic <- function(dataframe, fname) {
  dir.create(dirname(fname), recursive=TRUE, showWarnings=FALSE)
  temporary_fname <- tempfile(
    pattern=paste0('.', basename(fname), '.'), tmpdir=dirname(fname)
  )
  write.table(
    dataframe, temporary_fname, quote=FALSE, sep='\t',
    row.names=FALSE, na='NA'
  )
  if (!file.rename(temporary_fname, fname)) {
    unlink(temporary_fname)
    stop(paste('Unable to atomically write', fname))
  }
}

run_joint_evaluation <- function(input_fname, fold_output_fname,
                                 pivot_a_fname, pivot_b_fname,
                                 se_a_fname, se_b_fname, summary_fname,
                                 n_folds=10, seed=42) {
  df_raw <- read.csv(input_fname, header=TRUE, sep='\t', check.names=FALSE)
  n_input <- nrow(df_raw)
  ngram <- discover_predictors(df_raw, 'ngram_surprisal_context_')
  context <- discover_predictors(
    df_raw, 'context_limited_surprisal_context_'
  )
  df <- prepare_analysis_data(df_raw, ngram$variable, context$variable)
  n_complete <- nrow(df)

  # A joint model has 13 control/intercept coefficients and four coefficients
  # for each surprisal family. Require residual degrees of freedom in every
  # training fold rather than returning a saturated pilot result.
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
  a_scores <- list()
  for (i in seq_len(nrow(ngram))) {
    key <- as.character(ngram$context[[i]])
    a_scores[[key]] <- score_formula(
      make_model_formula(ngram$variable[[i]]), df, folds
    )
  }
  b_scores <- list()
  for (i in seq_len(nrow(context))) {
    key <- as.character(context$context[[i]])
    b_scores[[key]] <- score_formula(
      make_model_formula(context$variable[[i]]), df, folds
    )
  }

  result_rows <- list()
  result_index <- 1
  for (a_index in seq_len(nrow(ngram))) {
    for (b_index in seq_len(nrow(context))) {
      a_context <- ngram$context[[a_index]]
      b_context <- context$context[[b_index]]
      joint_scores <- score_formula(
        make_model_formula(c(
          ngram$variable[[a_index]], context$variable[[b_index]]
        )),
        df,
        folds
      )
      a_only <- a_scores[[as.character(a_context)]]
      b_only <- b_scores[[as.character(b_context)]]
      result_rows[[result_index]] <- data.frame(
        ngram_context=a_context,
        context_limited_context=b_context,
        fold=joint_scores$fold,
        n_train=joint_scores$n_train,
        n_test=joint_scores$n_test,
        ll_m0_mean=m0_scores$ll_mean,
        ll_m0_a_mean=a_only$ll_mean,
        ll_m0_b_mean=b_only$ll_mean,
        ll_joint_mean=joint_scores$ll_mean,
        delta_a_given_b_mean=joint_scores$ll_mean - b_only$ll_mean,
        delta_b_given_a_mean=joint_scores$ll_mean - a_only$ll_mean,
        ll_m0_sum=m0_scores$ll_sum,
        ll_m0_a_sum=a_only$ll_sum,
        ll_m0_b_sum=b_only$ll_sum,
        ll_joint_sum=joint_scores$ll_sum,
        delta_a_given_b_sum=joint_scores$ll_sum - b_only$ll_sum,
        delta_b_given_a_sum=joint_scores$ll_sum - a_only$ll_sum
      )
      result_index <- result_index + 1
    }
  }
  fold_results <- do.call(rbind, result_rows)

  pivot_a <- make_pivot(
    fold_results, 'delta_a_given_b_mean',
    ngram$context, context$context, 'mean'
  )
  pivot_b <- make_pivot(
    fold_results, 'delta_b_given_a_mean',
    ngram$context, context$context, 'mean'
  )
  se_a <- make_pivot(
    fold_results, 'delta_a_given_b_mean',
    ngram$context, context$context, 'se'
  )
  se_b <- make_pivot(
    fold_results, 'delta_b_given_a_mean',
    ngram$context, context$context, 'se'
  )
  summary <- data.frame(
    key=c(
      'input_rows', 'complete_case_rows', 'excluded_rows', 'folds', 'seed',
      'response', 'delta_unit', 'ngram_contexts',
      'context_limited_contexts', 'controls'
    ),
    value=c(
      n_input, n_complete, n_input - n_complete, n_folds, seed,
      'raw reading time (ms)', 'mean held-out log density per observation',
      paste(ngram$context, collapse=','),
      paste(context$context, collapse=','), lexical_control_formula
    )
  )

  write_tsv_atomic(fold_results, fold_output_fname)
  write_tsv_atomic(pivot_a, pivot_a_fname)
  write_tsv_atomic(pivot_b, pivot_b_fname)
  write_tsv_atomic(se_a, se_a_fname)
  write_tsv_atomic(se_b, se_b_fname)
  write_tsv_atomic(summary, summary_fname)
  invisible(list(
    folds=fold_results, pivot_a=pivot_a, pivot_b=pivot_b,
    se_a=se_a, se_b=se_b, summary=summary
  ))
}

main <- function(args=commandArgs(trailingOnly=TRUE)) {
  if (length(args) < 7) {
    stop(paste(
      'Usage: rt_vs_joint_pilot_surprisal.R INPUT FOLDS PIVOT_A PIVOT_B',
      'SE_A SE_B SUMMARY [N_FOLDS] [SEED]'
    ))
  }
  n_folds <- if (length(args) >= 8) as.integer(args[[8]]) else 10
  seed <- if (length(args) >= 9) as.integer(args[[9]]) else 42
  run_joint_evaluation(
    args[[1]], args[[2]], args[[3]], args[[4]], args[[5]], args[[6]],
    args[[7]], n_folds=n_folds, seed=seed
  )
}

if (sys.nframe() == 0) {
  main()
}
