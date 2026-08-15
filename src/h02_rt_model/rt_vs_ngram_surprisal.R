#!/usr/bin/env Rscript

# N-GRAM: This opt-in analysis mirrors rt_vs_surprisal.R while adding every
# ngram_surprisal_context_* column found in the input dataset. The original
# neural-LM analysis script remains unchanged.
source("src/h02_rt_model/r_utils.R")
source("src/h02_rt_model/predictors.R")
source("src/h02_rt_model/baselines.R")

# N-GRAM: Command-line arguments intentionally match the existing R analysis.
list[input_fname,output_fname,params_output_fname_base,merge_workers,is_linear] <- get_args()
is_linear = as.logical(is_linear)
merge_workers = as.logical(merge_workers)
n_folds = 10
tgt_var = 'time'

set.seed(42)
df <- load_and_preprocess_data(input_fname, merge_workers)

if (is_linear) {
  predictors <- c()

  # N-GRAM: Preserve both existing LM predictors and append configured n-gram
  # predictors in numeric context-length order.
  variables <- c('surprisal', 'surprisal_buggy')
  ngram_variables <- grep(
    '^ngram_surprisal_context_[0-9]+$', colnames(df), value=TRUE)
  if (length(ngram_variables) == 0) {
    stop('No ngram_surprisal_context_* predictors found in input data')
  }
  context_lengths <- as.integer(sub(
    '^ngram_surprisal_context_', '', ngram_variables))
  ngram_variables <- ngram_variables[order(context_lengths)]
  variables <- c(variables, ngram_variables)

  for (variable in variables) {
    predictors <- c(predictors, get_variable_predictors_all(variable))
  }
} else {
  sys.exit()
}

print(paste0('Processing dataset ',input_fname))

# N-GRAM: Use exactly the same lexical controls and neural-surprisal baselines
# as the existing analysis so delta log-likelihood values remain comparable.
if (is_linear & merge_workers) {
  essential_predictors <- 'word_len*freq + prev_freq*prev_word_len + prev2_freq*prev2_word_len + prev3_freq*prev3_word_len'
  baselines <- get_baselines()

  baseline_llhs <- hash()
  for (baseline in baselines) {
    baseline_name <- baseline[['name']]
    baseline_function <- baseline[['function']]

    baseline_llhs[[baseline_name]] <- get_baseline_score(tgt_var, essential_predictors, baseline_function)
  }
} else if (is_linear && !merge_workers) {
  sys.exit()
} else if (!is_linear && merge_workers) {
  sys.exit()
} else if (!is_linear && !merge_workers) {
  sys.exit()
}


# N-GRAM: The result schema is kept identical to the legacy LLH output so
# downstream tools can read either experiment.
full_diffs <- list()
full_diffs['predictor'] <- c()
full_diffs['predictor_type'] <- c()
full_diffs['name'] <- c()
full_diffs['fold'] <- c()
for(baseline_name in keys(baseline_llhs)) {
  full_diffs[paste0('diff_',baseline_name)] <- c()
}

for(i in 1:length(predictors)){
  predictor = predictors[[i]][['function']]
  predictor_type = predictors[[i]][['type']]
  name = predictors[[i]][['name']]

  print(paste0('Processing dataset ',input_fname,' with predictor ',predictor))
  if (is_linear && merge_workers) {
    formula = paste0(tgt_var," ~ ",predictor," + ",essential_predictors)
    list[full_llh,full_models] <- lme_cross_val(formula, df, tgt_var, random_effects=FALSE)
  } else if (is_linear && !merge_workers) {
    sys.exit()
  } else if (!is_linear && merge_workers) {
    sys.exit()
  } else if (!is_linear && !merge_workers) {
    sys.exit()
  }

  full_diffs[['fold']] <- c(full_diffs[['fold']], c(1:n_folds))
  full_diffs[['predictor']] <- c(full_diffs[['predictor']], rep(predictor, n_folds))
  full_diffs[['predictor_type']] <- c(full_diffs[['predictor_type']], rep(predictor_type, n_folds))
  full_diffs[['name']] <- c(full_diffs[['name']], rep(name, n_folds))

  for(baseline_name in keys(baseline_llhs)) {
    full_diffs[[paste0('diff_',baseline_name)]] <- c(
      full_diffs[[paste0('diff_',baseline_name)]],
      full_llh - baseline_llhs[[baseline_name]]
    )
  }

  models_params <- c()
  for(j in 1:length(full_models)) {
    models_params <- rbind(models_params, full_models[[j]][[1]])
  }
  params_fname <- paste0(params_output_fname_base,'-predictor_',name,'-type_',predictor_type,'.tsv')
  write.table(models_params,params_fname, quote=FALSE, sep='\t')
}

df_full <- c()
df_full <- cbind(df_full,
  full_diffs[['predictor']], full_diffs[['predictor_type']],
  full_diffs[['name']], full_diffs[['fold']],
  full_diffs[['diff_empty']], full_diffs[['diff_full_surprisal']]
  )

colnames(df_full) <- c(
  'predictor', 'predictor_type', 'name', 'fold',
  'diff_empty', 'diff_full_surprisal'
  )

write.table(df_full, output_fname, quote=FALSE, sep='\t')
