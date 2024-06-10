# ---
# jupyter:
#   jupytext:
#     formats: ipynb,R:light
#     text_representation:
#       extension: .R
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: R
#     language: R
#     name: ir
# ---

# # NAGuide R methods
#
# Setup basic methods and packages used for all methods
#
# - BiocManager could be moved to methods who are installed from BioConductor

# + tags=["hide-input"] vscode={"languageId": "r"}
# options("install.lock"=FALSE)

packages_base_R <-
  c("BiocManager", "reshape2", "data.table", "readr", "tibble")

install_rpackage  <- function(pkg) {
  # If not installed, install the package
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
  
}

# used in the large imputation function for two packages
install_bioconductor  <- function(pkg) {
  # If not installed, install the package
  if (!require(pkg, character.only = TRUE)) {
    BiocManager::install(pkg)
    library(pkg, character.only = TRUE)
  }
  
}


for (package in packages_base_R) {
  # Check if the package is already installed
  install_rpackage(pkg = package)
}

# -

# setup can be tricky... trying to integrate as much as possible into conda environment

# Copied from [NAGuideR's github](https://github.com/wangshisheng/NAguideR/blob/15ec86263d5821990ad39a8d9f378cf4d76b25fb/inst/NAguideRapp/app.R#L1705-L1849) RShiny application. Adapted to run as standalone function in context of the Snakemake workflow.
#
# - `df` and `df1` ?
# - seems quite hacky
# - code is only slightly adapted from repo to run here, mainly to install packages on the fly

# + tags=["hide-input"] vscode={"languageId": "r"}
nafunctions <- function(x, method = "zero") {
  df <- df1 <- as.data.frame(x)
  method <- tolower(method)
  if (method == "zero") {
    df[is.na(df)] <- 0
  }
  else if (method == "minimum") {
    df[is.na(df)] <- min(df1, na.rm = TRUE)
  }
  else if (method == "colmedian") {
    install_rpackage('e1071')
    df <- impute(df1, what = "median")
  }
  else if (method == "rowmedian") {
    install_rpackage('e1071')
    dfx <- impute(t(df1), what = "median")
    df <- t(dfx)
  }
  else if (method == "knn_impute") {
    install_bioconductor('impute')
    data_zero1 <-
      impute.knn(as.matrix(df1),
                 k = 10,
                 rowmax = 1,
                 colmax = 1)#rowmax = 0.9, colmax = 0.9
    df <- data_zero1$data
  }
  else if (method == "seqknn") {
    if (!require(SeqKnn)) {
      install.packages("src/R_NAGuideR/SeqKnn_1.0.1.tar.gz",
                       repos = NULL,
                       type = "source")
      library(SeqKnn)
    }
    df <- SeqKNN(df1, k = 10)
  }
  else if (method == "bpca") {
    install_bioconductor('pcaMethods')
    data_zero1 <-
      pcaMethods::pca(
        as.matrix(df1),
        nPcs = ncol(df1) - 1,
        method = "bpca",
        maxSteps = 100
      )
    df <- completeObs(data_zero1)
  }
  else if (method == "svdmethod") {
    install_bioconductor('pcaMethods')
    data_zero1 <-
      pcaMethods::pca(as.matrix(df1),
                      nPcs = ncol(df1) - 1,
                      method = "svdImpute")
    df <- completeObs(data_zero1)
  }
  else if (method == "lls") {
    install_bioconductor('pcaMethods')
    data_zero1 <- llsImpute(t(df1), k = 10)
    df <- t(completeObs(data_zero1))
  }
  else if (method == "mle") {
    install_rpackage('norm')
    xxm <- as.matrix(df1)
    ss <- norm::prelim.norm(xxm)
    thx <- norm::em.norm(ss)
    norm::rngseed(123)
    df <- norm::imp.norm(ss, thx, xxm)
  }
  else if (method == "qrilc") {
    install_bioconductor("impute")
    install_bioconductor("pcaMethods")
    install_rpackage('gmm')
    install_rpackage('imputeLCMD')
    xxm <- t(df1)
    data_zero1 <-
      imputeLCMD::impute.QRILC(xxm, tune.sigma = 1)[[1]]
    df <- t(data_zero1)
  }
  else if (method == "mindet") {
    install_bioconductor("impute")
    install_bioconductor("pcaMethods")
    install_rpackage('gmm')
    install_rpackage('imputeLCMD')
    xxm <- as.matrix(df1)
    df <- imputeLCMD::impute.MinDet(xxm, q = 0.01)
  }
  else if (method == "minprob") {
    install_bioconductor("impute")
    install_bioconductor("pcaMethods")
    install_rpackage('gmm')
    install_rpackage('imputeLCMD')
    xxm <- as.matrix(df1)
    df <-
      imputeLCMD::impute.MinProb(xxm, q = 0.01, tune.sigma = 1)
  }
  else if (method == "irm") {
    install_rpackage('VIM')
    df <- irmi(df1, trace = TRUE, imp_var = FALSE)
    rownames(df) <- rownames(df1)
  }
  else if (method == "impseq") {
    install_rpackage('rrcovNA')
    df <- impSeq(df1)
  }
  else if (method == "impseqrob") {
    install_rpackage('rrcovNA')
    data_zero1 <- impSeqRob(df1, alpha = 0.9)
    df <- data_zero1$x
  }
  else if (method == "mice-norm") {
    install_rpackage('mice')
    minum <- 5
    datareadmi <- mice(df1,
                       m = minum,
                       seed = 1234,
                       method = "norm")
    newdatareadmi <- 0
    for (i in 1:minum) {
      newdatareadmi <- complete(datareadmi, action = i) + newdatareadmi
    }
    df <- newdatareadmi / minum
    rownames(df) <- rownames(df1)
  }
  else if (method == "mice-cart") {
    install_rpackage('mice')
    minum <- 5
    datareadmi <- mice(df1,
                       m = minum,
                       seed = 1234,
                       method = "cart")
    newdatareadmi <- 0
    for (i in 1:minum) {
      newdatareadmi <- complete(datareadmi, action = i) + newdatareadmi
    }
    df <- newdatareadmi / minum
    rownames(df) <- rownames(df1)
  }
  else if (method == "trknn") {
    source('src/R_NAGuideR/Imput_funcs.r')
    # sim_trKNN_wrapper <- function(data) {
    #   result <- data %>% as.matrix %>% t %>% imputeKNN(., k=10, distance='truncation', perc=0) %>% t
    #   return(result)
    # }
    # df1x <- sim_trKNN_wrapper(t(df1))
    # df<-as.data.frame(t(df1x))
    df <-
      imputeKNN(as.matrix(df),
                k = 10,
                distance = 'truncation',
                perc = 0)
    df <- as.data.frame(df)
  }
  else if (method == "rf") {
    install_rpackage("missForest")
    data_zero1 <- missForest(
      t(df1),
      maxiter = 10,
      ntree = 20 # input$rfntrees
      ,
      mtry = floor(nrow(df1) ^ (1 / 3)),
      verbose = TRUE
    )
    df <- t(data_zero1$ximp)
  }
  else if (method == "pi") {
    width <- 0.3 # input$piwidth
    downshift <- 1.8 # input$pidownshift
    for (i in 1:ncol(df1)) {
      temp <- df1[[i]]
      if (sum(is.na(temp)) > 0) {
        temp.sd <- width * sd(temp[!is.na(temp)], na.rm = TRUE)
        temp.mean <-
          mean(temp[!is.na(temp)], na.rm = TRUE) - downshift * sd(temp[!is.na(temp)], na.rm = TRUE)
        n.missing <- sum(is.na(temp))
        temp[is.na(temp)] <-
          rnorm(n.missing, mean = temp.mean, sd = temp.sd)
        df[[i]] <- temp
      }
    }
    df
  }
  # else if(method=="grr"){
  #   library(DreamAI)
  #   df<-impute.RegImpute(data=as.matrix(df1), fillmethod = "row_mean", maxiter_RegImpute = 10,conv_nrmse = 1e-03)
  # }
  else if (method == "gms") {
    # install.packages('GMSimpute')
    if (!require(GMSimpute)) {
      install.packages(
        "src/R_NAGuideR/GMSimpute_0.0.1.1.tar.gz",
        repos = NULL,
        type = "source"
      )
      
      library(GMSimpute)
    }
    
    df <- GMS.Lasso(df1,
                    nfolds = 3,
                    log.scale = FALSE,
                    TS.Lasso = TRUE)
  }
  else if (method == "msimpute") {
    install_bioconductor("msImpute")
    df <- msImpute(as.matrix(df),
                   method = 'v2')
    df <- as.data.frame(df)
  }
  else if (method == "msimpute_mnar") {
    install_bioconductor("msImpute")
    df <-
      msImpute(as.matrix(df),
               method = 'v2-mnar',
               group = rep(1, dim(df)[2]))
    df <- as.data.frame(df)
  }
  else if (method == "gsimp") {
    options(stringsAsFactors = F)
    # dependencies parly for sourced file
    
    install_bioconductor("impute")
    install_bioconductor("pcaMethods")
    install_rpackage('gmm')
    install_rpackage('imputeLCMD')
    install_rpackage("magrittr")
    install_rpackage("glmnet")
    install_rpackage("abind")
    install_rpackage("foreach")
    install_rpackage("doParallel")
    source('src/R_NAGuideR/GSimp.R')
    
    # wrapper function with data pre-processing
    pre_processing_GS_wrapper <- function(data_raw_log) {
      # samples in rows, features in columns #
      # Initialization #
      data_raw_log_qrilc <- as.data.frame(data_raw_log) %>%
        impute.QRILC() %>% extract2(1)
      # Centralization and scaling #
      data_raw_log_qrilc_sc <-
        scale_recover(data_raw_log_qrilc, method = 'scale')
      # Data after centralization and scaling #
      data_raw_log_qrilc_sc_df <- data_raw_log_qrilc_sc[[1]]
      # Parameters for centralization and scaling (for scaling recovery) #
      data_raw_log_qrilc_sc_df_param <- data_raw_log_qrilc_sc[[2]]
      # NA position #
      NA_pos <- which(is.na(data_raw_log), arr.ind = T)
      # NA introduced to log-scaled-initialized data #
      data_raw_log_sc <- data_raw_log_qrilc_sc_df
      data_raw_log_sc[NA_pos] <- NA
      # Feed initialized and missing data into GSimp imputation #
      result <-
        data_raw_log_sc %>% GS_impute(
          .,
          iters_each = 50,
          iters_all = 10,
          initial = data_raw_log_qrilc_sc_df,
          lo = -Inf,
          hi = 'min',
          n_cores = 1,
          imp_model = 'glmnet_pred'
        )
      data_imp_log_sc <- result$data_imp
      # Data recovery #
      data_imp <- data_imp_log_sc %>%
        scale_recover(., method = 'recover',
                      param_df = data_raw_log_qrilc_sc_df_param) %>%
        extract2(1)
      return(data_imp)
    }
    df <- t(df) # samples in rows, feature in columns
    df <- pre_processing_GS_wrapper(df)
    df <- t(df) # features in rows, samples in columns
    
  }
  else{
    stop(paste("Unspported methods so far: ", method))
  }
  df <- as.data.frame(df)
  df
}
# -

# ## Parameters
#
# Choose one of the available methods. 
# Some methods might fail for your dataset for unknown reasons
# (and the error won't always be easy to understand)
# ```method
# method = 'ZERO'
# method = 'MINIMUM'
# method = 'COLMEDIAN'
# method = 'ROWMEDIAN'
# method = 'KNN_IMPUTE'
# method = 'SEQKNN'
# method = 'BPCA'
# method = 'SVDMETHOD'
# method = 'LLS'
# method = 'MLE'
# mehtod = 'LLS'
# method = 'QRILC'
# method = 'MINDET'
# method = 'MINPROB'
# method = 'IRM'
# method = 'IMPSEQ'
# method = 'IMPSEQROB'
# method = 'MICE-NORM'
# method = 'MICE-CART'
# method = 'RF'
# method = 'PI'
# method = 'GMS'
# method = 'TRKNN',
# method = 'MSIMPUTE'
# method = 'MSIMPUTE_MNAR'
# method = 'GSIMP'
# ```

# + tags=["parameters"] vscode={"languageId": "r"}
train_split = 'runs/example/data/data_wide_sample_cols.csv' # test
folder_experiment = 'runs/example/'
method = 'KNN_IMPUTE'
# -

# ## Dump predictions

# + vscode={"languageId": "r"}
df <-
  utils::read.csv(
    train_split,
    row.names = 1,
    header = TRUE,
    stringsAsFactors = FALSE
  )
df
# -

# - `data.frame` does not allow abritary column names, but only valid column names...
# - tibbles don't support rownames, and the imputation methods rely on normal `data.frame`s.
# Save the header row for later use.

# + vscode={"languageId": "r"}
original_header <- colnames(readr::read_csv(
  train_split,
  n_max = 1,
  col_names = TRUE,
  skip = 0
))
feat_name <- original_header[1]
original_header[1:5]
# -

# Uncomment to test certain methods (only for debugging, as at least one method per package is tested using Github Actions)

# + tags=["hide-input"] vscode={"languageId": "r"}
# to_test <- c(
# 'ZERO',
# 'MINIMUM',
# 'COLMEDIAN',
# 'ROWMEDIAN',
# 'KNN_IMPUTE',
# 'SEQKNN',
# 'BPCA',
# 'SVDMETHOD',
# 'LLS',
# 'MLE',
# 'LLS',
# 'QRILC',
# 'MINDET',
# 'MINPROB',
# 'IRM',
# 'IMPSEQ',
# 'IMPSEQROB',
# 'MICE-NORM',
# 'MICE-CART',
# 'RF',
# 'PI',
# 'GMS', # fails to install on Windows
# 'TRKNN',
# 'MSIMPUTE'
# 'MSIMPUTE_MNAR'
# 'GSIMP'
# )

# for (method in to_test) {
#     print(method)
#     pred <- nafunctions(df, method)
# }
# -

# Impute and save predictions with original feature and column names

# + vscode={"languageId": "r"}
pred <- nafunctions(df, method)
pred <- tibble::as_tibble(cbind(rownames(pred), pred))
names(pred) <- original_header
pred
# -

# Transform predictions to long format

# + vscode={"languageId": "r"}
pred <- reshape2::melt(pred, id.vars = feat_name)
names(pred) <- c(feat_name, 'Sample ID', method)
pred <- pred[reshape2::melt(is.na(df))['value'] == TRUE, ]
pred
# -

# Check dimension of long format dataframe

# + tags=["hide-input"] vscode={"languageId": "r"}
dim(pred)
# -

# Save predictions to disk

# + tags=["hide-input"] vscode={"languageId": "r"}
fname = file.path(folder_experiment,
                  'preds',
                  paste0('pred_all_', toupper(method), '.csv'))
write_csv(pred, path = fname)
fname
