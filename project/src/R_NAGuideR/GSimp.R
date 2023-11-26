# formatted from https://github.com/WandeRum/GSimp/blob/9f661e5ebf991b160ccedb7728c5bcd825dc963b/GSimp.R
# require(missForest)
require(imputeLCMD)
require(magrittr)
require(glmnet)
require(abind)
require(foreach)
require(doParallel)
# require(MASS)


## Source ##
# source('MVI_global.R')
# source('Prediction_funcs.R')

## Draw n samples from a truncated normal distribution N(mu, std^2|[lo, hi]) ##
rnorm_trunc <- function (n,
                         mu,
                         std,
                         lo = -Inf,
                         hi = Inf) {
  p_lo <- pnorm(lo, mu, std)
  p_hi <- pnorm(hi, mu, std)
  p_hi[p_hi < .01] <- .01
  u <- runif(n, p_lo, p_hi)
  return(qnorm(u, mu, std))
}

## Initialize the missing data ##
## lsym will draw samples from the right tail of the distribution and transformed to the left tail
miss_init <-
  function(miss_data,
           method = c('lsym', 'qrilc', 'rsym')[1]) {
    init_data <- miss_data
    if (method == 'lsym') {
      for (i in 1:ncol(init_data)) {
        col_temp <- init_data[, i]
        na_idx <- which(is.na(col_temp))
        prop <- mean(is.na(col_temp))
        min_temp <- min(col_temp, na.rm = T)
        col_temp[na_idx] <- min_temp - 1
        med_temp <- median(col_temp)
        col_temp[na_idx] <-
          med_temp - (sample(col_temp[col_temp >= quantile(col_temp, 1 - prop)], length(na_idx), replace =
                               T) - med_temp)
        init_data[, i] <- col_temp
      }
    }
    if (method == 'rsym') {
      for (i in 1:ncol(init_data)) {
        col_temp <- init_data[, i]
        na_idx <- which(is.na(col_temp))
        prop <- mean(is.na(col_temp))
        max_temp <- max(col_temp, na.rm = T)
        col_temp[na_idx] <- max_temp + 1
        med_temp <- median(col_temp)
        col_temp[na_idx] <-
          med_temp + (med_temp - sample(col_temp[col_temp <= quantile(col_temp, prop)], length(na_idx), replace =
                                          T))
        init_data[, i] <- col_temp
      }
    }
    if (method == 'qrilc') {
      init_data <- impute.QRILC(miss_data)[[1]]
    }
    return(init_data)
  }

## Single missing variable imputation based on Gibbs sampler ##
single_impute_iters <-
  function(x,
           y,
           y_miss,
           y_real = NULL,
           imp_model = 'glmnet_pred',
           lo = -Inf,
           hi = Inf,
           iters_each = 100,
           gibbs = c()) {
    y_res <- y
    x <- as.matrix(x)
    na_idx <- which(is.na(y_miss))
    imp_model_func <- getFunction(imp_model)
    nrmse_vec <- c()
    gibbs_res <- array(NA, dim = c(3, length(gibbs), iters_each))
    dimnames(gibbs_res) <- list(c('std', 'yhat', 'yres'), NULL, NULL)
    
    for (i in 1:iters_each) {
      y_hat <- imp_model_func(x, y_res)
      std <- sqrt(sum((y_hat[na_idx] - y_res[na_idx]) ^ 2) / length(na_idx))
      y_res[na_idx] <-
        rnorm_trunc(length(na_idx), y_hat[na_idx], std, lo, hi)
      if (length(gibbs) > 0) {
        gibbs_res[1, , i] <- std
        gibbs_res[2, , i] <- y_hat[gibbs]
        gibbs_res[3, , i] <- y_res[gibbs]
      }
      ## The following code is for prediction function testing when y_real availabe ##
      if (!is.null(y_real)) {
        Sys.sleep(.5)
        par(mfrow = c(2, 2))
        nrmse_vec <- c(nrmse_vec, nrmse(y_res, y_miss, y_real))
        plot(y_real ~ y_res)
        plot(y_real ~ y_hat)
        plot(y_hat ~ y_res)
        plot(nrmse_vec)
      }
    }
    return(list(y_imp = y_res, gibbs_res = gibbs_res))
  }


## Multiple missing variables imputation ##
## iters_each=number (100); vector of numbers, e.g. rep(100, 20) while iters_all=20
## lo/hi=numer; vector; functions like min/max/median/mean...
## initial=character ('qrilc'/'lysm'); initialized data maatrix
## n_cores=1 is sequentially (non-parallel) computing
multi_impute <-
  function(data_miss,
           iters_each = 100,
           iters_all = 20,
           initial = 'qrilc',
           lo = -Inf,
           hi = 'min',
           n_cores = 1,
           imp_model = 'glmnet_pred',
           gibbs = data.frame(row = integer(), col = integer())) {
    ## Convert to data.frame ##
    data_miss %<>% data.frame()
    
    ## Make vector for iters_each ##
    if (length(iters_each) == 1) {
      iters_each <- rep(iters_each, iters_all)
    } else if (length(iters_each) == iters_all) {
      iters_each <- iters_each
    } else {
      stop('improper argument: iters_each')
    }
    
    
    ## Missing count in each column ##
    miss_count <- data_miss %>% apply(., 2, function(x)
      sum(is.na(x)))
    ## Index of missing variables, sorted (increasing) by the number of missings
    miss_col_idx <-
      order(miss_count, decreasing = T) %>% extract(1:sum(miss_count != 0)) %>% rev()
    
    if (!all(gibbs$col %in% miss_col_idx)) {
      stop('improper argument: gibbs')
    }
    gibbs_sort <- gibbs
    if (nrow(gibbs_sort) > 0) {
      gibbs_sort$order <- c(1:nrow(gibbs_sort))
      gibbs_sort <- gibbs_sort[order(gibbs_sort$row),]
      gibbs_sort <-
        gibbs_sort[order(match(gibbs_sort$col, miss_col_idx)),]
    } else {
      gibbs_sort$order <- integer()
    }
    
    ## Make vectors for lo and hi ##
    if (length(lo) > 1) {
      if (length(lo) != ncol(data_miss)) {
        stop('Length of lo should equal to one or the number of variables')
      }
      else {
        lo_vec <- lo
      }
    } else if (is.numeric(lo)) {
      lo_vec <- rep(lo, ncol(data_miss))
    } else if (is.character(lo)) {
      lo_fun <- getFunction(lo)
      lo_vec <-
        apply(data_miss, 2, function(x)
          x %>% na.omit %>% lo_fun)
    }
    
    if (length(hi) > 1) {
      if (length(hi) != ncol(data_miss)) {
        stop('Length of hi should equal to one or the number of variables')
      }
      else {
        hi_vec <- hi
      }
    } else if (is.numeric(hi)) {
      hi_vec <- rep(hi, ncol(data_miss))
    } else if (is.character(hi)) {
      hi_fun <- getFunction(hi)
      hi_vec <-
        apply(data_miss, 2, function(x)
          x %>% na.omit %>% hi_fun)
    }
    
    # Check whether lo is lower than hi
    if (!all(lo_vec < hi_vec)) {
      stop('lo should be lower than hi')
    }
    
    ## Initialization using build-in method or input initial matrix ##
    if (is.character(initial)) {
      data_init <- miss_init(data_miss, method = initial)
    } else if (is.data.frame(initial) &
               identical(data_miss[!is.na(data_miss)], initial[!is.na(data_miss)])) {
      data_init <- initial
    } else {
      stop('improper argument: initial')
    }
    
    data_imp <- data_init
    gibbs_res_final <- array(NA, dim = c(3, nrow(gibbs), 0))
    
    ## Iterations for the whole data matrix ##
    for (i in 1:iters_all) {
      cat('Iteration', i, 'start...')
      
      ## Parallel computing ##
      if (n_cores > 1) {
        cat(paste0('Parallel computing (n_cores=', n_cores, ')...'))
        ## Parallel on missing variables
        cl <- makeCluster(n_cores)
        registerDoParallel(cl)
        core_res <-
          foreach (
            k = miss_col_idx,
            .combine = 'cbind_abind',
            .export = c('single_impute_iters', 'rnorm_trunc'),
            .packages = c('magrittr')
          ) %dopar% {
            source('Prediction_funcs.R')
            gibbs_sort_temp <- gibbs_sort[gibbs_sort$col == k,]
            y_imp_res <-
              single_impute_iters(
                data_imp[,-k],
                data_imp[, k],
                data_miss[, k],
                imp_model = imp_model,
                lo = lo_vec[k],
                hi = hi_vec[k],
                iters_each = iters_each[i],
                gibbs = gibbs_sort_temp$row
              )
            y_imp_df <- y_imp_res$y_imp %>% data.frame
            colnames(y_imp_df) <- colnames(data_miss)[k]
            gibbs_res <- y_imp_res$gibbs_res
            list(y_imp = y_imp_df, gibbs_res = gibbs_res)
          }
        stopCluster(cl)
        y_imp_df <- core_res$y_imp
        gibbs_res_final <-
          abind(gibbs_res_final, core_res$gibbs_res, along = 3)
        miss_col_idx_match <-
          match(colnames(y_imp_df), colnames(data_miss))
        data_imp[, miss_col_idx_match] <- y_imp_df
      } else {
        ## Sequential computing ##
        gibbs_res_j <- array(NA, dim = c(3, 0, iters_each[i]))
        for (j in miss_col_idx) {
          gibbs_sort_temp <- gibbs_sort[gibbs_sort$col == j,]
          y_miss <- data_miss[, j]
          y_imp_res <-
            single_impute_iters(
              data_imp[,-j],
              data_imp[, j],
              y_miss,
              imp_model = imp_model,
              lo = lo_vec[j],
              hi = hi_vec[j],
              iters_each = iters_each[i],
              gibbs = gibbs_sort_temp$row
            )
          y_imp <- y_imp_res$y_imp
          gibbs_res_j <-
            abind(gibbs_res_j, y_imp_res$gibbs_res, along = 2)
          data_imp[is.na(y_miss), j] <- y_imp[is.na(y_miss)]
        }
        gibbs_res_final <-
          abind(gibbs_res_final, gibbs_res_j, along = 3)
      }
      cat('end!\n')
    }
    gibbs_res_final_reorder <- gibbs_res_final[, gibbs_sort$order,]
    return(list(data_imp = data_imp, gibbs_res = gibbs_res_final_reorder))
  }


# GS_impute ---------------------------------------------------------------
GS_impute <- multi_impute

# ------------------------------------------------------------------------------

# GSimp: MVI_global.R
# https://github.com/WandeRum/GSimp/blob/9f661e5ebf991b160ccedb7728c5bcd825dc963b/MVI_global.R#L41-L61

# Scale and recover -------------------------------------------------------
scale_recover <- function(data,
                          method = 'scale',
                          param_df = NULL) {
  results <- list()
  data_res <- data
  if (!is.null(param_df)) {
    if (method == 'scale') {
      data_res[] <- scale(data, center = param_df$mean, scale = param_df$std)
    } else if (method == 'recover') {
      data_res[] <- t(t(data) * param_df$std + param_df$mean)
    }
  } else {
    if (method == 'scale') {
      param_df <-
        data.frame(mean = sapply(data, function(x)
          mean(x, na.rm = T)),
          std = sapply(data, function(x)
            sd(x, na.rm = T)))
      data_res[] <-
        scale(data, center = param_df$mean, scale = param_df$std)
    } else {
      stop('no param_df found for recover...')
    }
  }
  results[[1]] <- data_res
  results[[2]] <- param_df
  return(results)
}


# ------------------------------------------------------------------------------
# https://github.com/WandeRum/GSimp/blob/9f661e5ebf991b160ccedb7728c5bcd825dc963b/Prediction_funcs.R#L27C1-L32C2
# Prediction_funcs.R

glmnet_pred <- function(x, y, alpha = .5, lambda = .01) {
  x_mat <- as.matrix(x)
  model <- glmnet(
    x = x_mat,
    y = y,
    alpha = alpha,
    lambda = lambda
  )
  y_hat <- predict(model, newx = x_mat)[, 1]
  return(y_hat)
}
