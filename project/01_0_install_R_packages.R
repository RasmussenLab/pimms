# ---
# jupyter:
#   jupytext:
#     formats: ipynb,R:light
#     text_representation:
#       extension: .R
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: R
#     language: R
#     name: ir
# ---

# + tags=["parameters"] vscode={"languageId": "r"}
methods = 'KNN_IMPUTE,msImpute'

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


nafunctions <- function(method = "zero") {
  method <- tolower(method)
  if (method == "zero") {
  }
  else if (method == "minimum") {
  }
  else if (method == "colmedian") {
    install_rpackage('e1071')
  }
  else if (method == "rowmedian") {
    install_rpackage('e1071')
  }
  else if (method == "knn_impute") {
    install_bioconductor('impute')
  }
  else if (method == "seqknn") {
    if (!require(SeqKnn)) {
      install.packages("src/R_NAGuideR/SeqKnn_1.0.1.tar.gz",
                       repos = NULL,
                       type = "source")
    }
  }
  else if (method == "bpca") {
    install_bioconductor('pcaMethods')
  }
  else if (method == "svdmethod") {
    install_bioconductor('pcaMethods')
  }
  else if (method == "lls") {
    install_bioconductor('pcaMethods')
  }
  else if (method == "mle") {
    install_rpackage('norm')
  }
  else if (method == "qrilc") {
    install_bioconductor("impute")
    install_bioconductor("pcaMethods")
    install_rpackage('gmm')
    install_rpackage('imputeLCMD')
  }
  else if (method == "mindet") {
    install_bioconductor("impute")
    install_bioconductor("pcaMethods")
    install_rpackage('gmm')
    install_rpackage('imputeLCMD')
  }
  else if (method == "minprob") {
    install_bioconductor("impute")
    install_bioconductor("pcaMethods")
    install_rpackage('gmm')
    install_rpackage('imputeLCMD')
  }
  else if (method == "irm") {
    install_rpackage('VIM')
  }
  else if (method == "impseq") {
    install_rpackage('rrcovNA')
  }
  else if (method == "impseqrob") {
    install_rpackage('rrcovNA')
  }
  else if (method == "mice-norm") {
    install_rpackage('mice')
  }
  else if (method == "mice-cart") {
    install_rpackage('mice')
  }
  else if (method == "trknn") {
    source('src/R_NAGuideR/Imput_funcs.r')
  }
  else if (method == "rf") {
    install_rpackage("missForest")
  }
  else if (method == "pi") {
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
      }
  }
  else if (method == "msimpute") {
    install_bioconductor("msImpute")
  }
  else if (method == "msimpute_mnar") {
    install_bioconductor("msImpute")
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
    
    } 
  else{
    stop(paste("Unspported methods so far: ", method))
  }
  df <- as.data.frame(df)
  df
}


for (package in packages_base_R) {
  # Check if the package is already installed
  install_rpackage(pkg = package)
}

if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install(version = "3.20")



# + vscode={"languageId": "r"}
methods = unlist(strsplit(methods, split = ","))
for (package in methods) {
  # Check if the package is already installed, otherwise install
  nafunctions(method = package)
}

