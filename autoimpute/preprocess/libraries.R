list.of.packages <- c("R.matlab", "Matrix", "ggplot2", "Rtsne", "svd", "dplyr", "plyr",
                      "data.table", "mclust", "flexclust", "reshape2",
                      "irlba","dynamicTreeCut", "RColorBrewer","GenSA")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
print(new.packages)
# install.packages(new.packages)

# ----------------------------
# load relevant libraries
# ----------------------------
print("library(R.matlab)")
library(R.matlab)
print("library(Matrix)")
library(Matrix)
print("library(ggplot2)")
library(ggplot2)
print("library(Rtsne)")
library(Rtsne)
library(svd)
library(plyr)
library(dplyr)
library(data.table)
library(mclust)
library(flexclust)
library(reshape2)
library(irlba)
library(dynamicTreeCut)
library(RColorBrewer)
library(GenSA)
library(gplots)
# -------------------------------------
