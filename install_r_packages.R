# Install the R package dsp (Kowal et al.), needed for the BTF competitors.
# The paper uses dsp 1.6.0 (CRAN).
#
#   Rscript install_r_packages.R
#
# Installs into the personal R library.  On Linux every dependency is compiled
# from source; dsp needs RcppZiggurat, which needs the GNU Scientific Library
# (e.g. apt install libgsl-dev).
lib <- Sys.getenv("R_LIBS_USER")
dir.create(lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(lib, .libPaths()))
if (!requireNamespace("dsp", quietly = TRUE)) {
    install.packages("dsp", lib = lib, repos = "https://cloud.r-project.org")
}
suppressPackageStartupMessages(library(dsp))
cat("dsp", as.character(packageVersion("dsp")), "installed\n")
