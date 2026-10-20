# BTF-DHS competitors via the R package dsp [22] (Kowal, Matteson & Ruppert
# [8]; adaptive stochastic volatility: Cho & Matteson [9]), called by btf.py.
#
#   Rscript dsp_btf.R <in.txt> <out.txt> <obsSV> <nsave> <nburn> <nskip> <seed>
#
# in.txt : one observation per line (standardised by the caller, as the dsp
#          documentation recommends)
# out.txt: K rows: posterior means of mu_k, sigma_k^2 and the evolution
#          variance (tau lambda_k)^2
# obsSV  : "const" (BTF-DHS), "SV" (BTF-DHS-SV) or "ASV" (BTF-DHS-ASV); all
#          other options are dsp_spec / dsp_fit defaults
# stdout : "ELAPSED <seconds>": run time of dsp_fit alone (no R start-up,
#          package loading or file I/O)
suppressPackageStartupMessages(library(dsp))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 7) stop("usage: dsp_btf.R in out obsSV nsave nburn nskip seed")
y     <- scan(args[1], quiet = TRUE)
obsSV <- args[3]
nsave <- as.integer(args[4]); nburn <- as.integer(args[5]); nskip <- as.integer(args[6])
seed  <- as.integer(args[7])

spec <- suppressMessages(dsp_spec(family = "gaussian", model = "smoothing",
                                  evol_error = "DHS", D = 2, obsSV = obsSV))
t_fit <- system.time({
    fit <- dsp_fit(y, model_spec = spec, nsave = nsave, nburn = nburn, nskip = nskip,
                   computeDIC = FALSE, verbose = FALSE, seed = seed)
})[["elapsed"]]
o <- fit$mcmc_output
out <- cbind(colMeans(o$mu), colMeans(o$obs_sigma_t2), colMeans(o$evol_sigma_t2))
write.table(format(out, digits = 17), args[2], row.names = FALSE, col.names = FALSE,
            quote = FALSE)
cat(sprintf("ELAPSED %.6f\n", t_fit))
