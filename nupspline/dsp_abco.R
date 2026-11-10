# ABCO competitor (Wu, Schafer & Matteson [10]) via the R package dsp [22],
# called by abco.py.  Run without its outlier component (useAnom = FALSE).
#
#   Rscript dsp_abco.R <in.txt> <out.txt> <D> <obsSV> <nsave> <nburn> <nskip> <seed>
#
# in.txt : one observation per line (standardised by the caller)
# D      : order of the differences of the trend (the paper's m), 1 or 2
# obsSV  : "const" (Scenario 1) or "ASV" (Scenario 2); all other options are
#          dsp_spec / dsp_fit defaults
# out.txt: K rows: posterior mean of the trend, the posterior changepoint
#          probability P(omega_k^2 > exp(threshold) | data) of dsp's predict(),
#          and the posterior means of the evolution variance omega_k^2 of the
#          D-th differences and of the noise variance sigma_k^2 (columns 2 and 3
#          NA for the first D rows)
# stdout : "ELAPSED <seconds>": run time of dsp_fit alone
suppressPackageStartupMessages(library(dsp))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 8) stop("usage: dsp_abco.R in out D obsSV nsave nburn nskip seed")
y     <- scan(args[1], quiet = TRUE)
D     <- as.integer(args[3])
obsSV <- args[4]
nsave <- as.integer(args[5]); nburn <- as.integer(args[6]); nskip <- as.integer(args[7])
seed  <- as.integer(args[8])

spec <- suppressMessages(dsp_spec(family = "gaussian", model = "changepoint",
                                  D = D, useAnom = FALSE, obsSV = obsSV))
t_fit <- system.time({
    fit <- dsp_fit(y, model_spec = spec, nsave = nsave, nburn = nburn, nskip = nskip,
                   computeDIC = FALSE, verbose = FALSE, seed = seed)
})[["elapsed"]]
o <- fit$mcmc_output
cp_prob <- c(rep(NA, D), predict(fit, cp_thres = 0.5, cp_prop = TRUE)$cp_prop)
evol <- colMeans(o$evol_sigma_t2)
evol[seq_len(D)] <- NA                      # dsp's diffuse initial variances, not estimates
out <- cbind(colMeans(o$mu), cp_prob, evol, colMeans(o$obs_sigma_t2))
write.table(format(out, digits = 17), args[2], row.names = FALSE, col.names = FALSE,
            quote = FALSE)
cat(sprintf("ELAPSED %.6f\n", t_fit))
