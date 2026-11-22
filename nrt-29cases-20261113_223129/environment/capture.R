# Capture the visible R libraries, native runtime, and required backend imports.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args) == 1L, dir.exists(args[[1]]))
out <- args[[1]]
fields <- c("Repository", "NeedsCompilation", "RemoteType", "RemoteUrl", "RemoteSha")
packages <- as.data.frame(installed.packages(fields=fields), stringsAsFactors=FALSE)
packages <- packages[order(packages$Package, packages$LibPath), ]
write.csv(packages, file.path(out, "r-packages.csv"), row.names=FALSE, na="")
required <- c("survival", "icenReg", "partykit", "ICcforest")
loaded <- vapply(required, requireNamespace, quietly=TRUE, FUN.VALUE=logical(1))
runtime <- data.frame(version=as.character(getRversion()), platform=R.version$platform,
                      blas=unname(extSoftVersion()["BLAS"]),
                      lapack=as.character(La_version()),
                      rng=paste(RNGkind(), collapse=";"), stringsAsFactors=FALSE)
write.csv(runtime, file.path(out, "r-version.csv"), row.names=FALSE)
details <- capture.output({
    print(R.version)
    print(Sys.which(c("R", "Rscript")))
    print(.libPaths())
    print(extSoftVersion())
    print(RNGkind())
    print(sessionInfo())
    print(vapply(getLoadedDLLs(), function(x) x[["path"]], character(1)))
    print(loaded)
})
writeLines(details, file.path(out, "r-runtime.txt"))
if (!all(loaded)) stop(paste("Missing R backends:", paste(required[!loaded], collapse=", ")))
cat("Captured", nrow(packages), "R packages; all required backends loaded.\n")
