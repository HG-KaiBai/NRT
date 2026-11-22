# Restore pinned local source archives into a NEW user library; never upgrade dependencies.
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 2L) stop("Usage: Rscript --vanilla environment/restore-r.R r-packages.csv archive-directory")
expected <- read.csv(args[[1]], stringsAsFactors=FALSE, na.strings="")
if (anyDuplicated(expected$Package)) stop("Resolve duplicate package names in the snapshot first")
if (as.character(getRversion()) != "4.4.0") stop("The pinned R runtime is 4.4.0")
archives <- normalizePath(args[[2]], mustWork=TRUE)
library <- Sys.getenv("R_LIBS_USER")
if (!nzchar(library) || grepl(.Platform$path.sep, library, fixed=TRUE)) {
    stop("R_LIBS_USER must name one empty destination directory")
}
dir.create(library, recursive=TRUE, showWarnings=FALSE)
library <- normalizePath(library, mustWork=TRUE)
if (length(list.files(library, all.files=TRUE, no..=TRUE))) {
    stop("R_LIBS_USER must be empty; existing libraries are never modified")
}
.libPaths(c(library, .Library.site, .Library))
# The source snapshot distinguishes the user library from CVMFS system libraries.
user <- !grepl("/cvmfs/", expected$LibPath, fixed=TRUE)
system <- expected[!user, ]
actual <- as.data.frame(installed.packages(lib.loc=c(.Library.site, .Library)), stringsAsFactors=FALSE)
for (i in seq_len(nrow(system))) {
    row <- system[i, ]
    found <- actual[actual$Package == row$Package, ]
    if (nrow(found) != 1L || found$Version != row$Version || found$Built != row$Built) {
        stop(paste("System package/build mismatch:", row$Package, "requires", row$Version, "built with", row$Built))
    }
}
pending <- expected[user, ]
paths <- file.path(archives, paste0(pending$Package, "_", pending$Version, ".tar.gz"))
if (any(!file.exists(paths))) stop(paste("Missing pinned archives:", paste(paths[!file.exists(paths)], collapse=", ")))
dependencies <- function(row) {
    text <- unlist(row[c("Depends", "Imports", "LinkingTo")], use.names=FALSE)
    text <- text[!is.na(text)]
    names <- trimws(gsub("\\s*\\([^)]*\\)", "", unlist(strsplit(text, ","))))
    setdiff(names[nzchar(names)], "R")
}
while (nrow(pending)) {
    ready <- which(vapply(seq_len(nrow(pending)), function(i) {
        !any(dependencies(pending[i, ]) %in% pending$Package)
    }, logical(1)))
    if (!length(ready)) stop("Cannot resolve pinned Depends/Imports/LinkingTo order")
    for (i in ready) {
        row <- pending[i, ]
        source <- file.path(archives, paste0(row$Package, "_", row$Version, ".tar.gz"))
        install.packages(source, repos=NULL, type="source", lib=library, dependencies=FALSE)
        got <- installed.packages(lib.loc=library)
        if (!row$Package %in% rownames(got) || got[row$Package, "Version"] != row$Version) {
            stop(paste("Pinned package installation failed:", row$Package))
        }
    }
    pending <- pending[-ready, , drop=FALSE]
}
cat("Restored pinned R source packages. Run environment/verify.py to compare the full runtime.\n")
