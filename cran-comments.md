## R CMD check results

0 errors | 0 warnings | 1 note

## Test environments

* local Windows 11, R 4.5.1
* win-builder R-devel, R 4.6.0

## Initial submission

This is the initial CRAN submission of rkaf.

The only NOTE is the standard "New submission" note from CRAN incoming checks.
CRAN incoming also flags "KAF" and "multiclass" as possibly misspelled words;
both are intended: KAF abbreviates Kolmogorov-Arnold Fourier, and multiclass is
a standard machine-learning term.

## Notes

rkaf uses the torch package as its neural network backend. Tests and vignette
code that require the torch backend are skipped when the LibTorch/Lantern runtime
is unavailable in the check environment.
