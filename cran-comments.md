## R CMD check results

0 errors | 0 warnings | 0 notes

## Test environments

* local Windows 11, R 4.5.1
* win-builder R-release: pending / passed
* win-builder R-devel: pending / passed

## Initial submission

This is the initial CRAN submission of rkaf.

## Notes

## Notes

rkaf uses the torch package as its neural network backend. Tests and vignette
code that require the torch backend are skipped when the LibTorch/Lantern
runtime is unavailable in the check environment.
