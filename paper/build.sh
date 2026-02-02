#!/bin/sh
pdflatex main.tex
bibtex main
pdflatex main.tex && rm *.aux *.out *.toc *.lof *.lot *.bbl *.blg *.fls *.fdb_latexmk *.bbl *synctex.gz