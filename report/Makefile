filename=report

pdf:
	pdflatex ${filename}.tex
	bibtex ${filename}
	pdflatex ${filename}.tex
	pdflatex ${filename}.tex

read:
	mupdf ${filename}.pdf &

clean:
	rm -f ${filename}{.ps,.pdf,.log,.aux,.out,.dvi,.bbl,.blg,-blx.bib,.run.xml}
