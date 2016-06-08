report_dir=report
report_filename=report


all:
	@echo 'No default target'


pdf:
	pdflatex -output-directory ${report_dir} ${report_filename}.tex
	@cd ${report_dir} && bibtex ${report_filename}
	pdflatex -output-directory ${report_dir} ${report_filename}.tex
	pdflatex -output-directory ${report_dir} ${report_filename}.tex

read:
	@cd ${report_dir} && mupdf ${report_filename}.pdf &

clean-report:
	@cd ${report_dir} && \
		rm -f ${report_filename}{.ps,.pdf,.log,.aux,.out,.dvi,.bbl,.blg,-blx.bib,.run.xml}


predictions: train-ensemble train-transductor
	./transductor_predict.py

train-ensemble: clean-1
	./ensemble_train.py

train-transductor: clean-2
	./transductor_train.py

plots:
	./plots.py -s

clean-1:
	rm -f ./models/1st*.pkl ./submissions/1st*.{csv,pkl}

clean-2:
	rm -f ./models/2nd*.pkl ./submissions/2nd*.{csv,pkl}

clean: clean-1 clean-2
