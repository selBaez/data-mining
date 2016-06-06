predictions:
	./ensemble_train.py
	./transductor_train.py
	./transductor_predict.py

plots:
	./plots.py -s

clean-1:
	rm -f ./models/1st*.pkl ./submissions/1st*.{csv,pkl} ./stats/ensemble*.{log,pkl}

clean-2:
	rm -f ./models/2nd*.pkl ./submissions/2nd*.{csv,pkl} ./stats/transductor*.{log,pkl}

clean: clean-1 clean-2
