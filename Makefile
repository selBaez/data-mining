predictions:
	./ensemble_train.py
	./transductor_train.py
	./transductor_predict.py

clean:
	rm -f ./models/*.pkl ./submissions/*.csv ./submissions/*.pkl ./*.log
