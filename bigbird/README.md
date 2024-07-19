# Proteinext

This is the model for Proteinext. The files are as follows:

- train_params2.py is how the model was trained
- test.py takes input (test_data.csv in the data subdirectory) and makes predictions on that data
- testSim.py reads those predictions and calculates the precision, recall, and f1 scores of those predictions.

In order to make your own predictions, please update test_data.csv or change the path in the test file to a similarly-formatted file of data. Then, once your predictions are made, run testSim.py to see the quality of your results.