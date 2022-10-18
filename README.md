# machine_learning_lessons
Tasks undertaken to learn about machine learning

## Linear Regression
### house_price_predictions.py:
1. Load in the California house pricing data and unpack the features and labels
2. Import a linear regression model from sklearn
3. Fit the model
4. use test values of house features and predict it's price
5. Compute the score of the model on the training data
6. Compute the score of the model on the training data

### best_dataset.py
1. Find 3 sklearn regression datasets
2. Write a loop which fits a linear regression model for each dataset and saves its score
3. Take a quick look at what the score actually represents

The make_regression always achieves a perfect score of 1 as its a curated data set made for linear regression, The California house data set achieves a score of 0.606 and the diabetes dataset with 0.518 showing that this linear regression model would make better prdictions on the california housing data set than it would with the diabetes one.

### best_model.py
1. Train at least 3 different sklearn regression models on a dataset
2. Repeat the experiment for at least 3 different regression datasets
3. For each data set record 
    1. The score
    2. Time taken to fit
    3. Time taken to predict on the whole training set
    4. The size of the model occupied in memory (look up how to do that)

This script compares a several individual features from the califronia housing dataset with 3 different regression models. For each feature a subplot is created with 3 plots one for each regression model, displaying plots of the training data with the models hypothesis and the predictions of the test data with the real values. The models score, time taken to fit and predict as well as the models size is also shown in the titles of each subplot. This script shows the performance of the 3 selected models for each feature, it also shows what features are more suitable to predict house prices.

### linear_regression_from_scratch.py 
1. Load in the California dataset for testing and debugging
2. Create a class called LinearRegression
3. Randomly initialise two attributes for the weight and bias
4. Create a .predict method which takes in some data and returns a linear prediction
5. Create a .fit_analytical method which computes the analytical solution
6. Create a .fit method which uses gradient descent to optimise the model
7. Compare created model to SKlearns linear regression model