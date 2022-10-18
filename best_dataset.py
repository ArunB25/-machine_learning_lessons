from sklearn import datasets, model_selection, linear_model

#get multiple datasets
cali_house_X, cali_house_y = datasets.fetch_california_housing(return_X_y=True)
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
make_regression_X,make_regression_y = datasets.make_regression()
friedman_X, friedman_y = datasets.make_friedman1()

#list of dataset names
datasets_X = [cali_house_X,diabetes_X,make_regression_X,friedman_X]
datasets_y = [cali_house_y,diabetes_y,make_regression_y,friedman_y]
dataset_names = ['cali_house','diabetes','make_regression','friedman']

for num_dataset in range(0,4):
    X = datasets_X[num_dataset]
    y = datasets_y[num_dataset]
    model = linear_model.LinearRegression() #create instance of the linear regression model
    model.fit(X, y)
    score = model.score(X,y)
    print(dataset_names[num_dataset], "dataset got a score of ", score, "when fitted to the linear regression model")