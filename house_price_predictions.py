from sklearn import datasets, model_selection, linear_model, metrics

# link to details for the dataset https://github.com/scikit-learn/scikit-learn/blob/36958fb24/sklearn/datasets/_california_housing.py#L53
X, y = datasets.fetch_california_housing(return_X_y=True)

#divide the dataset into groups to train the model and test it. 30% of the dataset used for testing. remaining 70% used for training
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)


model = linear_model.LinearRegression() #create instance of the linear regression model
model.fit(X_train, y_train) #Train the model
y_pred = model.predict(X_test)  #make predictions for all the testing values
print("Model's Score:", model.score(X_train, y_train))
mse = metrics.mean_squared_error(y_test, y_pred) #calculate the mean squared error
print("Mean Squared Error:", mse)