from sklearn import datasets, linear_model, ensemble, model_selection
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
import sys

# link to details for the dataset https://github.com/scikit-learn/scikit-learn/blob/36958fb24/sklearn/datasets/_california_housing.py#L53
X_all, y_all = datasets.fetch_california_housing(return_X_y=True)
column_data = ['average income','housing average age', 'average rooms', 'average bedrooms', 'population','average occupation', 'latitude','longitude']

for i in range(0,1):
    #create the X and Y training and testing datasets
    X = X_all[0:1000,i] 
    X = np.reshape(X, (-1, 1))
    y = y_all[0:1000]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)

    start_time = time.time()
    Lin_model = linear_model.LinearRegression().fit(X_train, y_train) #create instance of the linear regression model
    time_to_fit = round(time.time() - start_time, 3)
    Lin_score = round(Lin_model.score(X_train,y_train),3)
    model_file = pickle.dumps(Lin_model)
    size_of_model = sys.getsizeof(model_file)
    start_time = time.time()
    y_predict = Lin_model.predict(X_test)
    time_to_predict = round(time.time() - start_time, 5)
    title = f"Linear Regression Score: {Lin_score} | Time to fit {time_to_fit}s | Time to predict {time_to_predict}s | Model Size: {size_of_model}bytes"
    plt.figure(i+1)
    plt.subplot(3, 1, 1)
    plt.scatter(X_train, y_train,color='g', label='Training Data',zorder=0)
    plt.plot(X_train, Lin_model.predict(X_train),color='b',label='Hypothesis',zorder=1)
    plt.scatter(X_test, y_predict,color='r',label='Predicted Test Data',zorder=2)
    plt.scatter(X_test, y_test,color='k',label='Real Test Data', marker="2",zorder=3)
    plt.ylabel('House Value')
    plt.legend()
    plt.title(title)

    start_time = time.time()
    gbr_model =ensemble.GradientBoostingRegressor(random_state=0).fit(X_train, y_train)
    time_to_fit = round(time.time() - start_time, 3)
    gbr_score = round(gbr_model.score(X_train,y_train),3)
    model_file = pickle.dumps(gbr_model)
    size_of_model = sys.getsizeof(model_file)
    start_time = time.time()
    y_predict = gbr_model.predict(X_test)
    time_to_predict = round(time.time() - start_time, 5)
    title = f"Gradient Boosting Regressor Score: {gbr_score} | Time to fit {time_to_fit}s | Time to predict {time_to_predict}s | Model Size: {size_of_model}bytes"
    plt.subplot(3, 1, 2)
    plt.scatter(X_train, y_train,color='g', label='Training Data',zorder=0)
    plt.plot(X_train, gbr_model.predict(X_train),color='b',label='Hypothesis',zorder=1)
    plt.scatter(X_test, y_predict,color='r',label='Predicted Test Data',zorder=2)
    plt.scatter(X_test, y_test,color='k',label='Real Test Data', marker="2",zorder=3)
    plt.ylabel('House Value')
    plt.legend()
    plt.title(title)

    start_time = time.time()
    ENet_model = linear_model.ElasticNet(random_state=0).fit(X_train, y_train)
    time_to_fit = round(time.time() - start_time, 3)
    ENet_score = round(ENet_model.score(X_train,y_train),3)
    model_file = pickle.dumps(ENet_model)
    size_of_model = sys.getsizeof(model_file)
    start_time = time.time()
    y_predict = ENet_model.predict(X_test)
    time_to_predict = round(time.time() - start_time, 5)
    title = f"Elastic Net Regression Score: {ENet_score} | Time to fit {time_to_fit}s | Time to predict {time_to_predict}s | Model Size: {size_of_model}bytes"
    plt.subplot(3, 1, 3)
    plt.scatter(X_train, y_train,color='g', label='Training Data',zorder=0)
    plt.plot(X_train, ENet_model.predict(X_train),color='b',label='Hypothesis',zorder=1)
    plt.scatter(X_test, y_predict,color='r',label='Predicted Test Data',zorder=2)
    plt.scatter(X_test, y_test,color='k',label='Real Test Data', marker="2",zorder=3)
    plt.xlabel(column_data[i])
    plt.ylabel('House Value')
    plt.title(title)
    plt.legend()
    plt.suptitle(f"Best Model Comparison for california housing value based on {column_data[i]}")

plt.show()