from ctypes import sizeof
from turtle import title
from sklearn import datasets , model_selection, linear_model,metrics
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class linear_regression():
    def __init__(self, n_features : int):
        '''
        creates a model with random wight and bias values
        '''
        np.random.seed(10)
        self.weight = np.random.randn(n_features, 1) ## randomly initialise weight
        self.bias = np.random.randn(1) ## randomly initialise bias

    def predict(self,X):
        '''
        Get a prediction from a set of features based on the weights and bias of the model
        '''
        ypred = np.dot(X, self.weight) + self.bias
        return ypred # return prediction

    def fit_analytical(self, X_train, y_train):
        '''
        Analytical solution to minimise the mean squared error 
        Not advised for large multi feature datasets as finding inverse of a matrix can be computationally heavy 
        '''
        X_with_bias = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
        optimal_w = np.matmul(
        np.linalg.inv(np.matmul(X_with_bias.T, X_with_bias)),
        np.matmul(X_with_bias.T, y_train),)
        weights = optimal_w[1:]
        bias = optimal_w[0]
        return weights, bias

    def set_params(self, weights, bias):
        '''
        set the parameters manually
        '''
        self.weight = weights
        self.bias = bias

    def fit(self,X,y,learningrate = 0.001,iterations = 2000):
        """ Find the multivarite regression model for the data set
        Parameters:
        X: independent variables matrix
        y: dependent variables matrix
        Return value: the final theta vector and the plot of cost function
        Credit to https://medium.com/@IwriteDSblog/gradient-descent-for-multivariable-regression-in-python-d430eb5d2cd8
        """
        def generateXvector(X):
            """ Taking the original independent variables matrix and add a row of 1 which corresponds to x_0
                Parameters:
                X:  independent variables matrix
                Return value: the matrix that contains all the values in the dataset, not include the outcomes variables. 
            """
            vectorX = np.c_[np.ones((len(X), 1)), X]
            return vectorX
        def theta_init(X):
            """ Generate an initial value of vector θ from the original independent variables matrix
                Parameters:
                X:  independent variables matrix
                Return value: a vector of theta filled with initial guess
            """
            theta = np.random.randn(len(X[0])+1, 1)
            return theta
        
        y_new = np.reshape(y, (len(y), 1))   
        cost_lst = []
        vectorX = generateXvector(X)
        theta = theta_init(X)
        m = len(X)
        for i in range(iterations):
            gradients = 2/m * vectorX.T.dot(vectorX.dot(theta) - y_new)
            theta = theta - learningrate * gradients
            y_pred = vectorX.dot(theta)
            cost_value = 1/(2*len(y))*((y_pred - y)**2) 
            #Calculate the loss for each training instance
            total = 0
            for i in range(len(y)):
                total += cost_value[i][0] 
                #Calculate the cost function for each iteration
            cost_lst.append(total)
        bias = theta[0]
        weights = theta[1:]
        return weights,bias

if __name__ == "__main__":
    X_all, y_all = datasets.fetch_california_housing(return_X_y=True)
    # X = X_all[0:1000,:] https://medium.com/@IwriteDSblog/gradient-descent-for-multivariable-regression-in-python-d430eb5d2cd8
    X = X_all[0:1000,0] 
    X = np.reshape(X, (-1, 1))
    y = y_all[0:1000]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)
    lin_reg_model = linear_regression(X.shape[1])
    weights,bias = lin_reg_model.fit(X_train,y_train)
    print(f"Scratch models coefs:{weights} | Intercept:{bias}")
    lin_reg_model.set_params(weights,bias)
    scratch_mse = round(metrics.mean_squared_error(y_test, lin_reg_model.predict(X_test)),5)
    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.scatter(X_train,y_train ,color='g', label='Training Data',zorder=0)
    plt.plot(X_train, lin_reg_model.predict(X_train),color='b',label='Hypothesis',zorder=1)
    plt.scatter(X_test, lin_reg_model.predict(X_test),color='r',label='Predicted Test Data',zorder=2)
    plt.scatter(X_test, y_test,color='k',label='Real Test Data', marker="2",zorder=3)
    plt.ylabel('House Value')
    plt.legend()
    plt.title(f'Made from scratch Linear Regression - MSE {scratch_mse}')

    sklearn_model = linear_model.LinearRegression().fit(X_train, y_train) #create instance of the linear regression model
    print(f"SKlearn coefs:{sklearn_model.coef_} | Intercept:{sklearn_model.intercept_}")
    sklearn_mse = round(metrics.mean_squared_error(y_test, sklearn_model.predict(X_test)),5)
    plt.subplot(2, 1, 2)
    plt.scatter(X_train,y_train ,color='g', label='Training Data',zorder=0)
    plt.plot(X_train, sklearn_model.predict(X_train),color='b',label='Hypothesis',zorder=1)
    plt.scatter(X_test, sklearn_model.predict(X_test),color='r',label='Predicted Test Data',zorder=2)
    plt.scatter(X_test, y_test,color='k',label='Real Test Data', marker="2",zorder=3)
    plt.ylabel('House Value')
    plt.legend()
    plt.title(f'SKlearn Linear Regression - MSE {scratch_mse}')
    
    plt.show()