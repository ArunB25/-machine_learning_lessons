from cmath import nan
from ctypes import sizeof
from random import uniform
from turtle import color, title
from sklearn import datasets , model_selection, linear_model,metrics
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time

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

    def score(self,X,y_true):
        y_pred = self.predict(X)
        u  = ((y_true - y_pred)** 2).sum()
        v = ((y_true - y_true.mean()) ** 2).sum()
        return (1 - (u/v))

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

    def get_params(self,deep=False):
        return{'Weight':self.weight,'Bias':self.bias}

    def fit(self,X,y,learningrate = 0.01,iterations = 4, plot = False, batch_size = 256):
        """ Find the multivarite regression model for the data set
        Parameters:
        X: independent variables matrix
        y: dependent variables matrix
        learningrate: factor to reduce change in gradient to avoid deviation, typically between 0 and 1
        iterations: number of times the entire dataset is evaluated
        plot: display inreal time the gradient decent, only works with one parameter.
        batch_size: for mini batch processing to improve efficiency, typically a factor of 32 (32,64,128,256,...)
        Return value: the final weight and bias values
        Credit to https://medium.com/@IwriteDSblog/gradient-descent-for-multivariable-regression-in-python-d430eb5d2cd8 & https://www.geeksforgeeks.org/ml-mini-batch-gradient-descent-with-python/
        """
        def create_mini_batchs(X,y,batch_size):
            '''
            reduces size of full dataset to batch size
            returns mini_batch containing all of the data broken up into batches
            '''
            mini_batches = []
            data = np.hstack((X, y))
            np.random.shuffle(data)
            n_minibatches = data.shape[0] // batch_size
            i = 0
            for i in range(n_minibatches + 1):
                mini_batch = data[i * batch_size:(i + 1)*batch_size, :]
                X_mini = mini_batch[:, :-1]
                Y_mini = mini_batch[:, -1].reshape((-1, 1))
                mini_batches.append((X_mini, Y_mini))
            if data.shape[0] % batch_size != 0:
                mini_batch = data[i * batch_size:data.shape[0]]
                X_mini = mini_batch[:, :-1]
                Y_mini = mini_batch[:, -1].reshape((-1, 1))
                mini_batches.append((X_mini, Y_mini))
            return mini_batches

        def generateXvector(X):
            """ Taking the original independent variables matrix and add a row of 1 which corresponds to x_0
                Parameters:
                Return value: the matrix that contains all the values in the dataset, not include the outcomes variables. 
            """
            vectorX = np.c_[np.ones((len(X), 1)), X]
            return vectorX

        theta = np.random.randn(len(X[0])+1, 1)
        m = len(X) #length of dataset
        if plot == True: #set up plot
            plt.ion()
            fig, ax = plt.subplots(figsize=(10, 8))
            hypothesis_line, = ax.plot(0, 0,color = 'k')
            training_data = ax.scatter(X,y,color='g', label='mini batch data',zorder=1)
            plt.scatter(X,y,color='r', label='Training Data',zorder=0)
            plt.suptitle("Gradient Decent")
            plt.legend()
        for i in range(iterations):
            mini_batches = create_mini_batchs(X, y, batch_size)
            for mini_batch in mini_batches:
                X_mini, y_mini = mini_batch  
                mini_vectorX = generateXvector(X_mini) #create the parameter vector
                gradients = 2/m * mini_vectorX.T.dot(mini_vectorX.dot(theta) - y_mini) #diferentiate loss with respect to theta(weights)
                theta = theta - learningrate * gradients #move gradients in opposite direction to the increase of loss multiplied be the learning rate factor
                if plot == True:
                    hypothesis_line.set_xdata(X) #update x and y values and draw new line
                    y_pred = X.dot(theta[1:])
                    hypothesis_line.set_ydata(y_pred)
                    training_data.set_offsets(np.c_[X_mini,y_mini])
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    plt.title(f"Itteration {i} of {iterations}")

        y_pred = X.dot(theta[1:]) #calculate predictions
        loss = 1/m *sum((y_pred - y)**2)  #calculate the loss from predictions made
        self.bias = theta[0]
        self.weight = theta[1:]
        return self

if __name__ == "__main__":
    plot_on = False #display live plot of gradient decent and results of the 2 models 
    X, y = datasets.fetch_california_housing(return_X_y=True)
    X = X[:,0:1] ##reduce parameters to 1
    y = y.reshape((-1, 1)) #convert to column vector
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)
    

    lin_reg_model = linear_regression(X.shape[1])
    start_time = time.time()
    lin_reg_model.fit(X_train,y_train,plot=plot_on)
    time_to_fit = round(time.time() - start_time, 3)
    scratch_mse = round(metrics.mean_squared_error(y_test, lin_reg_model.predict(X_test)),5)
    print(f"Scratch MSE:{scratch_mse} | Time to fit {time_to_fit}s | Scratch models coefs:{lin_reg_model.weight} | Intercept:{lin_reg_model.bias}")
    if plot_on:
        plt.figure
        plt.subplot(2, 1, 1)
        plt.scatter(X_train,y_train ,color='g', label='Training Data',zorder=0)
        plt.plot(X_train, lin_reg_model.predict(X_train),color='b',label='Hypothesis',zorder=1)
        plt.scatter(X_test, lin_reg_model.predict(X_test),color='r',label='Predicted Test Data',zorder=2)
        plt.scatter(X_test, y_test,color='k',label='Real Test Data', marker="2",zorder=3)
        plt.ylabel('House Value')
        plt.legend()
        plt.title(f'Made from scratch Linear Regression - MSE {scratch_mse} | Time to fit {time_to_fit}s')

    start_time = time.time()
    sklearn_model = linear_model.LinearRegression().fit(X_train, y_train) #create instance of the linear regression model
    time_to_fit = round(time.time() - start_time, 3)
    sklearn_mse = round(metrics.mean_squared_error(y_test, sklearn_model.predict(X_test)),5)
    print(f"Sklearn MSE:{sklearn_mse} | Time to fit {time_to_fit}s |SKlearn coefs:{sklearn_model.coef_} | Intercept:{sklearn_model.intercept_}")
    if plot_on:
        plt.subplot(2, 1, 2)
        plt.scatter(X_train,y_train ,color='g', label='Training Data',zorder=0)
        plt.plot(X_train, sklearn_model.predict(X_train),color='b',label='Hypothesis',zorder=1)
        plt.scatter(X_test, sklearn_model.predict(X_test),color='r',label='Predicted Test Data',zorder=2)
        plt.scatter(X_test, y_test,color='k',label='Real Test Data', marker="2",zorder=3)
        plt.ylabel('House Value')
        plt.legend()
        plt.title(f'SKlearn Linear Regression - MSE {sklearn_mse} | Time to fit {time_to_fit}s')
        plt.show()