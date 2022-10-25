from cProfile import label
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, model_selection, metrics

def get_true_y(X):
    return (2 + X + 0.4*np.square(X) + 0.7*np.square(X))


if __name__ == "__main__":
    X = np.random.rand(20,1)
    y = get_true_y(X)

    X_train, X_val, y_train, y_val = model_selection.train_test_split(X, y, test_size=0.3)

    plt.scatter(X_train,y_train,label = 'Train data',zorder=2,color = 'b')
    plt.scatter(X_val,y_val,label = 'Validation data',zorder=2,color = 'g')

    Xmax_power = 10
    model = linear_model.LinearRegression() #create instance of the linear regression model
    model.fit(X_train, y_train)
    plt.plot(X_train,model.predict(X_train), label = f'Fit X')  

    for i in range(2,(Xmax_power+1)):
        X_power_i = np.power(X_train, i)
        y_of_Xpower = get_true_y(X_power_i)
        model.fit(X_power_i, y_of_Xpower)
        train_mse =round(metrics.mean_squared_error(y_of_Xpower, model.predict(X_power_i)),3)
        val_mse = round(metrics.mean_squared_error(y_val, model.predict(X_val)),3)
        plt.plot(X_train,model.predict(X_train), label = f'Fit X^{i}|Train MSE {train_mse}|Val MSE {val_mse}',zorder=1)

    plt.legend()
    plt.title('L2 Ridge regression')
    plt.xlabel('X values')
    plt.ylabel('y values')
    plt.show()



