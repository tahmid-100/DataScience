import numpy as np
import matplotlib.pyplot as plt
class Perceptron:
    def __init__(self,lr=.01,epochs=100):
        self.lr = lr
        self.epochs = epochs

    def Activation(self,x):
        return 1 if x > 0 else 0

    def train(self,X,y):
        self.weights=np.zeros(X.shape[1])
        self.bias=0

        for i in range (self.epochs):
            for xi,target in zip(X,y):
                linear_output=np.dot(self.weights,xi)+self.bias
                #print(xi,linear_output)

                prediction=self.Activation(linear_output)
                update=self.lr*(target-prediction)
                self.weights+=update*xi
                self.bias+=update

    def predict(self, X):
        return [self.Activation(np.dot(xi, self.weights) + self.bias) for xi in X]

    def coefficient(self):
        return self.weights

    def intercept(self):
        return self.bias

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 0, 0, 1])

model=Perceptron()
model.train(X,y)
print(model.predict(X))
print(model.coefficient())
print(model.intercept())
