import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class Model:
    def __init__(self, learning_rate=0.001, epoch=2000, use_bias=True, num_of_layers=2, size_of_layers=[32, 16], is_sigmoid=False) -> None:
        np.random.seed(9)
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.use_bias = use_bias
        #adding one to include input and output layers
        self.num_of_layers = num_of_layers + 2
        self.layers = size_of_layers
        self.is_sigmoid = is_sigmoid
        #read data
        data = pd.read_csv("IrisData.txt")
        #encoding
        self.classes = data.Class.unique().tolist()
        #add input and output layers to layers list
        self.layers.insert(0, data.shape[1] - 1)
        self.layers.append(len(self.classes))
        data.Class = data.Class.apply(lambda c: self.classes.index(c))
        #split train test
        train, test = train_test_split(data, test_size=0.4, stratify=data.iloc[:,-1])
        train = train.to_numpy()

        self.train_x = train[:,:-1]
        self.train_y = train[:, -1]

        test = test.to_numpy()
        self.test_x = test[:,:-1]
        self.test_y = test[:, -1]
    
    def fit(self):
        self.params = self.initialize()
        for j in range(self.epoch + 1):
            for i in range(self.train_x.shape[0]):
                x = self.train_x[i,:].reshape(4,1)
                y = self.train_y[i]

                cache = self.multi_forward(x, self.params)
                errors = self.backward(cache, y, x)
                self.update(self.params, errors, cache)

            if j % 100 == 0:
                print("At epoch " + str(j))
                self.accuracy(self.train_x, self.train_y, mode="Training")

        y_predict = self.accuracy(self.train_x, self.train_y, mode="Final Training")
        self.confusion_matrix(self.train_y, y_predict, "Training")
        return

    def confusion_matrix(self, y, y_predict, mode="Testing"):
        cm = confusion_matrix(y, y_predict)
        cm_df = pd.DataFrame(cm, index = self.classes, columns = self.classes)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm_df, annot=True)
        plt.title(mode + ' Confusion Matrix')
        plt.ylabel('Actal Values')
        plt.xlabel('Predicted Values')
        plt.show()
        return

    def test(self):
        y_predict = self.accuracy(self.test_x, self.test_y)
        self.confusion_matrix(self.test_y, y_predict, "Test")
        return
    
    def accuracy(self, x, y, mode="Test"):
        y_predict = []
        acc = 0
        for i in range(x.shape[0]):
            xi = x[i].reshape(4,1)
            yi = y[i]
            y_hat = self.predict(xi)
            y_predict.append(y_hat.astype(int))
            if y_hat.astype(int) == yi.astype(int):
                acc += 1
        print(mode + " accuracy = " + str( 100 * (acc / self.train_x.shape[0])) + "%")
        return y_predict

    def predict(self, x):
        A = self.multi_forward(x, self.params)
        result = np.argmax(A['C' + str(self.num_of_layers - 1)][0])
        return result

    def decode(self, index):
        return self.classes[index]

    def encode(self, y):
        encodings = np.zeros([len(self.classes)]).reshape(3, 1).astype(int)
        encodings[y.astype(int)] = 1
        return encodings
    
    def initialize(self):
        params = {}
        for i in range(1, self.num_of_layers):
            params['W' + str(i)] = np.random.randn(self.layers[i], self.layers[i - 1]) * 0.2
            params['b' + str(i)] = np.zeros((self.layers[i], 1))
        return params
    
    def forward(self, A_prev, W, b):
        z = np.dot(W, A_prev) + b
        if self.is_sigmoid:
            A = sigmoid(z)
        else:
            A = tangent(z)
        return A
    
    def multi_forward(self, x, params):
        cache = {}
        A = x
        for i in range(1, self.num_of_layers):
            prev_A = A
            W = params['W' + str(i)]
            b = params['b' + str(i)]
            A = self.forward(prev_A, W, b)
            cache['C' + str(i)] = (A, W, b)
        return cache
    
    def backward(self, cache, label, x):
        deltas = []
        l = self.num_of_layers
        A = cache["C" + str(l - 1)][0]

        y = self.encode(label)
        if self.is_sigmoid:
            delta = (y - A) * (A * (1 - A))
        else:
            delta = (y - A) * ((1 - A) * (1 + A))
        deltas.append(delta)

        for i in range(self.num_of_layers - 2, -1, -1):
            delta_prev = delta
            if i != 0:
                A = cache['C' + str(i)][0]
            else:
                A = x
            W = cache['C' + str(i + 1)][1]
            b = cache['C' + str(i + 1)][2]
            if self.is_sigmoid:
                delta = np.dot(W.T, delta_prev) * (A * (1 - A))
            else:
                delta = np.dot(W.T, delta_prev) * ((1 - A) * (1 + A))
            deltas.append(delta)
        deltas.reverse()
        return deltas

    def update(self, params, deltas, cache):
        for i in range(1, self.num_of_layers):
            params['W' + str(i)] = params['W' + str(i)] + (self.learning_rate * deltas[i] * cache['C' + str(i)][0])
            if self.use_bias:
                params['b' + str(i)] = params['b' + str(i)] + (self.learning_rate * deltas[i])
        return
        
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derv(x):
    return x * (1 - x)

def tangent(x):
    return (1 - np.exp(-x)) / (1 + np.exp(-x))

def tanget_derv( x):
    return (1 - x) * (1 + x)


#model = Model()
#model.fit()
#model.test()