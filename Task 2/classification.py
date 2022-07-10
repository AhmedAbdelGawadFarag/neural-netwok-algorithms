from ast import Str
from tkinter import W
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib.offsetbox import AnchoredText
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.utils.multiclass import unique_labels
class Model:
    def __init__(self, class1, class2, feature1, feature2, learning_rate=0.001, epoch=10000, MSE_threshold = 0, use_biass=True) -> None:
        self.class1 = class1
        self.class2 = class2
        self.feature1 = feature1
        self.feature2 = feature2
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.MSE_threshold = MSE_threshold
        self.use_bias = use_biass

        data = pd.read_csv("IrisData.txt")
        # get column "feature1", "feature2", "Class" that pass the conditions data[class] == class1 or class2
        data_filtered = data.loc[
            (data["Class"] == class1) | (data["Class"] == class2),
            [feature1, feature2, "Class"],
        ]

        # split each class to split train test
        data_class1 = data_filtered.loc[data_filtered["Class"] == class1]
        train_class1, self.test_class1 = train_test_split(
            data_class1, test_size=0.4, shuffle=True
        )

        data_class2 = data_filtered.loc[data_filtered["Class"] == class2]
        train_class2, self.test_class2 = train_test_split(
            data_class2, test_size=0.4, shuffle=True
        )

        # combine train and test
        self.train_data = pd.concat([train_class1, train_class2])
        self.test_data = pd.concat([self.test_class1, self.test_class2])

        # encode class name to numbers
        mapping = {class1: 1, class2: -1}
        self.train_data = self.train_data.applymap(lambda c: mapping.get(c) if c in mapping else c)
        self.test_data = self.test_data.applymap(lambda c: mapping.get(c) if c in mapping else c)

    def train(self, debug=False):
        x_train = self.train_data.iloc[:, :-1].to_numpy(copy=True)
        y_train = self.train_data.iloc[:, -1].to_numpy(copy=True)
        y_train = y_train.reshape(y_train.shape[0], 1)
        self.W = np.random.randn(1, 3) * 0.01
        ones = np.ones((x_train.shape[0], 1))
        x_train = np.concatenate((ones, x_train), axis=1)

        if not self.use_bias:
            print("using bias: False")
            x_train[:, 0] = 0
            print("first row of x: " + str(x_train[0]))
            self.W[0] = 0
        epoch = 0
        while True and epoch < self.epoch:
            epoch = epoch + 1
            for i in range(x_train.shape[0]):
                y_hat = (np.dot(self.W, x_train[i, :]))
                if (y_hat - y_train[i]) != 0:
                    loss = y_train[i] - y_hat
                    self.W = self.W + self.learning_rate * loss * x_train[i, :]
            MSE = 0.0
            for i in range(x_train.shape[0]):
                y_hat = (np.dot(self.W, x_train[i, :]))
                ERROR = y_train[i] - y_hat
                MSE += (ERROR**2) / 2
            MSE /= x_train.shape[0]
            if MSE < self.MSE_threshold:
                break
        return self.W

    def test(self):
        x_test = self.test_data.iloc[:, :-1].to_numpy(copy=True)
        ones = np.ones((x_test.shape[0], 1))
        x_test = np.concatenate((ones, x_test), axis=1)
        y_test = self.test_data.iloc[:, -1].to_numpy(copy=True)
        y_test = y_test.reshape(y_test.shape[0], 1)
        y_pred = np.zeros(len(y_test))
        correct = 0
        for i in range(y_test.shape[0]):
            y_hat = np.sign(np.dot(self.W, x_test[i, :]))
            y_pred[i] = y_hat
            if y_hat == y_test[i]:
                correct += 1

        first = self.test_data.where(self.test_data['Class'] == 1).dropna()
        second = self.test_data.where(self.test_data['Class'] == -1).dropna()

        print("c1 : " + self.class1 + " c2: " + self.class2)
        min_f1 = min(first[self.feature1].min(), second[self.feature1].min())
        max_f1 = max(first[self.feature1].max(), second[self.feature1].max())

        point1 = -((self.W[0, 1] * min_f1) + self.W[0, 0]) / self.W[0, 2]
        point2 = -((self.W[0, 1] * max_f1) + self.W[0, 0]) / self.W[0, 2]

        plt.figure("Accuracy = " + str((correct / y_test.shape[0]) * 100) + "%")

        plt.scatter(
            first[self.feature1],
            first[self.feature2],
            label=self.class1,
            c="blue"
        )

        plt.scatter(
            second[self.feature1],
            second[self.feature2],
            label=self.class2,
            c="red"
        )
        plt.legend()
        print("point1 : " + str(min_f1) + " point2: " + str(max_f1))
        plt.plot([min_f1, max_f1], [point1, point2])
        print("Accuracy = " + str((correct / y_test.shape[0]) * 100) + "%")

        a = AnchoredText("Accuracy = " + str((correct / y_test.shape[0]) * 100) + "%", loc=1, pad=0.4, borderpad=0.5)
        plt.gca().add_artist(a)

        plt.show()

        #confusion matrix


        Labels = ["", ""]
        labels = unique_labels(y_test, y_pred)
        mp = {-1:self.class2, 1:self.class1}
        #mapping labels
        Labels[0] = mp[labels[0]]
        Labels[1] = mp[labels[1]]

        #Build confusion matrix
        conf = confusion_matrix(y_true=y_test, y_pred=y_pred)


        #plot confusion matrix using heatmap
        ax = sns.heatmap(conf, annot=True, cmap='Blues')

        ax.set_title('Confusion Matrix\n\n');
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Actual Values ');


        ax.xaxis.set_ticklabels(Labels)
        ax.yaxis.set_ticklabels(Labels)


        plt.show()
