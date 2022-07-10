import math
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import ttk
import pandas as pd
from functools import partial
from classification import Model

iris_data = pd.read_csv("IrisData.txt")
class_1 = iris_data[:51]
class_2 = iris_data[51:101]
class_3 = iris_data[101:]

# collect variables from GUI
def draw_features():
    F1 = feature1.get()
    F2 = feature2.get()

    plt.figure("Draw without Training")

    plt.scatter(class_1[F1], class_1[F2])
    plt.scatter(class_2[F1], class_2[F2])
    plt.scatter(class_3[F1], class_3[F2])

    plt.show()


def clear_data():
    feature1_comboBox.set("")
    feature2_comboBox.set("")
    feature1.set("")
    feature2.set("")
    class1.set("")
    class2.set("")
    class1_comboBox.set("")
    class2_comboBox.set("")
    learning_rate.set(0)
    number_of_epochs.set(0)
    feature1_comboBox.focus()
    print(str(bias))


def train_model():
    model = Model(class1.get(), class2.get(), feature1.get(), feature2.get(), float(learning_rate.get()),number_of_epochs.get(), float(MSE_threshold.get()), bias.get())
    weights = model.train() 
    #draw_boundry(weights, model)
    model.test()


def draw_boundry(weights, model):
    F1 = feature1.get()
    F2 = feature2.get()
    C1 = class1.get()
    C2 = class2.get()
    first = iris_data.where(iris_data['Class'] == C1).dropna()
    second = iris_data.where(iris_data['Class'] == C2).dropna()
    print("c1 : " + C1 + " c2: " + C2)
    min_f1 = min(first[F1].min(), second[F1].min())
    max_f1 = max(first[F1].max(), second[F1].max())
    
    point1 = -((weights[0, 1] * min_f1) + weights[0, 0]) / weights[0, 2]
    point2 = -((weights[0, 1] * max_f1) + weights[0, 0]) / weights[0, 2]
    print("first :point1 : " + str(min_f1) + " point2: " + str(max_f1))
    plt.figure("Decision Boundry", dpi=120)

    plt.scatter(first[F1], first[F2])
    plt.scatter(second[F1], second[F2])
    plt.plot([min_f1, max_f1],[point1, point2])
    plt.show()
    

master = Tk()
master.title("Assigment 2")
master.geometry("550x500")


# Features
feature1_label = Label(master, text="Feature 1", font=25)
feature1_label.grid(row=1, column=1, padx=10, pady=10)
feature2_label = Label(master, text="Feature 2", font=25)
feature2_label.grid(row=3, column=1, padx=10, pady=10)

features = ("X1", "X2", "X3", "X4")

feature1 = StringVar()
feature1_comboBox = ttk.Combobox(
    master, width=20, textvariable=feature1, values=features
)
feature1_comboBox.grid(row=1, column=10)
feature1_comboBox.focus()

feature2 = StringVar()
feature2_comboBox = ttk.Combobox(
    master, width=20, textvariable=feature2, values=features
)
feature2_comboBox.grid(row=3, column=10)

# Classes
class1_label = Label(master, text="Class 1", font=25)
class1_label.grid(row=5, column=1, padx=10, pady=10)
class2_label = Label(master, text="Class 2", font=25)
class2_label.grid(row=7, column=1, padx=10, pady=10)

classes = ("Iris-setosa", "Iris-versicolor", "Iris-virginica")

class1 = StringVar()
class1_comboBox = ttk.Combobox(master, width=20, textvariable=class1, values=classes)
class1_comboBox.grid(row=5, column=10)

class2 = StringVar()
class2_comboBox = ttk.Combobox(master, width=20, textvariable=class2, values=classes)
class2_comboBox.grid(row=7, column=10)


# Learning rate
learning_rate_label = Label(master, text="Learning Rate", font=25)
learning_rate = StringVar()
learning_rate_entry = Entry(master, font=20, textvariable=learning_rate)
learning_rate_label.grid(row=9, column=1, padx=10, pady=10)
learning_rate_entry.grid(row=9, column=10)

# number of epochs
Label(master, text="number of epochs", font=25).grid(row=11, column=1)
number_of_epochs = IntVar()
number_of_epochs_entry = Entry(master, font=20, textvariable=number_of_epochs)
number_of_epochs_entry.grid(row=11, column=10, padx=10, pady=10)

# MSE
Label(master, text="MSE Threshold", font=25).grid(row=14, column=1)
MSE_threshold = StringVar()
MSE_entry = Entry(master, font=20, textvariable=MSE_threshold)
MSE_entry.grid(row=14, column=10, padx=10, pady=10)

# Bias
Label(master, text="Bias", font=25).grid(row=17, column=1, padx=10, pady=10)
bias = BooleanVar(value=True)
Radiobutton(master, text="YES", variable=bias, value=True).grid(row=17, column=5)
Radiobutton(master, text="NO", variable=bias, value=False).grid(row=17, column=10)

# Buttons
Button(master, text="draw Features", command=draw_features).grid(row=3, column=12)
Button(master, text="Train", command=train_model).grid(row=20, columnspan=35, pady=10)
#Button(master, text="Test", command=).grid(row=30, columnspan=35, pady=10)
Button(master, text="Clear Data", command=clear_data).grid(row=40, columnspan=35, pady=10)


master.mainloop()
