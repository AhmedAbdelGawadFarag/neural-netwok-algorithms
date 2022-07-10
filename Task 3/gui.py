from distutils.cmd import Command
import tkinter
from tkinter import *
from classification import Model

class gui:
    def __init__(self) -> None:
        self.root = Tk()
        self.root.title("Assigment 3")
        self.root.geometry("550x500")

        #storing variables
        self.learning_rate = StringVar(value="0.001")
        self.bias = BooleanVar(value=True)
        self.number_of_epochs = IntVar(value=2000)
        self.is_sigmoid = BooleanVar(value=False)
        self.number_of_hidden_layers = IntVar(value=2)
        self.size_of_hidden_layers = StringVar(value="32 16")
        self.list_of_hidden_layers = []

        # Learning rate
        learning_rate_label = Label(self.root, text="Learning Rate", font=25)
        learning_rate_entry = Entry(self.root, font=20, textvariable=self.learning_rate)
        learning_rate_label.grid(row=9, column=1, padx=10, pady=10)
        learning_rate_entry.grid(row=9, column=10)
        
        # number of epochs
        Label(self.root, text="number of epochs", font=25).grid(row=11, column=1)
        number_of_epochs_entry = Entry(self.root, font=20, textvariable=self.number_of_epochs)
        number_of_epochs_entry.grid(row=11, column=10, padx=10, pady=10)

        # number of Layers
        Label(self.root, text="number of Layers", font=25).grid(row=13, column=1)
        number_of_epochs_entry = Entry(self.root, font=20, textvariable=self.number_of_hidden_layers)
        number_of_epochs_entry.grid(row=13, column=10, padx=10, pady=10)

        # size of Layers
        Label(self.root, text="Size of Layers", font=25).grid(row=15, column=1)
        number_of_epochs_entry = Entry(self.root, font=20, textvariable=self.size_of_hidden_layers)
        number_of_epochs_entry.grid(row=15, column=10, padx=10, pady=10)

        # Bias
        Label(self.root, text="Bias", font=25).grid(row=17, column=1, padx=10, pady=10)
        Radiobutton(self.root, text="YES", variable=self.bias, value=True).grid(row=17, column=5)
        Radiobutton(self.root, text="NO", variable=self.bias, value=False).grid(row=17, column=10)

        # Activation Function
        Label(self.root, text="Activation", font=25).grid(row=19, column=1, padx=10, pady=10)
        Radiobutton(self.root, text="Sigmoid", variable=self.is_sigmoid, value=True).grid(row=19, column=5)
        Radiobutton(self.root, text="tanh", variable=self.is_sigmoid, value=False).grid(row=19, column=10)
        
        # Buttons
        Button(self.root, text="Train", command=self.train).grid(row=21, columnspan=35, pady=10)
        Button(self.root, text="Clear Data",).grid(row=22, columnspan=35, pady=10)

    def clear(self):
        self.learning_rate.set(0)
        self.number_of_epochs.set(0)
        self.number_of_hidden_layers.set(0)
        

    def run(self):
        self.root.mainloop()

    def train(self):
        self.list_of_hidden_layers = self.size_of_hidden_layers.get().split()
        for i in range(len(self.list_of_hidden_layers)):
            self.list_of_hidden_layers[i] = int(self.list_of_hidden_layers[i])
        print("list of hidden layers: " + str(self.list_of_hidden_layers))

        if len(self.list_of_hidden_layers) != self.number_of_hidden_layers.get():
            print("Number of passed layers size does not match of entered number of hidden layers, use space to split.")
            return

        self.model = Model(
            learning_rate=float(self.learning_rate.get()), 
            epoch=self.number_of_epochs.get(), 
            use_bias=self.bias.get(), 
            num_of_layers=self.number_of_hidden_layers.get(), 
            size_of_layers=self.list_of_hidden_layers, 
            is_sigmoid=self.is_sigmoid.get(),
        )

        self.model.fit()
        self.model.test()