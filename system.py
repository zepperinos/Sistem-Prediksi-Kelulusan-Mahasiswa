import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import filedialog

def selectfile():
    # open file
    filepath = filedialog.askopenfilename(title='Choose a file', initialdir='/')
    
    # read file and remove row with Enrolled as Target
    global data
    data = pd.read_csv(filepath)
    data = data[data.Target != 'Enrolled']
    
    # show file name on GUI
    filename = os.path.basename(filepath)
    label2.config(text = filename)
    
def predict():
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    # encode target to int
    le = LabelEncoder()
    y = le.fit_transform(y)

    # split datast into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # K-NN
    if radiobtn.get() == "0":
        classifier = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=le.classes_))
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=None)
        disp.plot()
        plt.show()

    # SVM
    elif radiobtn.get() == "1":
        classifier = svm.SVC(kernel='linear')
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=le.classes_))
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=None)
        disp.plot()
        plt.show()
        
    # Logistic Regression
    elif radiobtn.get() == "2":
        classifier = LogisticRegression()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=le.classes_))
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=None)
        disp.plot()
        plt.show()

# gui
window = tk.Tk()
window.title('Student Graduation Prediction')
radiobtn = tk.StringVar()

frame = tk.Frame(window, width=350, height=350, bg='grey')
frame.pack()

# input dir
label = tk.Label(window, text="Dataset :", bg='grey', fg='black', font=("Helvetica", 13))
label.place(x=50, y=40)

label2 = tk.Label(window, text="", bg='grey', fg='black', font=("Helvetica", 13))
label2.place(x=120, y=40)

btn = tk.Button(window, text="Choose Dataset File", bg='grey', fg='black', font=("Helvetica", 13), command=selectfile)
btn.place(x=50, y=70)

# choose algorithm
label3 = tk.Label(window, text="Choose an Algorithm :", bg='grey', fg='black', font=("Helvetica", 13))
label3.place(x=50, y=130)

radio1 = tk.Radiobutton(window, text="K-Nearest Neighbor", variable=radiobtn, value="0", bg='grey', fg='black', font=("Helvetica", 13))
radio1.place(x=50, y=160)

radio2 = tk.Radiobutton(window, text="Support Machine Vector", variable=radiobtn, value="1", bg='grey', fg='black', font=("Helvetica", 13))
radio2.place(x=50, y=190)

radio3 = tk.Radiobutton(window, text="Logistic Regression", variable=radiobtn, value="2", bg='grey', fg='black', font=("Helvetica", 13))
radio3.place(x=50, y=220)

# run
btn1 = tk.Button(window, text="Run", bg='grey', fg='black', font=("Helvetica", 13), command=predict)
btn1.place(x=50, y=280)

window.mainloop()