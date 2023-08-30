from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

main = tkinter.Tk()
main.title("Human Activity Recognization Using SmartPhone Data")
main.geometry("1300x1200")
global dataset
global train
global test
global X_train, X_text, y_train, y_test
def upload():
  global dataset
  dataset=filedialog.askdirectory(initialdir=".")
  pathlabel.config(text=dataset)
  text.delete('1.0', END)
  text.insert(END,dataset+" loaded\n");
def readData():
  global train
  global test
  train=pd.read_csv(dataset+"/train.csv")
  test=pd.read_csv(dataset+"/test.csv")
  text.delete('1.0', END)
  text.insert(END,train.head())
  text.insert(END,test.head())
def preprocess():
  global X_train, X_test, y_train, y_test
  y_train = train.Activity
  X_train = pd.DataFrame(train.drop(['Activity','subject'],axis=1))
  y_test = test.Activity
  X_test = pd.DataFrame(test.drop(['Activity','subject'],axis=1))
  
  y_train.replace(to_replace='WALKING',value=1,inplace=True)
  y_train.replace(to_replace='WALKING_UPSTAIRS',value=2,inplace=True)
  y_train.replace(to_replace='WALKING_DOWNSTAIRS',value=3,inplace=True)
  y_train.replace(to_replace='SITTING',value=4,inplace=True)
  y_train.replace(to_replace='STANDING',value=5,inplace=True)
  y_train.replace(to_replace='LAYING',value=6,inplace=True)


  y_test.replace(to_replace='WALKING',value=1,inplace=True)
  y_test.replace(to_replace='WALKING_UPSTAIRS',value=2,inplace=True)
  y_test.replace(to_replace='WALKING_DOWNSTAIRS',value=3,inplace=True)
  y_test.replace(to_replace='SITTING',value=4,inplace=True)
  y_test.replace(to_replace='STANDING',value=5,inplace=True)
  y_test.replace(to_replace='LAYING',value=6,inplace=True)
  
  text.delete('1.0', END)
  text.insert(END,"70% Training Data : "+str(X_train.shape[0])+"\n")
  text.insert(END,"30% Testing Data : "+str(X_test.shape[0])+"\n")
global svm_acc
def runSVM():
  global svm_acc
  cls = svm.SVC(C=2.0,gamma='scale',kernel = 'rbf', random_state = 2)
  cls.fit(X_train, y_train)
  prediction_data =cls.predict(X_test)
  svm_acc = accuracy_score(y_test, prediction_data)
  text.delete('1.0', END)
  text.insert(END,"SVM Accuracy : "+str(svm_acc*100)+"\n")
global rfc_acc
def runRForest():
  global rfc_acc
  ran_mdl= RandomForestClassifier(n_estimators = 10)
  ran_mdl.fit(X_train, y_train)
  pred_mdl=ran_mdl.predict(X_test)
  rfc_acc= accuracy_score(y_test,pred_mdl)
  text.insert(END,"RandomForest Accuracy : "+str(rfc_acc*100)+"\n")
global dec_acc
def decisionTree():
  global dec_acc
  dec_model=DecisionTreeClassifier()
  dec_model.fit(X_train, y_train)
  dec_pred=dec_model.predict(X_test)
  dec_acc=accuracy_score(y_test,dec_pred)
  text.insert(END,"Decision Tree Accuracy : "+str(dec_acc*100)+"\n")

def graph():
   height = [svm_acc, rfc_acc, dec_acc]
   bars = ('SVM Accuracy','RandomForest Accuracy','DecisionTree Accuracy')
   y_pos = np.arange(len(bars))
   plt.bar(y_pos, height)
   plt.xticks(y_pos, bars)
   plt.show()
def recognize():
  plt.title("Plot of count of activities")
  sns.countplot(train.Activity)
  plt.xlabel("Activity")
  plt.xticks(rotation=45)
  plt.ylabel("Count of activities")
  plt.show()
def exitSystem():
    main.destroy()

font = ('times', 16, 'bold')
title = Label(main, text='Human Activity Recognisation Using SmartPhone Data')
title.config(bg='brown', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Dataset", command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=460,y=100)

exploreButton = Button(main, text="Explore Dataset",command=readData)
exploreButton.place(x=50,y=150)
exploreButton.config(font=font1)

processButton = Button(main, text="Preprocessing",command=preprocess)
processButton.place(x=200,y=150)
processButton.config(font=font1)

SVMButton = Button(main, text="Run SVM Algorithm",command=runSVM)
SVMButton.place(x=360,y=150)
SVMButton.config(font=font1)

rlButton = Button(main, text="Run Random Forest",command=runRForest)
rlButton.place(x=550,y=150)
rlButton.config(font=font1)

DecisionlButton = Button(main, text="Run Decision Tree",command=decisionTree)
DecisionlButton.place(x=780,y=150)
DecisionlButton.config(font=font1) 

ComparisionButton = Button(main, text="Comparision Graph",command=graph)
ComparisionButton.place(x=50,y=200)
ComparisionButton.config(font=font1)

RecognizeButton = Button(main, text="Recognize Activity",command=recognize)
RecognizeButton.place(x=280,y=200)
RecognizeButton.config(font=font1)

exitButton = Button(main, text="Close Here", command=exitSystem)
exitButton.place(x=500,y=200)
exitButton.config(font=font1) 


font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1)


main.config(bg='green')
main.mainloop()
