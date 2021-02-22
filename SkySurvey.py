#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time

scoreDict = {}
timeDict={}



def print_info(clfname , Score , t_time):
    print('\n')
    print(clfname.upper())
    print("Score of {} : {}".format(clfname,Score))
    print("Accuracy of {} : {} %".format(clfname , Score*100))
    print("Training Time: : {} seconds".format(t_time))
    scoreDict[clfname] = Score*100
    timeDict[clfname] = t_time

def make_pred(arr_pred , name , pred_proba):
    print("{} Prediction:".format(name))
    print(arr_pred)
    if(arr_pred[0] == 0):
        print('STAR')
    else:
        print("GALAXY")
    print('Prediction Probability:')
    print(pred_proba)
        

def histogram(data, x_label, y_label, title):
    _, ax = plt.subplots()
    ax.hist(data, color = ['b', 'r'])
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)

data_file = '/Users/saarthakgupta/Downloads/Skyserver_SQL2_27_2018 6_51_39 PM.csv'
data = pd.read_csv(data_file)
data = data.dropna()

print(data.head())
print(data.info())

data.drop(["objid", "specobjid", "rerun"], axis = 1, inplace = True)
data = data[data["class"] != "QSO"]

sns.countplot(x= "class", data = data)
print('\n')
print(data["class"].value_counts())

sns.pairplot(data.loc[:,['u' , 'g' , 'r' , 'i' , 'z' , 'class']], hue="class")
plt.show()


data["class"] = [1 if each == "GALAXY" else 0 for each in data["class"]] 

y = data['class'].values
x_data = data.drop(["class"] , axis = 1)
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

print(x.head())

x_train , x_test , y_train , y_test = train_test_split(x , y ,test_size = 0.2 , random_state = 42)

item = x_test.values[0]     #CHANGE ITEM
item = (np.expand_dims(item,0))

#Logistic Regression
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
start = time.time()
lr.fit(x_train, y_train)
end = time.time()
lr_time = end - start
Score1 = lr.score(x_test, y_test)
print_info('Logistic Regression' , Score1 , lr_time)

prediction_single = lr.predict(item) 
pp=lr.predict_proba(item) * 100
make_pred(prediction_single, 'Logistic Regression', pp)
#Logistic Regression

#Decision Tree
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
start = time.time()
dtc.fit(x_train , y_train)
end = time.time()
dtc_time = end - start
Score2 = dtc.score(x_test , y_test)
print_info('Decision Tree' , Score2 , dtc_time)

prediction_single = dtc.predict(item) 
pp=dtc.predict_proba(item) * 100
make_pred(prediction_single, 'Decision Tree', pp)
#Decision Tree

#SVM
from sklearn.svm import SVC

svm = SVC(random_state=42, probability=True)
start = time.time()
svm.fit(x_train , y_train)
end = time.time()
svm_time = end - start
Score3 = svm.score(x_test , y_test)
print_info('Support Vector Machines' , Score3 , svm_time)

prediction_single = svm.predict(item) 
pp=svm.predict_proba(item) * 100
make_pred(prediction_single, 'Support Vector Machine', pp)
#SVM

#Naive Bayes
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
start = time.time()
nb.fit(x_train , y_train)
end = time.time()
nb_time = end - start
Score4 = nb.score(x_test , y_test)
print_info('Naive Bayes' , Score4 , nb_time)

prediction_single = nb.predict(item) 
pp=nb.predict_proba(item) * 100
make_pred(prediction_single, 'Naive Bayes', pp)
#Naive Bayes

#Random Forest
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
start = time.time()
rfc.fit(x_train , y_train)
end = time.time()
rfc_time = end - start
Score5 = rfc.score(x_test , y_test)
print_info('Random Forest' , Score5 , rfc_time)

prediction_single = rfc.predict(item) 
pp=rfc.predict_proba(item) * 100
make_pred(prediction_single, 'Random Forest', pp)
#Random Forest

#K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
start = time.time()
knn.fit(x_train , y_train)
end = time.time()
knn_time = end - start
Score6 = knn.score(x_test , y_test)
print_info('K-Nearest Neighbors' , Score6 , knn_time)

prediction_single = knn.predict(item) 
pp=knn.predict_proba(item) * 100
make_pred(prediction_single, 'K-Nearest Neighbors', pp)
#K-Nearest Neighbors

print("\nComparison of Classifiers:")
print("\nSCORE:\n")
print(scoreDict)
print('\nTIME:\n')
print(timeDict)

comparisonData = pd.DataFrame.from_dict(scoreDict,
                                        orient = 'index',
                                        columns = ["Accuracy"])
comparisonData.head(10) 

comparisonTimeData = pd.DataFrame.from_dict(timeDict,
                                            orient = 'index',
                                            columns = ["Time"])
comparisonTimeData.head(10)                                               

plt.figure(figsize = (20 , 7))
sns.barplot(x = comparisonData.index, y = comparisonData.Accuracy)
plt.ylabel("Accuracy")
plt.xlabel("Classification Model")
plt.title("Comparison of Accuracy of Classification Models", color = 'blue' , fontsize = 15)
plt.show()

plt.figure(figsize=(20,7))
sns.barplot(x= comparisonTimeData.index , y = comparisonTimeData.Time)
plt.ylabel("Time")
plt.xlabel("Classification Model")
plt.title("Comparison of Training Times of Classification Models", color = 'blue' , fontsize = 15)
plt.show()

#Neural Network
model = keras.Sequential([
        keras.layers.Dense(14 , activation=tf.nn.relu),
        keras.layers.Dense(128 , activation=tf.nn.relu),
        keras.layers.Dense(1 , activation=tf.nn.sigmoid)
        ])

model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

start = time.time()
model.fit(x_train.values , y_train , epochs=10 , batch_size=20)
end=time.time()

scores = model.evaluate(x_test , y_test)

print("\nTraining Time: {}".format(end-start))
print("\n%s:%.2f%%"%(model.metrics_names[1] , scores[1]*100))

predictions_single = model.predict(item)
print('Neural Network Prediction:')
a = predictions_single[0][0]
print(predictions_single)
if(a>0.98):
    print("GALAXY")
else:
    print('STAR')

#TAKING INPUT AND MAKING PREDICTION

l1=[input("ra= ")]
l2=[input("dec= ")]
l3=[input("u= ")]
l4=[input("g= ")]
l5=[input("r= ")]
l6=[input("i= ")]
l7=[input("z= ")]
l8=[input("run= ")]
l9=[input("camcol= ")]
l10=[input("field= ")]
l11=[input("redshift= ")]
l12=[input("plate= ")]
l13=[input("mjd= ")]
l14=[input("fiberid= ")]


data_new = {'ra':l1,'dec':l2,'u':l3,'g':l4,'r':l5,'i':l6,'z':l7,'run':l8,'camcol':l9,'feild':l10,'redshift':l11,'plate':l12,'mjd':l13,'fiberid':l14}

df = pd.DataFrame(data_new)
print(df)

prediction_new = dtc.predict(df) 

make_pred(prediction_new, 'Decision Tree', pp)