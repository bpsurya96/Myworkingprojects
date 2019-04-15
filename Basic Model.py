from sklearn import metrics

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as ps

import os

os.chdir("D:\\data science\\My working programs\\Assignment\\2")

def creditCSV():
  data = ps.read_csv("credit.csv")
  data.head()
  
  default = {'yes':1,'no':0}
  job = {'skilled' : 0, 'unskilled' : 1, 'management' : 2, 'unemployed' : 3}
  credit = {'critical' : 0, 'good' :1,'perfect':2,'poor':3,'very good' :4} 
  
  data.default = [default[i] for i in data.default]
  data.job = [job[i] for i in data.job]
  data.credit_history = [credit[j] for j in data.credit_history]
  
  #Selecting columns
  pred = data.iloc[ :,[1,2,4,7,8,9,12,13,14]]
  pred.head()
  
  #target - Defaults
  target = data.iloc[:,[16]]
  target.head()
  
  trainX,testX,trainY,testY = train_test_split(pred,target)
  main(trainX,testX,trainY,testY)

def DT(trainX,testX,trainY,testY):  
  dfc = DecisionTreeClassifier(max_depth=7,random_state = 10,max_features = 3)
  pr =  predict(dfc,"Decision Tree",trainX,testX,trainY,testY)
  return pr;
  
def RF(trainX,testX,trainY,testY):
  rfc = RandomForestClassifier(n_estimators = 160,max_depth=6,random_state=8,oob_score=True)
  pr = predict(rfc,"RandomForest",trainX,testX,trainY,testY)
  return pr;
  
def LT(trainX,testX,trainY,testY):
  lr = LogisticRegression(class_weight = 'balanced')
  pr = predict(lr,"Logistic Regression",trainX,testX,trainY,testY)
  return pr;
  
def predict(ski,model,trainX,testX,trainY,testY):
  ski.fit(trainX,trainY)
  y_predict = ski.predict(testX)
  y_prob = ski.predict_proba(testX)
  findMetrics(y_predict,y_prob,model,trainX,testX,trainY,testY)


def findMetrics(y_predict,y_prob,model,trainX,testX,trainY,testY):
  print(model);
  logLoss = metrics.log_loss(y_predict,y_prob)
  print("Log Loss",logLoss)
  acc = metrics.accuracy_score(y_predict,testY)
  print("Accurancy",acc)
  confusion = metrics.confusion_matrix(testY,y_predict)
  TP = confusion[1,1]
  TN = confusion[0, 0]
  FP = confusion[0, 1]
  FN = confusion[1, 0]
  print(TP,TN,FP,FN)
  sensitivity = metrics.recall_score(testY,y_predict)
  print("sensitivity",sensitivity)
  specificity = TN / (TN+TP)
  print("specificity",specificity)
  ppv = metrics.precision_score(testY,y_predict)
  print("PPV",ppv)
  npv = TN / float(TN + FN)
  print("NPV",npv)
  fpr, tpr, thresholds = metrics.roc_curve(testY, y_prob[:,1])
  auc = metrics.roc_auc_score(testY, y_prob[:,1])
  print("AUC" , auc)

def main(trainX,testX,trainY,testY):
  LT(trainX,testX,trainY,testY)
  print("-------------------------------------------------")
  DT(trainX,testX,trainY,testY)
  print("-------------------------------------------------")
  RF(trainX,testX,trainY,testY)
  print("-------------------------------------------------")  

creditCSV()
