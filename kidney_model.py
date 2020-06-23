import pandas as pd 
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle


df=pd.read_csv('kidney_disease.csv')

df['classification'].replace('ckd\t','ckd',inplace=True)
df.interpolate(inplace=True)
df.fillna(method='ffill',inplace=True)
df['rbc'].fillna(0,inplace=True)
df['rbc'].replace({'normal':0,'abnormal':1},inplace=True)
df['classification'].replace({'ckd':1,'notckd':0},inplace=True)
df['rbc'].fillna(df['rbc'].mode(),inplace=True)


X=df.iloc[:,[1,2,3,4,5,6]]

y=df['classification']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

model=LogisticRegression(max_iter=1200000)

model.fit(X_train,y_train)

with open('kidney_predict','wb') as f:
	pickle.dump(model,f)


