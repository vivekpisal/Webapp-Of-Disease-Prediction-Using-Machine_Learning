from flask import Flask,request,render_template
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle

#pip freeze > requirements.txt
app=Flask(__name__)



@app.route("/diabetes",methods=["GET","POST"])
def diabetes():
	if request.method=='GET':
		return render_template("form.html")
	else:
		ds=pd.read_csv('diabetes1.csv')
		X=ds.iloc[:,[0,1,2,5,6]]
		y=ds.iloc[::,-1]
		X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.10,random_state=0)
		model=LogisticRegression()
		model.fit(X_train,y_train)
		Pregnancies=int(request.form['Pregnancies'])
		Glucose=int(request.form['Glucose'])
		BloodPressure=int(request.form['BloodPressure'])
		BMI=float(request.form['BMI'])
		DiabetesPedigreeFunction=float(request.form['DiabetesPedigreeFunction'])
		new=np.array([[Pregnancies,Glucose,BloodPressure,BMI,DiabetesPedigreeFunction]])
		y_pred=model.predict(new)
		return render_template("result.html",y_pred=y_pred)




@app.route("/heart",methods=["GET","POST"])
def heart():
	if request.method=='GET':
		return render_template('form1.html')
	else:
		ds=pd.read_csv('heart.csv')
		X=ds.drop('target',axis=1)
		y=ds.iloc[:,-1]
		X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.10,random_state=0)
		reg=LogisticRegression(max_iter=1200000)
		reg.fit(X_train,y_train)
		Age=int(request.form['Age'])
		gender=int(request.form['gender'])
		cp=int(request.form['cp'])
		trestbps=int(request.form['trestbps'])
		chol=int(request.form['chol'])
		fbs=int(request.form['fbs'])
		restecg=int(request.form['restecg'])
		thalach=int(request.form['thalach'])
		new=np.array([[Age,gender,cp,trestbps,chol,fbs,restecg,thalach,ds['exang'].mean(),ds['oldpeak'].mean(),ds['slope'].mean(),ds['ca'].mean(),ds['thal'].mean()]])
		y_pred=reg.predict(new)
		return render_template("result1.html",y_pred=y_pred)




@app.route("/liverprediction",methods=["GET","POST"])
def liver():
	if request.method=='GET':
		return render_template('form2.html')
	else:
		df=pd.read_csv('indian_liver_patient.csv')
		dummies=pd.get_dummies(df['Gender'])
		df['Albumin_and_Globulin_Ratio'].fillna(df['Albumin_and_Globulin_Ratio'].mean(),inplace=True)
		df1=pd.concat([df,dummies],axis='columns')
		X=df1.drop(['Gender','Female','Dataset'],axis=1)
		y=df1.iloc[:,-3]
		X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.10,random_state=0)
		model = LogisticRegression(max_iter=12000000)
		model.fit(X_train,y_train)
		Age=int(request.form['Age'])
		gender=int(request.form['gender'])
		Total_Bilirubin=int(request.form['Total_Bilirubin'])
		Alkaline_Phosphotase=int(request.form['Alkaline_Phosphotase'])
		Alamine_Aminotransferase=int(request.form['Alamine_Aminotransferase'])
		Aspartate_Aminotransferase=int(request.form['Aspartate_Aminotransferase'])
		Total_Protiens=int(request.form['Total_Protiens'])		
		Albumin=int(request.form['Albumin'])
		new=np.array([[Age,Total_Bilirubin,df1['Direct_Bilirubin'].mean(),Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,Albumin,df1['Albumin_and_Globulin_Ratio'].mean(),gender]])
		y_pred=model.predict(new)
		return render_template('result2.html',y_pred=y_pred)
		

@app.route('/kidneydisease',methods=['GET','POST'])
def kidney():
	if request.method=='GET':
		return render_template('kidneyform.html')
	else:
		with open('kidney_predict','rb') as f:
			model=pickle.load(f)
		Age=int(request.form['Age'])
		Blood_Pressure=int(request.form['Blood_Pressure'])
		Specific_Gravity=float(request.form['Specific_Gravity'])
		Albumin=int(request.form['Albumin'])
		Sugar=int(request.form['Sugar'])
		Red_Blood_Cells=int(request.form['Red_Blood_Cells'])
		new=np.array([[Age,Blood_Pressure,Specific_Gravity,Albumin,Sugar,Red_Blood_Cells]])
		y_pred=model.predict(new)
		return render_template('result3.html',y_pred=y_pred)



@app.route('/about')
def about():
	return render_template('about.html')



@app.route('/')
def home():
	return render_template('home.html')



if __name__ == '__main__':
	app.run(debug=True)