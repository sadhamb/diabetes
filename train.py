import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import statsmodels.api as sm

df=pd.read_csv("diabetes2.csv")     
print(df)

df1_num=df.select_dtypes(include=['float64','int64'])
df1_num=df1_num.loc[:,'Pregnancies':'Outcome']
df_cat=df.select_dtypes(include='float')
dummy=pd.get_dummies(df_cat,drop_first=False)

sc=StandardScaler()

df1_num1_sc=pd.DataFrame(sc.fit_transform(df1_num),columns=df1_num.columns)
x=np.array(df['Glucose']).reshape(-1,1)
y=np.array(df['BloodPressure'])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
MLR_model1=sm.OLS(y_train,x_train).fit()
print(MLR_model1.summary())

