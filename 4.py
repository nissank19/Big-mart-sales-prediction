import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
data_Set=pd.read_csv('big_mart_data.csv')
data_Set['Item_Weight'].fillna(data_Set['Item_Weight'].mean())

#the following code is used to obtain the modes of each putlet type
mode=data_Set.pivot_table(values='Outlet_Size',columns='Outlet_Type',aggfunc=(lambda x:x.mode()[0]))
missing=data_Set['Outlet_Size'].isnull()
#following is used to replace
data_Set.loc[missing,'Outlet_Size']=data_Set.loc[missing,'Outlet_Type'].apply(lambda x:mode[x])
data_Set.replace({'Item_Fat_Content':{'low fat':'Low Fat','LF':'Low Fat','reg':'Regular'}},inplace=True)
encoder=LabelEncoder()
data_Set['Item_Identifier'] = encoder.fit_transform(data_Set['Item_Identifier'])

data_Set['Item_Fat_Content'] = encoder.fit_transform(data_Set['Item_Fat_Content'])

data_Set['Item_Type'] = encoder.fit_transform(data_Set['Item_Type'])

data_Set['Outlet_Identifier'] = encoder.fit_transform(data_Set['Outlet_Identifier'])

data_Set['Outlet_Size'] = encoder.fit_transform(data_Set['Outlet_Size'])

data_Set['Outlet_Location_Type'] = encoder.fit_transform(data_Set['Outlet_Location_Type'])
data_Set['Outlet_Type'] = encoder.fit_transform(data_Set['Outlet_Type'])


X=data_Set.drop('Item_Outlet_Sales',axis=1)
Y=data_Set['Item_Outlet_Sales']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)

regressor=XGBRegressor()
regressor.fit(X_train, Y_train)
training_data_prediction = regressor.predict(X_train)

r2_train=metrics.r2_score(Y_train,training_data_prediction)
print(r2_train)
