# -*- coding: utf-8 -*-
"""

#LogisticRegression
"""

import pandas as pd
import numpy as np
import seaborn as sns

"""###Data importing and Data Preprocessing"""

df=pd.read_csv('/content/sample_data/House-Price.csv',header=0)

df.head()

df.info()

df.shape

df.describe()

"""##Numerical Variables"""

sns.jointplot(x='n_hot_rooms',y='Sold',data=df)

sns.boxplot(x='n_hot_rooms',data=df)

sns.jointplot(x='rainfall',y='Sold',data=df)

"""#Categorical Variables"""

sns.countplot(x='airport',data=df)

sns.countplot(x='waterbody',data=df)

sns.countplot(x='bus_ter',data=df)

"""#Observations 
* Missing values in n_hos_beds
* bus_ter has only yes values
* n_hot_rooms and rainfall outliers
"""

#Outliers
uv=np.percentile(df.n_hot_rooms,[99])[0]

df[df.n_hot_rooms>uv]

df.n_hot_rooms[df.n_hot_rooms>3*uv]=3*uv

lv=np.percentile(df.rainfall,[1])[0]

df.rainfall[df['rainfall']>0.3*lv]=0.3*lv

df.describe()

#Missing Values Treatment
df.n_hos_beds=df.n_hos_beds.fillna(df.n_hos_beds.mean())

df.info()

#Single Value Treatment
del df['bus_ter']
df.head()

df['avg_dist']=(df.dist1+df.dist2+df.dist3+df.dist4)/4

del df['dist1']
del df['dist2']
del df['dist3']
del df['dist4']

df.head()

#Now for categorical variables
df=pd.get_dummies(df)

df.head()

del df['airport_NO']

del df['waterbody_None']

"""###Creating LogisticRegression Model"""

#Using sklearn
from sklearn.linear_model import LogisticRegression

X=df[['price']]

y=df['Sold']

X.head()

y.head()

clf_lrs=LogisticRegression()

clf_lrs.fit(X,y)

clf_lrs.coef_

clf_lrs.intercept_

##Model using statsmode
import statsmodels.api as sn

X_cons=sn.add_constant(X)

X_cons.head()

import statsmodels.discrete.discrete_model as sm

logit=sm.Logit(y,X_cons).fit()

logit.summary()

"""##LogisticRegression with Multiple Regression"""

X=df.loc[:,df.columns!='Sold']

y=df['Sold']

clf_lr=LogisticRegression()
clf_lr.fit(X,y)

clf_lr.coef_

clf_lr.intercept_

##Using stats
X_cons=sn.add_constant(X)

logit=sm.Logit(y,X_cons).fit()

logit.summary()

"""##Predicting and Confusion Matrix"""

clf_lr.predict_proba(X)

y_pred=clf_lr.predict(X)##By default this takes boundary condition as 0.5
y_pred

y_pred_03=clf_lr.predict_proba(X)[:,1]>=0.3
y_pred_03

from sklearn.metrics import confusion_matrix
confusion_matrix(y,y_pred)

confusion_matrix(y,y_pred_03) #Boundary condition as 0.3

"""#Performance Matrix"""

from sklearn.metrics import precision_score,recall_score,roc_auc_score

precision_score(y,y_pred)

recall_score(y,y_pred)

roc_auc_score(y,y_pred)

"""###Test Train Split"""

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=0)

print(X_test,X_train,y_train,y_test)

clf_LR=LogisticRegression()

clf_LR.fit(X_train,y_train)

y_test_pred=clf_LR.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix

confusion_matrix(y_test,y_test_pred)

accuracy_score(y_test,y_test_pred)



