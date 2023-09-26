#!/usr/bin/env python
# coding: utf-8

# In[73]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


# In[2]:


df=pd.read_csv("loan_prediction.csv")


# In[3]:


df.head()


# In[4]:


df=df.drop("Loan_ID",axis=1)


# In[5]:


df.isnull().sum()


# In[6]:


df["Gender"].fillna(df["Gender"].mode()[0],inplace=True)


# In[7]:


df["Married"].fillna(df["Married"].mode()[0],inplace=True)
df["Dependents"].fillna(df["Dependents"].mode()[0],inplace=True)
df["Self_Employed"].fillna(df["Self_Employed"].mode()[0],inplace=True)


# In[8]:


df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)


# In[9]:


import plotly.express as px


# In[10]:


gender_count = df['Gender'].value_counts()
fig_gender = px.bar(gender_count, 
                    x=gender_count.index, 
                    y=gender_count.values, 
                    title='Gender Distribution')
fig_gender.show()


# In[11]:


gender_count = df['Married'].value_counts()
fig_gender = px.bar(gender_count, 
                    x=gender_count.index, 
                    y=gender_count.values, 
                    title='Married')
fig_gender.show()


# In[12]:


fig_income = px.box(df, x='Loan_Status', 
                    y='ApplicantIncome',
                    color="Loan_Status", 
                    title='Loan_Status vs ApplicantIncome')
fig_income.show()


# In[13]:


Q1 = df['ApplicantIncome'].quantile(0.25)
Q3 = df['ApplicantIncome'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['ApplicantIncome'] >= lower_bound) & (df['ApplicantIncome'] <= upper_bound)]


# In[14]:


fig_income = px.box(df, x='Loan_Status', 
                    y='ApplicantIncome',
                    color="Loan_Status", 
                    title='Loan_Status vs ApplicantIncome')
fig_income.show()


# In[15]:


Q1 = df['CoapplicantIncome'].quantile(0.25)
Q3 = df['CoapplicantIncome'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['CoapplicantIncome'] >= lower_bound) & (df['CoapplicantIncome'] <= upper_bound)]


# In[16]:


cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
df = pd.get_dummies(df, columns=cat_cols)


# In[17]:


df


# In[18]:


X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[82]:


cl=RandomForestClassifier(n_estimators=5,criterion="entropy")


# In[83]:


cl.fit(X_train,y_train)


# In[84]:


y_pred=cl.predict(X_test)


# In[85]:


y_pred


# In[79]:


from sklearn.metrics import confusion_matrix


# In[86]:


confusion_matrix(y_test,y_pred)


# In[87]:


#cl.score(X_train,y_train)
cl.score(X_test,y_test)


# In[ ]:




