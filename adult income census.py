#!/usr/bin/env python
# coding: utf-8

# In[134]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[135]:


df=pd.read_csv("adult.csv")


# In[136]:


df.head()


# In[137]:


df.isnull().sum()


# In[138]:


df.columns


# In[139]:


df.replace("?",pd.NA,inplace=True)


# In[140]:


df.isnull().sum()


# In[141]:


df.dropna(inplace=True)


# In[142]:


df.isnull().sum()


# In[143]:


corr=df.corr()


# In[144]:


sns.heatmap(corr,annot=True,fmt=".2f")


# In[145]:


Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
lower_threshold = Q1 - 1.5 * IQR
upper_threshold = Q3 + 1.5 * IQR
outliers = (df < lower_threshold) | (df > upper_threshold)
print(outliers)


# In[146]:


df1= df[~outliers.any(axis=1)]


# In[147]:


df1.head()


# In[148]:


df1['income_numeric'] = df1['income'].map({'<=50K': 0, '>50K': 1})

plt.figure(figsize=(12, 8))
sns.barplot(data=df1,x='native.country', y='income_numeric', estimator=lambda x: sum(x) / len(x))
plt.xticks(rotation=90) 
plt.xlabel('Native Country')
plt.ylabel('Proportion of Income >50K')
plt.title('Proportion of Income >50K Across Native Countries')
plt.show()


# In[149]:


sns.barplot(data=df1,x='sex', y='income_numeric')
plt.xlabel('gender')
plt.ylabel('Proportion of Income >50K')
plt.title('Proportion of Income >50K by  different genders')
plt.show()


# In[150]:


plt.figure(figsize=(12, 8))
sns.barplot(data=df1,x='education', y='income_numeric')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.xlabel('education')
plt.ylabel('Proportion of Income >50K')
plt.title('Proportion of Income >50Kbased on education')
plt.show()


# In[71]:


sns.barplot(data=df1,x='race', y='income_numeric')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.xlabel('race')
plt.ylabel('Proportion of Income >50K')
plt.title('Proportion of Income >50K by race')
plt.show()


# In[151]:


df1.drop(columns=["fnlwgt"],inplace =True)


# In[152]:


education_mapping = {
    'Preschool': 1,
    '1st-4th': 2,
    '5th-6th': 3,
    '7th-8th': 4,
    '9th': 5,
    '10th': 6,
    '11th': 7,
    '12th': 8,
    'HS-grad': 9,
    'Some-college': 10,
    'Assoc-voc': 11,
    'Assoc-acdm': 12,
    'Bachelors': 13,
    'Masters': 14,
    'Doctorate': 15,
    'Prof-school': 16
}


df1['education_numeric'] = df1['education'].map(education_mapping)


# In[153]:


df1.drop(columns=["education","education.num"],inplace=True)


# In[154]:


df1.head()


# In[155]:


uv=df1["workclass"].unique()
print(uv)


# In[156]:


workclass_map = {
   "Private":0,
    "Self-emp-not-inc":1,
    "State-gov":2,
    "Federal-gov":3,
    "Local-gov":4,
    "Self-emp-inc":5,
    "Without-pay":6
}
df1["workclass"]=df1["workclass"].replace(workclass_map)


# ## df1.head()

# In[157]:


uv=df1["marital.status"].unique()
print(uv)


# In[158]:


marital_status_map= {
   "Divorced":0,
    "Married-civ-spouse":1,
    "Never-married":2,
    "Separated":3,
    "Widowed":4,
    "Married-spouse-absent":5,
    "Married-AF-spouse":6
}
df1["marital.status"]=df1["marital.status"].replace(marital_status_map)


# In[159]:


df1.head()


# In[160]:


uv=df1["occupation"].unique()
print(uv)


# In[161]:


occupation_map= {
   "Handlers-cleaners":0,
    "Prof-specialty":1,
    "Exec-managerial":2,
    "Sales":3,
    "Transport-moving":4,
    "Farming-fishing":5,
    "Machine-op-inspct":6,
    "Tech-support":7,
    "Craft-repair":8,
    "Protective-serv":9,
    "Adm-clerical":10,
    "Other-service":11,
    "Armed-Forces":12,
    "Priv-house-serv":13
}
df1["occupation"]=df1["occupation"].replace(occupation_map)


# In[162]:


df1.head()


# In[163]:


uv=df1["race"].unique()
print(uv)


# In[164]:


race_map = {
   "White":0,
    "Black":1,
    "Asian-Pac-Islander":2,
    "Amer-Indian-Eskimo":3,
    "Other":4,

}
df1["race"]=df1["race"].replace(race_map)


# In[165]:


df1.head()


# In[166]:


uv=df1["sex"].unique()
print(uv)


# In[167]:


sex_map={"Male":0,"Female":1}
df1["sex"]=df1["sex"].replace(sex_map)


# In[168]:


df1.head()


# In[169]:


uv=df1["native.country"].unique()
print(uv)


# In[170]:



country_mapping = {
    'United-States': 1,
    'Cuba': 2,
    'India': 3,
    'Mexico': 4,
    'Puerto-Rico': 5,
    'England': 6,
    'Germany': 7,
    'Iran': 8,
    'Philippines': 9,
    'Poland': 10,
    'Cambodia': 11,
    'Ecuador': 12,
    'Laos': 13,
    'Portugal': 14,
    'El-Salvador': 15,
    'France': 16,
    'Taiwan': 17,
    'Dominican-Republic': 18,
    'Jamaica': 19,
    'Honduras': 20,
    'Haiti': 21,
    'South': 22,
    'Japan': 23,
    'Yugoslavia': 24,
    'Canada': 25,
    'Italy': 26,
    'Peru': 27,
    'China': 28,
    'Outlying-US(Guam-USVI-etc)': 29,
    'Scotland': 30,
    'Trinadad&Tobago': 31,
    'Greece': 32,
    'Nicaragua': 33,
    'Guatemala': 34,
    'Vietnam': 35,
    'Hong': 36,
    'Ireland': 37,
    'Columbia': 38,
    'Thailand': 39,
    'Hungary': 40
}


df1['native.country'] = df1['native.country'].replace(country_mapping)


# In[171]:


df1.head()


# In[172]:


df1.drop(columns=["relationship","income"],inplace=True)


# In[173]:


df1.head()


# In[174]:



income = df1['income_numeric']
training_data = df1.drop(columns=['income_numeric'])


# In[175]:


from sklearn.model_selection import train_test_split 


# In[176]:


X_train, X_test, y_train, y_test = train_test_split(training_data,income,test_size=0.2, random_state=42)


# In[177]:


from sklearn.linear_model import LogisticRegression


# In[178]:


lr= LogisticRegression()


# In[179]:


lr.fit(X_train,y_train)


# In[180]:


from sklearn.metrics import accuracy_score
y_pred = lr.predict(X_test)
y_true = y_test
print(y_pred)

accuracy = accuracy_score(y_true, y_pred)

print("Accuracy:", accuracy)


# In[181]:


X_train.shape


# In[182]:




x = np.array([55, 4, 1, 0, 0, 1, 0, 0, 55, 3, 16])
x_reshaped = np.reshape(x, (1, -1))


# In[183]:


output=lr.predict(x_reshaped)


# In[184]:


if output==0:
    print("<=50k")
else:
    print(">50k")


# In[115]:


import joblib
joblib.dump(lr, 'income_prediction_model.pkl')


# In[ ]:




