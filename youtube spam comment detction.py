#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as  np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB


# In[2]:


df=pd.read_csv("Youtube01-psy.csv")


# In[3]:


print(df.head(5))


# In[4]:


df=df[["CONTENT","CLASS"]]


# In[5]:


print(df.head(5))


# In[6]:


df["CLASS"]=df["CLASS"].map({0: "not Spam",1: "spam"})


# In[7]:


print(df.head(5))


# In[12]:


x=np.array(df["CONTENT"])
y=np.array(df["CLASS"])
c=CountVectorizer()
x=c.fit_transform(x)
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=44)
model=BernoulliNB()
model.fit(xtrain,ytrain)
print(model.score(xtest,ytest))


# In[16]:


sample="i turned it on mute as soon is i came on i just wanted to check the  views...ï»¿"
df=c.transform([sample]).toarray()
print(model.predict(df))


# In[15]:


sample="watch?v=vtaRGgvGtWQ   Check this out .ï»¿"
df=c.transform([sample]).toarray()
print(model.predict(df))


# In[ ]:




