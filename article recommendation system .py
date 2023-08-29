#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[7]:


article_df=pd.read_csv("shared_articles.csv")
inter_df=pd.read_csv("users_interactions.csv")


# In[8]:


article_df.head()


# In[4]:


inter_df.head()


# In[10]:


article_df=article_df[article_df["eventType"]=="CONTENT SHARED"]


# In[11]:


article_df.head()


# In[13]:


article_df["authorPersonId"].unique()


# In[14]:


len(article_df["authorPersonId"].unique())


# In[16]:


print(article_df['lang'].isna().sum(axis=0))


# In[19]:


article_df=article_df[article_df["lang"]=="en"]


# In[20]:


article_df.shape


# In[23]:


article_df=pd.DataFrame(article_df,columns=["contentId","authorPersonId","content","title","text"])


# In[24]:


article_df.shape


# In[25]:


def soup(x):
    soup="".join(x["text"])
    return soup

    


# In[27]:


article_df["soup"]=article_df.apply(soup,axis=1)


# In[30]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[32]:


tfid= TfidfVectorizer(stop_words="english")


# In[33]:


tfid_matrix=tfid.fit_transform(article_df["text"])


# In[34]:


tfid_matrix.shape


# In[36]:


from sklearn.metrics.pairwise import cosine_similarity


# In[37]:


cos=cosine_similarity(tfid_matrix,tfid_matrix,True)


# In[39]:


cos.shape


# In[40]:


cos


# In[41]:


meta_data=article_df.reset_index()


# In[43]:


indices = pd.Series(meta_data.index, index=meta_data['title']).drop_duplicates()


# In[54]:


def get_recommendations(title):
    indice=indices
    cosine=cos
    data=meta_data
    idx = indice[title]
    sim_scores = list(enumerate(cosine[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return data['title'].iloc[movie_indices]


# In[55]:


print(get_recommendations('Intel\'s internal IoT platform for real-time enterprise analytics'))


# In[61]:


get_recommendations("Why Decentralized Conglomerates Will Scale Better than Bitcoin - Interview with OpenLedger CEO - Bitcoin News")


# In[ ]:




