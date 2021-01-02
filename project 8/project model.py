#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.simplefilter('ignore')


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 
# pickle is used to save the model created by us


# In[3]:


df = pd.read_csv(r"C:\Users\HP\Documents\aiml assignments\heroku\hiring.csv")
df.head()


# In[4]:


df.isna().sum()


# In[5]:


# experience

df['experience'].fillna(0, inplace=True)


# In[6]:


df.isna().sum()


# In[7]:


df['test_score'].mean()


# In[8]:


df['test_score'].fillna(df['test_score'].mean(), inplace=True)


# In[9]:


df.isna().sum()


# In[10]:


#Dataset is Clean now.


# In[11]:


df.head()


# In[12]:


X = df.iloc[:,:-1]


# In[13]:


X.head()


# In[14]:


X.shape


# In[15]:


X.experience


# In[16]:


# Convert text in the cols to integer values

def conv(x):
    dict = {'two':2, 'three':3, 'five':5, 'seven':7, 'ten':10, 0:0, 'eleven':11 }
    return dict[x]


# In[17]:


X['experience'] = X['experience'].apply(lambda x: conv(x))


# In[18]:


X.head()


# In[19]:


X.info()


# In[20]:


#X is ready.


# In[21]:


y = df.iloc[:,-1]
y


# In[22]:


# Modeling

from sklearn.linear_model import LinearRegression
lr = LinearRegression()


# In[23]:


# Fit the model
lr.fit(X, y)


# In[24]:


# Prediction Phase
y_pred=lr.predict(X)
y_pred


# In[25]:


y


# In[26]:


from sklearn.metrics import r2_score
r2_score(y_pred, y)


# In[27]:


X


# In[28]:


lr.predict([[3,9,7]])


# In[29]:


lr.predict([[10,10,10]])


# In[30]:


lr.predict([[10,2,3]])


# # Model Deployment

# In[32]:




import pickle

pickle.dump(lr,open('model.py','wb'))
#dump this model by nmae model.py
#wb=write bytes


# In[33]:


#lets now try to load the same moddel by reading it from the sys n using it fro protection

model2 =pickle.load(open('model.py','rb'))


# In[34]:


model2.predict([[3,9,7]])


# In[35]:


model2.predict([[10,10,10]])


# In[36]:


model2.predict([[10,2,3]])

