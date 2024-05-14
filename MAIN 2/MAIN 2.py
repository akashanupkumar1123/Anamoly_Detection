#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest


# In[ ]:





# In[2]:


df = pd.read_csv('salary.csv')
df.head(10)


# In[ ]:





# In[ ]:


model=IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.1),max_features=1.0)
model.fit(df[['salary']])


# In[ ]:





# In[ ]:


df['scores']=model.decision_function(df[['salary']])
df['anomaly']=model.predict(df[['salary']])
df.head(20)


# In[ ]:





# In[ ]:


anomaly=df.loc[df['anomaly']==-1]
anomaly_index=list(anomaly.index)
print(anomaly)


# In[ ]:





# In[ ]:


outliers_counter = len(df[df['salary'] > 99999])
outliers_counter


# In[ ]:





# In[ ]:


print("Accuracy percentage:", 100*list(df['anomaly']).count(-1)/(outliers_counter))

