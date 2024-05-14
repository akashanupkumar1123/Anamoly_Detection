#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_excel('ex8data1.xlsx', sheet_name='X', header=None)
df.head()


# In[3]:


plt.figure()
plt.scatter(df[0], df[1])
plt.show()


# In[4]:


m = len(df)


# In[5]:


s = np.sum(df, axis=0)
mu = s/m
mu


# In[ ]:





# In[6]:


vr = np.sum((df - mu)**2, axis=0)
variance = vr/m
variance


# In[ ]:





# In[7]:


var_dia = np.diag(variance)
var_dia


# In[ ]:





# In[8]:


k = len(mu)
X = df - mu
p = 1/((2*np.pi)**(k/2)*(np.linalg.det(var_dia)**0.5))* np.exp(-0.5* np.sum(X @ np.linalg.pinv(var_dia) * X,axis=1))
p


# In[ ]:





# In[9]:


def probability(df):
    s = np.sum(df, axis=0)
    m = len(df)
    mu = s/m
    vr = np.sum((df - mu)**2, axis=0)
    variance = vr/m
    var_dia = np.diag(variance)
    k = len(mu)
    X = df - mu
    p = 1/((2*np.pi)**(k/2)*(np.linalg.det(var_dia)**0.5))* np.exp(-0.5* np.sum(X @ np.linalg.pinv(var_dia) * X,axis=1))
    return p


# In[ ]:





# In[10]:


cvx = pd.read_excel('ex8data1.xlsx', sheet_name='Xval', header=None)
cvx.head()


# In[11]:


cvy = pd.read_excel('ex8data1.xlsx', sheet_name='y', header=None)
cvy.head()


# In[ ]:





# In[20]:


p1 = probability(cvx)


# In[21]:


y = np.array(cvy)
y


# In[ ]:





# In[22]:


p.describe()


# In[ ]:





# In[44]:


def tpfpfn(ep, p):
    tp, fp, fn = 0, 0, 0
    for i in range(len(y)):
        if p[i] <= ep and y[i][0] == 1:
            tp += 1
        elif p[i] <= ep and y[i][0] == 0:
            fp += 1
        elif p[i] > ep and y[i][0] == 1:
            fn += 1
    return tp, fp, fn


# In[45]:


eps = [i for i in p1 if i <= p1.mean()]


# In[46]:


len(eps)


# In[ ]:





# In[47]:


def f1(ep, p):
    tp, fp, fn = tpfpfn(ep)
    prec = tp/(tp + fp)
    rec = tp/(tp + fn)
    f1 = 2*prec*rec/(prec + rec)
    return f1


# In[ ]:





# In[48]:


f = []
for i in eps:
    f.append(f1(i, p1))
f


# In[ ]:





# In[49]:


np.array(f).argmax()


# In[ ]:





# In[ ]:


e = eps[127]
e


# In[ ]:





# In[ ]:


label = []
for i in range(len(df)):
    if p[i] <= e:
        label.append(1)
    else:
        label.append(0)
label


# In[ ]:





# In[ ]:


label = []
for i in range(len(df)):
    if p[i] <= e:
        label.append(1)
    else:
        label.append(0)
label


# In[ ]:




