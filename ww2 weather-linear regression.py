
# coding: utf-8

# In[52]:


import pandas as pd
import numpy as np
import random as rd


# In[6]:


dataset=pd.read_csv("Summary of Weather.csv")
print (dataset)


# In[8]:


import matplotlib.pyplot as plt


# In[9]:


dataset.corr()


# In[10]:


feature=dataset["MinTemp"]
target_var=dataset["MeanTemp"]
plt.scatter(feature,target_var)
plt.show()


# In[14]:


def line(m,c,x):
    return m*x+c
def error(m,x,c,y):
    return np.mean((line(m,c,x)-y)**2)
def derivative_slope(m,x,c,y):
    return 2*np.mean(x*(line(m,x,c)-y))
def derivative_intercept(m,x,c,y):
    return 2*np.mean(line(m,c,x)-y)
def accuracy(error,y):
    return 100-((error/np.mean(y**2))*100)


# In[63]:


m=rd.randint(0,2)
c=rd.randint(0,2)
temp=[]
iteration=1000
learning_rate=.000001
for i in range (0,iteration):
    m=m-learning_rate*derivative_slope(m,feature,c,target_var)
    c=c-learning_rate*derivative_intercept(m,feature,c,target_var)
    temp.append(error(m,feature,c,target_var))
    predicted_answer=line(m,feature,c)
plt.scatter(feature,target_var)
plt.scatter(feature,predicted_answer)

plt.show()
print ("your prediction accuracy",accuracy(error(m,feature,c,target_var),target_var),"percent")
plt.plot(temp)
plt.show()


# In[66]:


m=2.2
c=3

predicted_answer=line(m,feature,c)
plt.scatter(feature,target_var)
plt.scatter(feature,predicted_answer)

plt.show()

