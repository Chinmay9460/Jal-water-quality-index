#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd

import os 
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[6]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
from sklearn import tree
import warnings
warnings.simplefilter("ignore")
from sklearn.metrics import accuracy_score


# In[9]:


df = pd.read_csv("G:\Projects_data\sample_project1/water_potability.csv")
df.dropna(inplace =True)
df.reset_index(inplace = True, drop = True)
X = df.loc[:, df.columns != 'Potability']
y = df.Potability
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
random_classifier = RandomForestClassifier(n_estimators =50)
random_classifier.fit(X_train,y_train)
y_pred = random_classifier.predict(X_test)
predictions_and_actual = pd.DataFrame(y_pred,y_test)
predictions_and_actual.reset_index(inplace = True)
predictions_and_actual.columns = ['predictions','Original']


# In[10]:


print("Accuracy score: ", accuracy_score(y_test,y_pred))


# In[11]:


df.corr()


# In[12]:


sns.heatmap(df.corr())


# In[ ]:




