
# coding: utf-8

# In[26]:


import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn import preprocessing


# In[10]:


attr = pd.read_csv("data/HR-Employee-Attrition.csv")
attr.head()


# In[16]:


'''
attr_data = attr.loc[:, attr.columns != 'Attrition']
attr_target = attr.loc[:, attr.columns == 'Attrition']
X,y = attr_data, attr_target
X.shape
X_new = SelectKBest(score_func=chi2, k=4).fit(X,y)
X_new.shape
'''


# In[31]:


attr_df = pd.DataFrame(attr)
attr_data = pd.get_dummies(attr_df)
attr_scaled = pd.DataFrame(preprocessing.scale(attr_data),columns=attr_data.columns)
pca = PCA(n_components=2)
attr_pca = pca.fit_transform(attr_scaled)
print(pd.DataFrame(pca.components_, columns=attr_scaled.columns, index= ['PC-1','PC-2']))


# In[32]:


pca.explained_variance_ratio_

