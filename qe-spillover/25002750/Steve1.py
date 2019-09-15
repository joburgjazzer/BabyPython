#!/usr/bin/env python
# coding: utf-8

# In[62]:


import os
os.getcwd() 


# In[1]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[5]:


df = pd.read_csv("max.csv")


# In[8]:


df.tail()


# In[14]:


df.columns


# In[29]:


head = df.head()
head.shape
head.index


# In[31]:


cols = head.columns
# head[:, 'time']
df.describe()


# In[55]:


fQE = df['QE'].value_counts().sort_index().to_frame('QE')
fQE['pct'] = 100 * fQE['QE'] / fQE['QE'].sum()
fQE


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# START TINA 

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import ProbPlot
plt.style.use('seaborn') # pretty matplotlib plots
plt.rc('font', size=14)
plt.rc('figure', titlesize=18)
plt.rc('axes', labelsize=15)
plt.rc('axes', titlesize=18)


# In[2]:


from sklearn.datasets import load_boston

boston = load_boston()

X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.DataFrame(boston.target)

# generate OLS model
model = sm.OLS(y, sm.add_constant(X))
model_fit = model.fit()

# create dataframe from X, y for easier plot handling
dataframe = pd.concat([X, y], axis=1)

