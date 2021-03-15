#!/usr/bin/env python
# coding: utf-8

# In[38]:


import numpy as nb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data=pd.read_csv('E:\Machinfy\secion 16\census.csv - census.csv.csv',sep=',',encoding='utf-8')


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.describe()


# In[7]:


data.hist(figsize=(8,8))


# In[8]:


data.corr()


# In[11]:


data['workclass'].value_counts()


# In[12]:


data['education_level'].value_counts()


# In[13]:


data['marital-status'].value_counts()


# In[14]:


data['occupation'].value_counts()


# In[15]:


data['relationship'].value_counts()


# In[16]:


data['race'].value_counts()


# In[17]:


data['sex'].value_counts()


# In[18]:


data['native-country'].value_counts()


# In[19]:


data['income'].value_counts()


# In[20]:


from sklearn.preprocessing import LabelEncoder


# In[22]:


workclass_le=LabelEncoder()
education_le=LabelEncoder()
marital_le=LabelEncoder()
occupation_le=LabelEncoder()
relationship_le=LabelEncoder()
race_le=LabelEncoder()
sex_le=LabelEncoder()
country_le=LabelEncoder()
income_le=LabelEncoder()


# In[23]:


data['workclass']=workclass_le.fit_transform(data['workclass'])
data['education_level']=education_le.fit_transform(data['education_level'])
data['marital-status']=marital_le.fit_transform(data['marital-status'])
data['occupation']=occupation_le.fit_transform(data['occupation'])
data['relationship']=relationship_le.fit_transform(data['relationship'])
data['race']=race_le.fit_transform(data['race'])
data['sex']=sex_le.fit_transform(data['sex'])
data['native-country']=country_le.fit_transform(data['native-country'])
data['income']=income_le.fit_transform(data['income'])


# In[25]:


data['workclass'].value_counts()


# In[26]:


data.head()


# In[39]:


plt.figure(figsize=(15,10))
sns.boxplot(data=data)


# In[40]:


data['capital-gain']=(data['capital-gain']-data['capital-gain'].min())/(data['capital-gain'].max()-data['capital-gain'].min())
data['capital-loss']=(data['capital-loss']-data['capital-loss'].min())/(data['capital-loss'].max()-data['capital-loss'].min())


# In[47]:


plt.figure(figsize=(25,25))
sns.boxplot(data=data)


# In[46]:


plt.figure(figsize=(25,25))
sns.heatmap(data=data.corr(),cbar=True,annot=True,cmap='coolwarm').set_title('correlation')


# In[45]:


data['hours-per-week']=(data['hours-per-week']-data['hours-per-week'].min())/(data['hours-per-week'].max()-data['hours-per-week'].min())


# In[52]:


from sklearn.model_selection import train_test_split


# In[53]:


X = data.drop('income' , axis= 1).values 
Y = data['income'].values


# In[54]:


x_train , x_test , y_train , y_test = train_test_split(X,Y, test_size= 0.30, random_state =30)


# In[55]:


x_train.shape


# In[56]:


x_test.shape


# In[57]:


from sklearn.svm import SVC


# In[58]:


svm=SVC(kernel='rbf')


# In[59]:


svm.fit(x_train,y_train)


# In[60]:


svm.score(x_train,y_train)


# In[61]:


svm.score(x_test,y_test)


# In[62]:


from sklearn.naive_bayes import GaussianNB


# In[63]:


gnb = GaussianNB()


# In[64]:


gnb.fit(x_train,y_train)


# In[65]:


gnb.score(x_train,y_train)


# In[66]:


gnb.score(x_test,y_test)


# In[ ]:




