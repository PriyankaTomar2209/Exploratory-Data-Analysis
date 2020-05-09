#!/usr/bin/env python
# coding: utf-8

# In[32]:


# Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import matplotlib.pyplot as plt
import seaborn as sns


# In[16]:



rating = pd.read_csv("ratings.dat", sep='::', names=["useID", "movieID","rating", "timeStamp"], engine= 'python')
rating.head(5)


# In[17]:


user = pd.read_csv("users.dat", sep='::', names=["useID", "gender","age", "occupation","zipcode"], engine= 'python', index_col= False)
user.head(5)


# In[18]:


movie = pd.read_csv("movies.dat", sep='::', names=["movieID", "title","genres"], engine= 'python', index_col= False)
movie.head(5)


# In[24]:


left_merge = pd.merge( rating, user , on='useID' , how='left')
left_merge.head(5)


# In[102]:


Master_data = pd.merge( left_merge , movie , on ='movieID',  how= 'left' )
Master_data[['movieID','title','useID','age','gender','occupation','rating','genres']].head(5)


# In[49]:


import seaborn as sns
user.groupby('age')['useID'].count()


# # Maximum users are from age group 25 

# In[50]:


import seaborn as sns
fig1 = sns.countplot(data=user, x="age")


# In[51]:


user.groupby('gender')['useID'].count()


# In[52]:


import seaborn as sns
fig2 = sns.countplot(data=user, x="gender")


# # MAXIMUM USERS ARE "MALE"

# In[54]:


user.groupby('occupation')['useID'].count()


# In[62]:


import seaborn as sns
fig3 = sns.countplot(data=user, x="occupation")
#fig3.xlabel("number of user")


# # MAXIMUM USERS ARE STUDENTS

# In[61]:


Master_data[Master_data.title == 'Toy Story (1995)'].groupby('rating')['useID'].count()


# In[69]:


import seaborn as sns
fig4 = sns.countplot(data=Master_data, x="rating")
#fig3.xlabel("number of user")


# # MAXIMUM USERS GAVE RATING 4

# In[75]:


Master_data[Master_data.title == 'Toy Story (1995)'].groupby('rating')['useID'].count().plot(kind = "bar" , figsize = (8,8))


# 
# # TOY STORY GOT RATING 5 MAXIMUM TIMES 

# In[72]:


Master_data.groupby('movieID')['rating'].count().sort_values(ascending = False)[:25]


# In[73]:


Master_data.groupby('movieID')['rating'].count().nlargest(25)


# In[74]:


Master_data.groupby('movieID')['rating'].count().sort_values(ascending = False)[:25].plot(kind='bar', figsize = (8,8))


# # MOVIE ID 2858 HAVE MAXIMUM NUMBER OF RATING

# In[80]:


user_id = Master_data[Master_data.useID == 2696].groupby('rating')['movieID'].count().plot(kind = 'pie' , figsize =(9,9))


# # USER ID 2696 HAVE GIVEN RATING 4 MAXIMUM TIMES

# In[103]:


Master_data.info()


# In[104]:


Master_data = Master_data.drop(Master_data.columns[[3,7]],axis=1)


# In[126]:

from sklearn.feature_extraction.text import CountVectorizer
Master_data = Master_data.drop(Master_data.columns[[3, 7]], axis=1) 
cv = CountVectorizer()
t = pd.DataFrame(cv.fit_transform(Master_data.genres.fillna('').str.replace(r'\|', ' ')).A,
                    columns=cv.get_feature_names(),
                  index=Master_data.index).add_prefix('genres_')
				  
				  
#In[128]

Master_data = Master_data.join(t)
dummy = pd.get_dummies(Master_data['gender'])
Master_data = Master_data.join(dummy)
dummy1 = pd.get_dummies(Master_data['age'])
Master_data = Master_data.join(dummy1)
Master_data = Master_data.drop(Master_data.columns[[3,4,7]], axis=1)


#In[129]

Master_data.head(10)















