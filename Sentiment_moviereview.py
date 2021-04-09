#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
data=pd.read_csv('movie_reviews.csv')
data.head()


# In[17]:


data.shape


# In[47]:


data=data[['id','profileName','text','title','rating']]
data


# In[48]:


data.shape


# In[49]:


data.info()


# In[50]:


data.describe()


# In[51]:


data.isnull().any()


# In[52]:


data.duplicated().sum()


# In[53]:


data['rating'].plot()


# In[54]:


data['rating'].plot(kind='hist')


# # Checking the sentiment on Avenger's movie reviews

# In[55]:


import nltk
nltk.download('vader_lexicon')


# In[56]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia=SentimentIntensityAnalyzer()


# In[57]:


review1='This was an interesting movie'
sia.polarity_scores(review1)


# In[58]:


review2='This would be the most horrible movie to watch with your family'
sia.polarity_scores(review2)


# In[59]:


data['text']


# In[60]:


data['text'] = data['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
data['text']


# In[61]:


#Remve stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')
data['text'] = data['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
data['text']


# In[62]:


#Stemming
from nltk.stem import PorterStemmer
st = PorterStemmer()
data['text'] = data['text'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
data['text']


# In[63]:


#Sentiment score
data['scores'] = data['text'].apply(lambda text: sia.polarity_scores(text))
data['scores']


# In[64]:


data.columns


# In[65]:


data.columns


# In[66]:


data.head()


# In[67]:


#Compound score
data['compound']  = data['scores'].apply(lambda score_dict: score_dict['compound'])
data['compound']


# In[68]:


data['label'] = data['compound'].apply(lambda x: 'positive' if x >=0.05 else('negative' if x<= -0.05 else 'neutral'))
data['label']


# In[69]:


data.head()


# In[70]:


data['label'].value_counts()


# In[72]:


data['label'].value_counts().plot(kind='bar')


# In[90]:


col=['label']
from sklearn import preprocessing
number = preprocessing.LabelEncoder()
for i in col:
    train[i] = number.fit_transform(train[i])
    test[i] = number.fit_transform(test[i])


# In[73]:


from sklearn.model_selection import train_test_split
train,test=train_test_split(data,test_size=0.3)


# In[75]:


x_train=train.iloc[:,6]
x_train


# In[89]:


x_test=test.iloc[:,6]
x_test


# In[91]:


y_train=train.iloc[:,7]
y_train


# In[92]:


y_test=test.iloc[:,7]
y_test


# In[93]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train.values.reshape(-1,1),y_train)


# In[94]:


model.predict(x_test.values.reshape(-1,1))


# In[99]:


y_test


# In[97]:


from sklearn import metrics
print("Train accuracy=",metrics.accuracy_score(y_train,model.predict(x_train.values.reshape(-1,1))))


# In[98]:


from sklearn import metrics
print("Test accuracy=",metrics.accuracy_score(y_test,model.predict(x_test.values.reshape(-1,1))))


# In[ ]:




