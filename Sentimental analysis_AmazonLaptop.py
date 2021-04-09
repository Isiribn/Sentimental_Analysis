#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import nltk


# In[3]:


#VADER(Valence Aware Dictionary and sEntiment Reaoning) used for analyzing the sentiment 
#of social media whether it is positive, negative or neutral
nltk.download('vader_lexicon')


# In[9]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia=SentimentIntensityAnalyzer()


# In[5]:


review1='This is an amazing product'
sia.polarity_scores(review1)


# In[6]:


review2='This is the worst product ever'
sia.polarity_scores(review2)


# # Checking the sentiment for Amazon review

# In[8]:


data=pd.read_csv('reviews_prod.csv')
data.head()


# In[11]:


data.info()


# In[12]:


data.shape


# In[13]:


data=data[['id','profileName','text','title','rating']]
data.head()


# In[14]:


data.shape


# In[15]:


data.info()


# In[16]:


data.isnull().any()


# In[17]:


data.duplicated().any()


# In[18]:


data['rating'].value_counts().plot(kind='bar')


# # Data processing for textual variables

# In[19]:


#Converting to lowercase
data['text']


# In[20]:


data['text'] = data['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
data['text']


# In[21]:


#Remve stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')
data['text'] = data['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
data['text']


# In[22]:


#Stemming
from nltk.stem import PorterStemmer
st = PorterStemmer()
data['text'] = data['text'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
data['text']


# In[23]:


#Sentiment score
data['scores'] = data['text'].apply(lambda text: sia.polarity_scores(text))
data['scores']


# In[24]:


data.columns


# In[25]:


#or Sentiment score, where the first part is how much is positive or negative and second part is the subjective text
from textblob import TextBlob
def senti(x):
    return TextBlob(x).sentiment  

data['sentiment'] = data['text'].apply(senti)
data.sentiment


# In[29]:


def polar(data):
    data=sia.polarity_scores(data.text)#['compound']
    return data
data['sentiment']=data.apply(polar,axis=1)
data['sentiment']


# In[30]:


#Compound score
data['compound']  = data['scores'].apply(lambda score_dict: score_dict['compound'])
data['compound']


# In[31]:


data.columns


# In[32]:


data['label'] = data['compound'].apply(lambda x: 'positive' if x >=0.05 else('negative' if x<= -0.05 else 'neutral'))
data['label']


# In[33]:


data['label'].value_counts()


# In[34]:


data['label'].value_counts().plot(kind='bar')


# In[35]:


data.columns


# In[36]:


data.head()


# In[37]:


from sklearn.model_selection import train_test_split
train,test=train_test_split(data,test_size=0.3)


# In[38]:


col=['label']
from sklearn import preprocessing
number = preprocessing.LabelEncoder()
for i in col:
    train[i] = number.fit_transform(train[i])
    test[i] = number.fit_transform(test[i])


# In[39]:


x_train=train.iloc[:,7]
x_train


# In[40]:


x_test=test.iloc[:,7]
x_test


# In[41]:


y_train=train.iloc[:,8]
y_train


# In[42]:


y_test=test.iloc[:,8]
y_test


# In[44]:


from sklearn.linear_model import LogisticRegression
log=LogisticRegression()
log.fit(x_train.values.reshape(-1,1),y_train)


# In[45]:


log.predict(x_test.values.reshape(-1,1))


# In[46]:


y_test


# In[47]:


from sklearn import metrics
print("Train accuracy=",metrics.accuracy_score(y_train,log.predict(x_train.values.reshape(-1,1))))


# In[48]:


from sklearn import metrics
print("Test accuracy=",metrics.accuracy_score(y_test,log.predict(x_test.values.reshape(-1,1))))


# In[ ]:




