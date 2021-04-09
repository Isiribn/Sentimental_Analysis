#!/usr/bin/env python
# coding: utf-8

# In[8]:


#Python library for creating image wordclouds
get_ipython().system('pip install wordcloud')


# In[85]:


#Tweets about Donald Trump
import pandas as pd
data=pd.read_csv('tweets.csv',delimiter=',', header=0)
data.head()


# In[86]:


data=data.drop('Unnamed: 3',axis=1)
data.head()


# In[87]:


data.shape


# In[50]:


data['sentiment'].value_counts()


# In[51]:


data.isnull().any()


# In[52]:


data.duplicated().sum()


# In[53]:


data['sentiment'].value_counts().plot(kind='bar')


# In[ ]:


#Top 25 frequetly occuring words


# In[58]:


from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

cv = CountVectorizer(stop_words = 'english')
words = cv.fit_transform(data.text)

sum_words = words.sum(axis=0)

words_freq = [(word, sum_words[0, i]) for word, i in cv.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)

frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])

frequency.head(30).plot(x='word', y='freq', kind='bar', figsize=(15, 7), color = 'blue')
plt.title("Most Frequently Occuring Words - Top 25")


# In[ ]:


#Wordcloud of Most Frequently occured words


# In[59]:


from wordcloud import WordCloud

wordcloud = WordCloud(background_color = 'pink', width = 1000, height = 1000).generate_from_frequencies(dict(words_freq))

plt.figure(figsize=(10,8))
plt.imshow(wordcloud)
plt.title("WordCloud - Vocabulary from Reviews", fontsize = 22)


# In[ ]:


#Wordcloud of neutral words


# In[60]:


normal_words =' '.join([text for text in data['text'][data['sentiment'] == 'neutral']])

wordcloud = WordCloud(width=800, height=500, random_state = 0, max_font_size = 110).generate(normal_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title('The Neutral Words')
plt.show()


# In[ ]:


#Wordcloud of Postive words


# In[63]:


normal_words =' '.join([text for text in data['text'][data['sentiment'] == 'positive']])

wordcloud = WordCloud(width=800, height=500, random_state = 0, max_font_size = 110).generate(normal_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title('The Positive Words')
plt.show()


# In[ ]:


#Wordcloud on Negative words


# In[64]:


normal_words =' '.join([text for text in data['text'][data['sentiment'] == 'negative']])

wordcloud = WordCloud(width=800, height=500, random_state = 0, max_font_size = 110).generate(normal_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title('The Negative Words')
plt.show()


# In[ ]:


#Collecting hashtags


# In[68]:


import re
def hashtag_extract(x):
    hashtags = []
    
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)

    return hashtags


# In[69]:


HT_positive = hashtag_extract(data['text'][data['sentiment'] == 'positive'])
HT_neutral = hashtag_extract(data['text'][data['sentiment'] == 'neutral'])
HT_negative = hashtag_extract(data['text'][data['sentiment'] == 'negative'])


# In[70]:


HT_positive = sum(HT_positive,[])
HT_negative = sum(HT_negative,[])
HT_neutral = sum(HT_neutral,[])


# In[76]:


import nltk
import seaborn as sn
a = nltk.FreqDist(HT_positive)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})

# selecting top 5 most frequent hashtags     
d = d.nlargest(columns="Count", n = 5) 
plt.figure(figsize=(16,5))
ax = sn.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()


# In[78]:


a = nltk.FreqDist(HT_neutral)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})

# selecting top 5 most frequent hashtags     
d = d.nlargest(columns="Count", n = 5) 
plt.figure(figsize=(16,5))
ax = sn.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()


# In[79]:


a = nltk.FreqDist(HT_negative)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})

# selecting top 5 most frequent hashtags     
d = d.nlargest(columns="Count", n = 5) 
plt.figure(figsize=(16,5))
ax = sn.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()


# In[80]:


HT_negative


# # Data processing for tweets

# In[88]:


data['text']


# In[89]:


import re
for i in range(len(data)):
    txt = data.loc[i]["text"]
    txt=re.sub(r'@[A-Z0-9a-z_:]+','',txt)#replace username-tags
    txt=re.sub(r'^[RT]+','',txt)#replace RT-tags
    txt = re.sub('https?://[A-Za-z0-9./]+','',txt)#replace URLs
    txt=re.sub("[^a-zA-Z]", " ",txt)#replace hashtags
    data.at[i,"text"]=txt


# In[90]:


data['text']


# In[91]:


#Converting to lowercase
data['text'] = data['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
data['text']


# In[92]:


#Remve stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')
data['text'] = data['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
data['text']


# In[93]:


#Stemming
from nltk.stem import PorterStemmer
st = PorterStemmer()
data['text'] = data['text'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
data['text']


# In[95]:


#Sentiment score
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia=SentimentIntensityAnalyzer()
data['scores'] = data['text'].apply(lambda text: sia.polarity_scores(text))
data['scores']


# In[96]:


#Compound score
data['compound']  = data['scores'].apply(lambda score_dict: score_dict['compound'])
data['compound']


# In[97]:


data['label'] = data['compound'].apply(lambda x: 'positive' if x >=0.05 else('negative' if x<= -0.05 else 'neutral'))
data['label']


# In[98]:


data.head()


# In[99]:


data.shape


# In[100]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
print(classification_report(data['sentiment'],data['label']))


# In[101]:


print(confusion_matrix(data['sentiment'],data['label']))


# In[102]:


accuracy_score(data['sentiment'],data['label'])


# In[ ]:




