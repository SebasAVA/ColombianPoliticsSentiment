# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 13:27:37 2022

@author: sebas
"""
#Libraries to connect, preprocess and determine de LDA
from sshtunnel import SSHTunnelForwarder
from nltk.stem import PorterStemmer 
import spacy
import psycopg2
import gensim
from nltk.corpus import stopwords
import re
import numpy as np
import pandas as pd

nlp = spacy.load('es_core_news_lg')
#Information to connect to the database
REMOTE_PASSWORD ="colombi@2021Prj"
REMOTE_HOST = "147.182.253.185"
REMOTE_SSH_PORT = 22
REMOTE_USERNAME = "root"


PORT=5432
#Connection to the database via SSH

server = SSHTunnelForwarder((REMOTE_HOST, REMOTE_SSH_PORT),
         ssh_username=REMOTE_USERNAME,
         ssh_password=REMOTE_PASSWORD,
         remote_bind_address=('localhost', PORT),
         local_bind_address=('localhost', PORT))
server.start()

DATABASE = "tweetproject"
USER = "postgres"
PWD = "padova2021"

conn = psycopg2.connect(
    database=DATABASE,
    user=USER,
    host=server.local_bind_host,
    port=server.local_bind_port,
    password=PWD)


cur = conn.cursor()
#Query of the DB to the table that contains the tweets 
cur.execute("SELECT full_text FROM public.tweet where lang = 'es';")
data = cur.fetchall()
server.close()

#Convert the query into a dataframe of pandas
tweets_sp = pd.DataFrame(data)
tweets_sp = tweets_sp.rename(columns={0: 'tweet'})

#Define functions to pre process the data lemmatize, remove url, 

palabrasVacias_nltk = stopwords.words('spanish')
def lemmatize(text,nlp):
    # can be parallelized
    doc = nlp(text)
    lemma = [n.lemma_ for n in doc]
    
        
    return lemma

def preprocess(text,nlp):
    
    result = []
    
    for token in gensim.utils.simple_preprocess(text): #  gensim.utils.simple_preprocess tokenizer
        token = ''.join(x for x in token.lower() if x.isalpha())
        if token not in palabrasVacias_nltk and len(token) > 2:
            result.append(token)
        #result = lemmatize(' '.join(result),nlp)
    return result

def remove_words(text):
    text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b','',text, flags=re.MULTILINE) #Remove URL
    #text = re.sub(r'@\w+','', text) # remove mentions
    return text

tweets_sp['hashtag'] = tweets_sp['tweet'].apply(lambda x: re.findall(r'#(\w+)', x))

sample = np.random.choice(tweets_sp.index)
doc_sample = tweets_sp.iloc[sample].values[0]
print(f'Original Document: {sample}')
words = []
for word in doc_sample.split(' '):
    words.append(word)



doc_sample =remove_words(doc_sample)
print(words)
print('Lemmatized text')
print(lemmatize(doc_sample,nlp))
print('Clean text')
print(preprocess(doc_sample,nlp))


preprocessed_tweets = tweets_sp['tweet'].apply(lambda x: preprocess(remove_words(x), nlp))
preprocessed_tweets.head(10).values

dictionary_spanish = gensim.corpora.Dictionary(preprocessed_tweets)
count = 0
for k, v in dictionary_spanish.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break
    
dictionary_spanish.most_common()[:10]
dictionary_spanish.most_common()[-10:]
dictionary_spanish.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
dictionary_spanish.most_common()[:10]
dictionary_spanish.most_common()[-10:]


bow_corpus = [dictionary_spanish.doc2bow(doc) for doc in preprocessed_tweets]

sample = np.random.choice(tweets_sp.index)

doc_sample = tweets_sp.iloc[sample].values[0]
print(f'\nOriginal Document: {sample}')
print(doc_sample,'\n')

print('Bag of Words (BoW):\n')
print(bow_corpus[sample],'\n')

bow_doc_4310 = bow_corpus[sample]
for i in range(len(bow_doc_4310)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_4310[i][0], 
                                               dictionary_spanish[bow_doc_4310[i][0]], 
bow_doc_4310[i][1]))
    
from gensim import models
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

print(preprocessed_tweets[0],'\n')
print(bow_corpus[0],'\n')
from pprint import pprint
for doc in corpus_tfidf:
    pprint(doc)
    break

lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=5, id2word=dictionary_spanish, passes=4, workers=8)


for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))
    
    
sample = np.random.choice(tweets_sp.index)

doc_sample = tweets_sp.iloc[sample].values[0]

print(f'\nOriginal Document: {sample}')
print(doc_sample,'\n')

bow_vector = dictionary_spanish.doc2bow(preprocess(doc_sample,nlp))
tfidf_vector = tfidf[bow_vector]

for index, score in sorted(lda_model_tfidf[tfidf_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model_tfidf.print_topic(index, 5)))
    
    
    
    
unseen_document = 'Hay demasiada inseguridad en todas las ciudades de Colombia.'

print(f'\nOriginal Document:')
print(unseen_document,'\n')

bow_vector = dictionary_spanish.doc2bow(preprocess(unseen_document,nlp))
tfidf_vector = tfidf[bow_vector]
for index, score in sorted(lda_model_tfidf[tfidf_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model_tfidf.print_topic(index, 5)))