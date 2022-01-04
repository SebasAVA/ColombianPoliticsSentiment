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
import language_tool_python

np.random.seed(2022)

corrector = language_tool_python.LanguageTool('es')

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
#Eliminar usted dentro de los stopwords
#Revisar sinonimos
palabrasVacias_nltk = stopwords.words('spanish')
palabrasVacias_nltk.append("usted")
palabrasVacias_nltk.append("hacer")
palabrasVacias_nltk.append("bien")
palabrasVacias_nltk.append("navidad")
palabrasVacias_nltk.append("jajaja")
def lemmatize(text,nlp):
    # can be parallelized
    doc = nlp(text)
    lemma = [n.lemma_ for n in doc]
    return lemma

def preprocess(text,nlp):
    
    result = []
    for token in gensim.utils.simple_preprocess(text): #  gensim.utils.simple_preprocess tokenizes el texto
        token = ''.join(x for x in token.lower() if x.isalpha())
        if token not in palabrasVacias_nltk and len(token) > 2:
            result.append(token)       
    result = lemmatize(' '.join(result),nlp)
    return result

def remove_words(text):
    # Reemplazar simobolo por palabra para que no me elimine los hashtags
    text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b','',text, flags=re.MULTILINE) #Remove URL
    text = re.sub(r'@\w+','', text) # remove mentions
    return text

def correct_text(text):
    coincidencias = corrector.check(text)
    corrected = corrector.correct(text)
    return corrected

tweets_sp['hashtag'] = tweets_sp['tweet'].apply(lambda x: re.findall(r'#(\w+)', x))


tweets_sp['tweet_c'] = tweets_sp['tweet'].apply(lambda x: remove_words(x))



"""
sample = 6
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
"""

preprocessed_tweets = tweets_sp['tweet_c'].apply(lambda x: preprocess(x, nlp))
preprocessed_tweets.head(10).values

dictionary_spanish = gensim.corpora.Dictionary(preprocessed_tweets)
count = 0
for k, v in dictionary_spanish.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break
    
    
#Distribucion de las palabras para resolver el cutter  
#########################  

dictionary_spanish.filter_extremes(no_below=10, no_above=0.5, keep_n=100000)
dictionary_spanish.most_common()[:10]
dictionary_spanish.most_common()[-10:]
# assume the word 'b' is to be deleted, put its id in a variable

words_to_del = ['ver','mas','cómo','así','cuál','tal','dejar','jajajajajaja','jajajajaja','hacer','afinia','decir','ser','ir','señor','salir','jajajaja','asi','bla','jajajajajajaja','siguemeytesigo','siganme','síganme']

for i in words_to_del:
    del_ids = [k for k,v in dictionary_spanish.items() if v==i]
    dictionary_spanish.filter_tokens(bad_ids=del_ids)


bow_corpus = [dictionary_spanish.doc2bow(doc) for doc in preprocessed_tweets]


#EJEMPLO BOW
"""
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
"""
from gensim import models
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]
"""
print(preprocessed_tweets[0],'\n')
print(bow_corpus[0],'\n')
from pprint import pprint
for doc in corpus_tfidf:
    pprint(doc)
    break
"""

##MODEL

lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=16, id2word=dictionary_spanish, 
                                             random_state=100,
                                             passes=10,
                                             workers=8)
topics = lda_model_tfidf.show_topics()

for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic[:200]))
    
    
sample = np.random.choice(tweets_sp.index)

doc_sample = tweets_sp.iloc[sample].values[0]

print(f'\nOriginal Document: {sample}')
print(doc_sample,'\n')

bow_vector = dictionary_spanish.doc2bow(preprocess(doc_sample,nlp))
tfidf_vector = tfidf[bow_vector]

for index, score in sorted(lda_model_tfidf[tfidf_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model_tfidf.print_topic(index, 5)))
    
    
    
unseen_document = 'Situacion politica corrupcion .'

print(f'\nOriginal Document:')
print(unseen_document,'\n')

bow_vector = dictionary_spanish.doc2bow(preprocess(unseen_document,nlp))
tfidf_vector = tfidf[bow_vector]
for index, score in sorted(lda_model_tfidf[tfidf_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model_tfidf.print_topic(index, 5)))
    
 
    
 
    
 
#COHERENCE
from gensim.models import CoherenceModel
# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model_tfidf, texts=preprocessed_tweets, dictionary=dictionary_spanish, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda,'\n')


# supporting function
def compute_coherence_values(corpus, dictionary, k):
    
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=k, 
                                           random_state=100,
                                           passes=10,
                                           workers=8)
    
    coherence_model_lda = CoherenceModel(model=lda_model, texts=preprocessed_tweets, dictionary=dictionary_spanish, coherence='c_v')
    
    return coherence_model_lda.get_coherence()

import tqdm

# Topics range
min_topics = 2
max_topics = 20
step_size = 1
topics_range = range(min_topics, max_topics, step_size)
# Alpha parameter
#alpha = list(np.arange(0.01, 1, 0.3))
#alpha.append('symmetric')
#alpha.append('asymmetric')
# Beta parameter
#beta = list(np.arange(0.01, 1, 0.3))
#beta.append('symmetric')

model_results = {'Topics': [],
                 'Coherence': []
                }

# Can take a long time to run
if 1 == 1:
    pbar = tqdm.tqdm(total=len(topics_range))
    
    # iterate through number of topics
    for k in topics_range:
        cv = compute_coherence_values(corpus=corpus_tfidf, dictionary=dictionary_spanish, k=k)
        model_results['Topics'].append(k)
        model_results['Coherence'].append(cv)
        pbar.update(1)
        
    pd.DataFrame(model_results).to_csv('lda_tuning_results.csv', index=False)
    pbar.close()
    
    
pd.DataFrame(model_results).sort_values(by='Coherence',ascending=False)
