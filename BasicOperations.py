#required imports
from __future__ import print_function
import spacy
import textacy
import os
import re
import nltk
import gensim
import pyLDAvis.gensim
# import string
# import pandas as pd
# import numpy as np

from gensim import corpora, models, similarities
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
# import ends

# reading abstracts from Abstracts folder( 7 at a time)
new_list = []
for root, dirs, files in os.walk("/Users/GowthamYesG/Desktop/KDM/Project/Abstracts"):
    print(files)
    for file in files:
        if file.endswith('.txt'):
            with open(os.path.join(root, file), 'r') as f:
                text = f.read()
                new_list.append(text)

#Stored all the abstracts data in new_list

#Cleaning data
clean_data = ''.join(str(e) for e in new_list)

#Sentence tokenization
clean_data_sent=sent_tokenize(clean_data)
print("Sentence Tokenizer results:")
print(clean_data_sent)

#Word tokenization
clean_data_word=word_tokenize(clean_data)
print("Word tokenizer results:")
print(clean_data_word)

# Stemming the words
stemmer = PorterStemmer()
stemming = [stemmer.stem(str(e)) for e in clean_data_word]
stemming = [word for word in stemming if len(word) > 1] # removing single letter words
print("Stemming words:")
print(stemming)

#Lemmitizing words
lemmatizer = WordNetLemmatizer()
lemmatizing = [lemmatizer.lemmatize(str(e)) for e in clean_data_word]
print("Lemmitization output is : ");
print(lemmatizing);

#Generating Triplets - start
nlp = spacy.load("en_core_web_sm")
tuples_list = []
for sentence in clean_data_sent:
    val = nlp(sentence)
    tuples = textacy.extract.subject_verb_object_triples(val)
    if tuples:
        tuples_to_list = list(tuples)
        tuples_list.append(tuples_to_list)

#Removing empty tuples in the list
final=[]
def Remove(tuples):
    final = [t for t in tuples if t]
    return final
triplets= Remove(tuples_list)
triplets = ''.join(str(e) for e in triplets)
triplets = ''.join(str(e) for e in triplets)
print("Triplets:", triplets)

#Generating Triplets - end

#LDA - Topic Modelling

#Get all the stop words from nlkt which are in english language
stop_words = stopwords.words('english')
stop_words.extend(['news', 'say','use', 'not', 'would', 'say', 'could', '(', ')', ',', '%', '_', 'be', 'know', 'good', 'go', 'get', 'do','took','time','year',
'done', 'try', 'many', 'some','nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line','even', 'also', 'may', 'take', 'come', 'new','said', 'like','people'])

# add custom stop words
def remove_stop_words(text):
     return [word for word in text if word not in stop_words]
def initial_clean(text):
    """
    Function to clean text-remove punctuations, lowercase text etc.
    """
    text = re.sub("[^a-zA-Z ]", "", text) #just pick words lower or upper case
    text = text.lower()  # change to lower case text
    text = nltk.word_tokenize(text)
    return (text)

#Create a Gensim dictionary from the tokenized data
tokenized_abstracts = remove_stop_words(initial_clean(clean_data))
tokenized = tokenized_abstracts
tokenized = [d.split() for d in tokenized]

#Creating term dictionary of corpus, where each unique term is assigned an index.
dictionary = corpora.Dictionary(tokenized)
#Filter terms which occurs in less than 1 abstract and more than 80% of the abstract.
dictionary.filter_extremes(no_below=1, no_above=0.8)
#convert the dictionary to a bag of words corpus
corpus = [dictionary.doc2bow(tokens) for tokens in tokenized]
#print("corpus")
#print(corpus[:1])
print([[(dictionary[id], freq) for id, freq in cp] for cp in corpus[:1]])
tokenized_abstracts = remove_stop_words(initial_clean(clean_data))

#LDA - starts
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = 6, id2word=dictionary, passes=15)
#saving the model
ldamodel.save('model_combined.gensim')
topics = ldamodel.print_topics(num_words=4)
print('\n')
print("Now printing the topics and their composition")
print("This output shows the Topic-Words matrix for the 6 topics created and the 4 words within each topic")
for topic in topics:
   print(topic)

#finding the similarity of the first abstracts with topics
get_document_topics = ldamodel.get_document_topics(corpus[0])
print('\n')
print("The similarity of this abstracts with the topics and respective similarity score are ")
print(get_document_topics)

#visualizing topics
lda_viz = gensim.models.ldamodel.LdaModel.load('model_combined.gensim')
lda_display = pyLDAvis.gensim.prepare(lda_viz, corpus, dictionary, sort_topics=True)
pyLDAvis.show(lda_display)
#LDA - ends
with open("output.txt", "a") as text_file:
# f = open('output.txt','w')
    text_file.write(clean_data_sent + "/n" + clean_data_word)
# f.write(stemming)
# f.write(lemmatizing)
# f.write("Triplets")
# f.write(triplets)
# f.close()