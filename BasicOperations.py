from __future__ import print_function
import spacy
import textacy
import os
import pandas as pd
import re
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from gensim import corpora, models, similarities
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd
import numpy as np

from nltk.tokenize import sent_tokenize, word_tokenize
new_list = []
for root, dirs, files in os.walk("/Users/GowthamYesG/Desktop/KDM/Project/Abstracts"):
    print(files)
    for file in files:
        if file.endswith('.txt'):
            with open(os.path.join(root, file), 'r') as f:
                text = f.read()
                new_list.append(text)


clean_data = ''.join(str(e) for e in new_list)
clean_data_sent=sent_tokenize(clean_data)
print("Sentence Tokenizer results:")
print(clean_data_sent)

clean_data_word=word_tokenize(clean_data)
print("Word tokenizer results")

print(clean_data_word)
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

# def initial_clean(text):
#     """
#     Function to clean text-remove punctuations, lowercase text etc.
#     """
#     text = re.sub("[^a-zA-Z ]", "", text)
#     text = text.lower()  # lower case text
#     text = nltk.word_tokenize(text)
#     return (text)
#
# stop_words = stopwords.words('english')
# stop_words.extend(
#     ['news', 'say', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'took', 'time',
#      'year',
#      'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make',
#      'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come', 'new', 'said',
#      'like', 'people'])
#
# def remove_stop_words(text):
#     return [word for word in text if word not in stop_words]
#
# stemmer = PorterStemmer()
#
# def stem_words(text):
#     """
#     Function to stem words
#     """
#     try:
#         text = [stemmer.stem(word) for word in text]
#         text = [word for word in text if len(word) > 1]  # no single letter words
#     except IndexError:
#         pass
#
#     return text
#
# def apply_all(text):
#     """
#     This function applies all the functions above into one
#     """
#     return stem_words(remove_stop_words(initial_clean(text)))



f = open('output.txt','w')
f.write(triplets)
f.close()