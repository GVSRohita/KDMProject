from __future__ import print_function
import spacy
import textacy
import os
import pandas as pd
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