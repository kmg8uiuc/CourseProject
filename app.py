import streamlit as st
import pandas as pd
import numpy as np
import spacy
import gensim
import gensim.corpora as corpora
from gensim import models

nlp = spacy.load("en_core_web_sm")

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""

st.header('Topic Modeling for Meeting Transcripts',)

def preprocess(ECallTxt):
        # Clean text
        ECallTxt = ECallTxt.strip()  # Remove white space at the beginning and end
        ECallTxt = ECallTxt.replace('\n', ' ') # Replace the \n (new line) character with space
        ECallTxt = ECallTxt.replace('\r', '') # Replace the \r (carriage returns -if you're on windows) with null
        ECallTxt = ECallTxt.replace(' ', ' ') # Replace " " (a special character for space in HTML) with space. 
        ECallTxt = ECallTxt.replace(' ', ' ') # Replace " " (a special character for space in HTML) with space.
        while '  ' in ECallTxt:
            ECallTxt = ECallTxt.replace('  ', ' ') # Remove extra spaces
        
        # Parse document with SpaCy
        ECall = nlp(ECallTxt)
        
        ECallDoc = [] # Temporary list to store individual document
    
        # Further cleaning and selection of text characteristics
        for token in ECall:
            if token.is_stop == False and token.is_punct == False and (token.pos_ == "NOUN" or token.pos_ == "ADJ" or token.pos_ =="VERB"): # Retain words that are not a stop word nor punctuation, and only if a Noun, Adjective or Verb
                ECallDoc.append(token.lemma_.lower()) # Convert to lower case and retain the lemmatized version of the word (this is a string object)
        return ECallDoc   
    

lda_model = gensim.models.LdaModel.load('lda_model.model')
TFIDF = models.TfidfModel.load('tfidf_model.model')
ID2word = corpora.Dictionary.load('corpora_dict')

for index, topic in lda_model.print_topics(num_words=5):
    st.write('Topic {}: {}\n'.format(index+1,topic))

doc = st.text_input('Input your transcript here and press enter:')
doc_corpus = ID2word.doc2bow(preprocess(doc))
TFIDF_doc = TFIDF[doc_corpus]

for index, topic in lda_model.get_document_topics(TFIDF_doc):
    st.write('Topic {} Probability: {:.4f}\n'.format(index+1,topic))



