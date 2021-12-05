import pandas as pd
import numpy as np
import spacy
import gensim
import gensim.corpora as corpora
from gensim import models
from os import listdir

nlp = spacy.load("en_core_web_sm")

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""

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
    


# Name of the folder containing the files
folder_path = "test-transcripts"

# Get a list of filenames
filenames = listdir(folder_path)

ECallDocuments = [] # List to store all documents in the training corpus as a 'list of lists'

# For each file
for filename in filenames:
    # Create the filepath
    file_path = f"{folder_path}/{filename}"

    # Open the file (using "with" for file opening will autoclose the file at the end. It's a good practice)
    with open(file_path, "r") as f:
        # Get the file content
        ECallTxt = f.read()

        ECallDoc = preprocess(ECallTxt)

         # Append the content to the list
        ECallDocuments.append(ECallDoc) # Build the training corpus 'list of lists'

### NUMERIC REPRESENTATION OF TRAINING CORPUS USING BAG OF WORDS AND TF-IDF ###

# Form dictionary by mapping word IDs to words
ID2word = corpora.Dictionary(ECallDocuments)

# Set up Bag of Words and TFIDF
corpus = [ID2word.doc2bow(doc) for doc in ECallDocuments] # Apply Bag of Words to all documents in training corpus
TFIDF = models.TfidfModel(corpus) # Fit TF-IDF model
trans_TFIDF = TFIDF[corpus] # Apply TF-IDF model

### SET UP & TRAIN LDA MODEL ###

SEED = 75 # Set random seed
NUM_topics = 3 # Set number of topics
ALPHA = 0.9 # Set alpha
ETA = 0.35 # Set eta
# Train LDA model on the training corpus
lda_model = gensim.models.LdaModel(corpus=trans_TFIDF, num_topics=NUM_topics, id2word=ID2word, random_state=SEED, alpha=ALPHA, eta=ETA, passes=100)

lda_model.save('lda_model.model')
TFIDF.save('tfidf_model.model')
ID2word.save('corpora_dict')


