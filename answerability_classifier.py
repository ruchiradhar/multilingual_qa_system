# Create function to analyse answerability of questions in dataset
# Input Data: Data Output: Analysis
# Dataset: https://huggingface.co/datasets/coastalcph/tydi_xor_rc

# importing dependencies
import logging
import sys
import requests
import pandas as pd
from datasets import load_dataset
import nltk
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

# setting logging config
logging.basicConfig(filename="ans_classifier.log",level=logging.INFO,
                    format='%(asctime)s-%(levelname)s-%(message)s')
sys.stdout=open('ans_classifier.log','a')
sys.stderr=sys.stdout

# device checker
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS (Apple GPU) is available.")
else:
    device = torch.device("cpu")
    print("MPS is not available, using CPU.")

def ds_analyzer(path):
    '''Takes in link to a huggingface dataset and gives details of the dataset.
    Common details inlcude len, number of columns, columns
    For each language, we calculate most common words in questions. 
    For 3 langua answerability of questions.
    
    Output: Training Set, Validation Set'''

    #loading dataset
    dfs=[]
    df_train=load_dataset(path,split='train').to_pandas()
    dfs.append(df_train)
    df_val=load_dataset(path,split='validation').to_pandas()
    dfs.append(df_val)

    # get length of datasets
    print("The length of train set is", len(df_train))
    print("The length of val set is", len(df_val))

    # get language information 
    langs= df_train['lang'].unique()
    df_train_langs=[]
    for lang in langs:
        vars()[f'df_train_{lang}']=df_train[df_train['lang']==lang]
        df_train_langs.append(vars()[f'df_train_{lang}']) 
    
    # get most common word for fi, ru, ja
    def most_common_word (string, num_words):
        words=nltk.word_tokenize(string.lower())
        words= [w for w in words if w.isalpha()]
        word_counts=Counter(words)
        most_common = word_counts.most_common(num_words)

        return most_common
    
    for item in zip(langs,df_train_langs): 
        df_lang= item[1]
        qtext=''
        for i in range (len(df_lang)):
            q=df_lang.iloc[i]['question']
            qtext+=str(q)
        most_common=most_common_word(qtext, 5)
    
    return df_train, df_val


# implement rule-based classifier
def answerability_classifier(lang, dataframe):
    '''Takes in questions and contexts to determine whether it is answerable.'''
    # setting models
    pipe_ja = pipeline("translation", model="Helsinki-NLP/opus-mt-ja-en")
    pipe_ru = pipeline("translation", model="Helsinki-NLP/opus-mt-ru-en")
    pipe_fi = pipeline("translation", model="Helsinki-NLP/opus-mt-fi-en")
    ner_tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    ner_model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    ner_model.to(device)
    ner_nlp = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer)
        
    answerability_rb=[]
    for i in range (len(dataframe)): #finding named entities in question and context
        q=str(dataframe.iloc[i]['question'])
        if lang=="ja":
            q=pipe_ja(q)
            q=q[0]['translation_text']
        if lang=="ru":
            q=pipe_ru(q)
            q=q[0]['translation_text']
        if lang=="fi":
            q=pipe_fi(q)
            q=q[0]['translation_text']
        ner_q = ner_nlp(q)
        c=str(dataframe.iloc[i]['context'])
        ner_c = ner_nlp(c)
            
        ner_q_words=[]
        ner_c_words=[]
        for i in range(len(ner_q)):
            ner_q_words.append(ner_q[i]['word'])
        for i in range(len(ner_c)):
            ner_c_words.append(ner_c[i]['word'])
        if bool(set(ner_q_words) & set (ner_c_words))==True: #comparing common entities
            answerability_rb.append(True)
        else:
            answerability_rb.append(False)
    
    print("The classifier is ready")
        
    return answerability_rb
    
# calculating accuracy of classifier on validation set
if __name__ == "__main__":
    path = 'coastalcph/tydi_xor_rc'
    df_train,df_val= ds_analyzer(path)
    langs= ['ja', 'ru', 'fi']
    df_val_langs=[]
    for lang in langs:
        vars()[f'df_val_{lang}']=df_val[df_val['lang']==lang]
        df_val_langs.append(vars()[f'df_val_{lang}'])
    for item in zip(langs,df_val_langs):
        l=item[0]
        d=item[1]
        answerability_rb= answerability_classifier(l, d)
        accuracies=[]
        for i in range(len(d)):
            a_new=answerability_rb[i]
            a_ground= item[1].iloc[i]['answerable']
            if a_new==a_ground: 
                accuracies.append (True)
            else: 
                accuracies.append (False)
        acc= (accuracies.count(True)/len(accuracies))*100
        print("Accuracy for ", item[0], "is",acc )
