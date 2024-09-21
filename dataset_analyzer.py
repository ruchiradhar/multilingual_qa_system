# Create function to analyse a dataset
# Input Data: Data Output: Analysis
# Dataset: https://huggingface.co/datasets/coastalcph/tydi_xor_rc

# importing dependencies
import logging
import sys
import torch
import pandas as pd
from datasets import load_dataset
import nltk
from collections import Counter

# setting logging config
logging.basicConfig(filename="dataset_analyzer.log",level=logging.INFO,
                    format='%(asctime)s-%(levelname)s-%(message)s')
sys.stdout=open('dataset_analyzer.log','a')
sys.stderr=sys.stdout

# device checker
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS (Apple GPU) is available.")
else:
    device = torch.device("cpu")
    print("MPS is not available, using CPU.")

# function to analyze dataset and give results
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
    # get column information
    for df in dfs:
        col_list= df.columns.tolist()
        print("Number of columns:",len(col_list))
        print("Columns:", col_list)

    # get language information 
    langs= df_train['lang'].unique()
    print("The languages in dataset are:", langs)
    df_train_langs=[]
    for lang in langs:
        vars()[f'df_train_{lang}']=df_train[df_train['lang']==lang]
        df_train_langs.append(vars()[f'df_train_{lang}']) 
    for item in zip(langs,df_train_langs):
        print("The len of", item[0],"is", len(item[1]))
    
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
        print("The most common words in",item[0], "are",most_common )

            
    return df_train, df_val

# end function

if __name__ == "__main__":
    path = 'coastalcph/tydi_xor_rc'
    ds_analyzer(path)