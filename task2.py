# Create a language model on Japanese language questions
# Input: Japanese Questions
# Output: LM based input data

import logging
import sys
import torch
from datasets import load_dataset

# setting logging config
logging.basicConfig(filename="bigram_lm.log",level=logging.INFO,
                    format='%(asctime)s-%(levelname)s-%(message)s')
sys.stdout=open('bigram_lm.log','a')
sys.stderr=sys.stdout

# device checker
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS (Apple GPU) is available.")
else:
    device = torch.device("cpu")
    print("MPS is not available, using CPU.")

def ds_loader(path):
    '''Takes in dataset and returns the train and validation split'''

    df_train=load_dataset(path,split='train').to_pandas()
    df_val=load_dataset(path,split='validation').to_pandas()

    return df_val, df_train

def get_question_data(path, lang):
    '''Gets the question data of the language selected'''
    df_train,df_val= ds_loader(path)

    train_d= vars()[f'df_train_{lang}']=df_train[df_train['lang']==lang]
    train_q= train_d['question']
    val_d= vars()[f'df_val_{lang}']=df_val[df_val['lang']==lang]
    val_q= val_d['question']

    return train_q, val_q


def train_bigram(data): 
    '''Creates a bigram language model trained on given data'''
    return 

def test_bigram(data):
    '''Evaluates bigram model on validation data'''






