## Basic Details

Central Project: Build a multilingual question answering system (MQA)

Dataset: TyDi QA (Clark et al., 2020), Cross-lingual Open-Retrieval Question Answering dataset (XOR RC; Asai et al., 2021),XOR-AttriQA dataset (Muller et al., 2023)

Dataset Details: Some of the questions in them are unanswerable (impossible to
answer given just the provided context); the context documents and answers are
in English, but the questions are in other languages (Arabic, Bengali, Finnish,
Japanese, Korean, Russian and Telugu). 

## Task Segmentation

- Task 1: There are three parts to this task-  explore the dataset, find five most common words in each language, and implement a rule-based classifier to take in question & context and output answerable/unanswerable. Languages used: Finnish, Japanese, Russian

- Task 2: There are two parts to this task- train language models for questions in each language and for contexts in English. Then evaluate each of the models on the validation data and report their performance. Languages used: Finnish, Japanese, Russian.

- Task 3: There are two parts to this task- for each language train a classifier to take question & context and predict whether question is answerable. Then evaluate the classifiers on the validation sets. Languages used: Finnish, Japanese, Russian.

- Task 4: There are two parts to this task-  or each language train a sequence labelling model which predicts the tokens in a document that constitute the answer. Then evlauate the models using a sequence labelling metric where the correct output must be an empty list with no tokens. Languages used: Finnish, Japanese, Russian.

- Task 5: There are two parts to this task- first pretrain/finetune a transformer model to take in question & context and answer the question. Then also prepare a transformer model that only takes question as input and outputs the answer. In both cases, the model should answer in the question language only. Languages used: Finnish, Japanese, Russian. 